"""Сервис рекомендаций товаров.

По идентификатору пользователя возвращает список рекомендованных товаров:
- офлайн-рекомендации — заранее посчитанные на этапе 3 (итоговые, после
  ранжирования; если пользователя там нет — топ популярных);
- онлайн-рекомендации — товары, похожие на последние события пользователя
  (события сервис принимает сам и держит в памяти);
- итоговые рекомендации — смешивание двух списков: онлайн-рекомендации
  на нечётных местах, офлайн — на чётных.

В отличие от учебной схемы из трёх микросервисов всё живёт в одном
процессе: блендинг дёргает хранилища напрямую, без HTTP-запросов
к самому себе. Метрики Prometheus объявлены в recsys_app.metrics
и отдаются на /metrics.

Запуск: uvicorn recsys_app.main:app --port 8010
Перед запуском нужно скачать артефакты: python download_recommendations.py
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from recsys_app import metrics
from recsys_app.constants import (
    DEFAULT_K,
    DEFAULT_RECS_PATH,
    MAX_EVENTS_PER_USER,
    MAX_K,
    N_ONLINE_EVENTS,
    PERSONAL_RECS_PATH,
    SIMILAR_ITEMS_PATH,
)
from recsys_app.stores import EventStore, Recommendations, SimilarItems

logger = logging.getLogger("uvicorn.error")

rec_store = Recommendations()
sim_items_store = SimilarItems()
events_store = EventStore(max_events_per_user=MAX_EVENTS_PER_USER)


class EventIn(BaseModel):
    """
    Событие пользователя: просмотр товара или добавление его в корзину
    """

    user_id: int = Field(..., ge=1)
    item_id: int = Field(..., ge=1)


def dedup_ids(ids):
    """
    Дедублицирует список идентификаторов, оставляя только первое вхождение
    """
    seen = set()
    ids = [id for id in ids if not (id in seen or seen.add(id))]

    return ids


def note_source(source: str):
    """
    Отмечает источник выдачи в метриках; выдача из топа популярных
    означает, что персональных рекомендаций у пользователя нет,
    то есть это холодный старт
    """
    metrics.recs_source_total.labels(source=source).inc()
    if source == "default":
        metrics.cold_start_total.inc()


def offline_recs(user_id: int, k: int):
    """
    Возвращает офлайн-рекомендации пользователя и источник выдачи
    """

    return rec_store.get(user_id, k)


def online_recs(user_id: int, k: int):
    """
    Строит онлайн-рекомендации по последним событиям пользователя:
    собирает похожие товары для каждого события и сортирует их по скору
    """
    events = events_store.get(user_id, N_ONLINE_EVENTS)

    items = []
    scores = []
    for item_id in events:
        i2i = sim_items_store.get(item_id, k)
        items += i2i["sim_item_id"]
        scores += i2i["score"]

    # сортируем похожие товары по убыванию скора
    combined = sorted(zip(items, scores), key=lambda pair: pair[1], reverse=True)
    combined = [item_id for item_id, _ in combined]

    # удаляем дубликаты, чтобы не выдавать одинаковые рекомендации
    return dedup_ids(combined)[:k]


def blend(recs_online, recs_offline, k: int):
    """
    Смешивает онлайн- и офлайн-рекомендации: онлайн на нечётных местах,
    офлайн на чётных, затем дедупликация и обрезка до k
    """
    recs_blended = []

    min_length = min(len(recs_online), len(recs_offline))
    for i in range(min_length):
        recs_blended.append(recs_online[i])
        recs_blended.append(recs_offline[i])

    # добавляем оставшиеся элементы в конец
    recs_blended += recs_online[min_length:]
    recs_blended += recs_offline[min_length:]

    return dedup_ids(recs_blended)[:k]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # код ниже (до yield) выполнится только один раз при запуске сервиса
    rec_store.load("personal", PERSONAL_RECS_PATH)
    rec_store.load("default", DEFAULT_RECS_PATH)
    sim_items_store.load(SIMILAR_ITEMS_PATH)
    metrics.model_loaded.set(1)
    logger.info("Ready!")
    yield
    # этот код выполнится только один раз при остановке сервиса
    metrics.model_loaded.set(0)
    rec_store.stats()
    sim_items_store.stats()
    events_store.stats()


# создаём приложение FastAPI
app = FastAPI(title="recsys", lifespan=lifespan)

# инструментатор добавляет стандартные HTTP-метрики и поднимает /metrics,
# куда попадают и собственные метрики из recsys_app.metrics
Instrumentator().instrument(app).expose(app)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """
    Отдаёт 422 на некорректный запрос и отмечает его в метриках:
    иначе такие запросы не попали бы в recsys_requests_total,
    потому что до обработчика эндпоинта дело не доходит
    """
    endpoint = request.url.path
    if endpoint == "/events":
        # то же разведение по методу, что и в самих обработчиках /events
        endpoint = f"{request.method} {endpoint}"
    metrics.requests_total.labels(endpoint=endpoint, status="422").inc()
    logger.warning(f"Invalid request to {endpoint}: {exc.errors()}")

    return JSONResponse(
        status_code=422, content={"detail": jsonable_encoder(exc.errors())}
    )


# /health намеренно не считаем в recsys_requests_total: пробы ходят сюда
# постоянно и размыли бы долю холодных стартов, которая делится на общее
# число запросов; сами пробы видны в метриках инструментатора
@app.get("/health")
async def health():
    """
    Возвращает состояние сервиса и признак загруженных артефактов
    """
    model_loaded = int(rec_store.is_loaded() and sim_items_store.is_loaded())

    return {"status": "ok", "model_loaded": model_loaded}


@app.get("/recommendations")
async def recommendations(
    user_id: int = Query(..., ge=1),
    k: int = Query(DEFAULT_K, ge=1, le=MAX_K),
):
    """
    Возвращает список рекомендаций длиной k для пользователя user_id,
    смешивая онлайн- и офлайн-рекомендации
    """
    with metrics.track("/recommendations"):
        try:
            recs_offline, source = offline_recs(user_id, k)
            recs_online = online_recs(user_id, k)
            recs = blend(recs_online, recs_offline, k)

            note_source(source)
            if recs_online:
                metrics.recs_source_total.labels(source="online").inc()
            metrics.recs_returned.observe(len(recs))

            logger.info(
                f"/recommendations user_id={user_id} k={k} source={source} "
                f"online={len(recs_online)} returned={len(recs)}"
            )
        except Exception:
            logger.exception(f"Failed to blend recommendations, user_id={user_id}")
            raise

        return {"recs": recs}


@app.get("/recommendations_offline")
async def recommendations_offline(
    user_id: int = Query(..., ge=1),
    k: int = Query(DEFAULT_K, ge=1, le=MAX_K),
):
    """
    Возвращает список офлайн-рекомендаций длиной k для пользователя user_id
    """
    with metrics.track("/recommendations_offline"):
        try:
            recs, source = offline_recs(user_id, k)

            note_source(source)
            metrics.recs_returned.observe(len(recs))

            logger.info(
                f"/recommendations_offline user_id={user_id} k={k} "
                f"source={source} returned={len(recs)}"
            )
        except Exception:
            logger.exception(f"Failed to get offline recs, user_id={user_id}")
            raise

        return {"recs": recs}


@app.get("/similar_items")
async def similar_items(
    item_id: int = Query(..., ge=1),
    k: int = Query(DEFAULT_K, ge=1, le=MAX_K),
):
    """
    Возвращает список похожих товаров длиной k для товара item_id
    """
    with metrics.track("/similar_items"):
        try:
            i2i = sim_items_store.get(item_id, k)

            metrics.recs_returned.observe(len(i2i["sim_item_id"]))
            logger.info(
                f"/similar_items item_id={item_id} k={k} "
                f"returned={len(i2i['sim_item_id'])}"
            )
        except Exception:
            logger.exception(f"Failed to get similar items, item_id={item_id}")
            raise

        return i2i


# у /events два метода, поэтому в метриках разводим их по способу вызова
@app.post("/events")
async def put_event(event: EventIn):
    """
    Сохраняет событие пользователя в память сервиса
    """
    with metrics.track("POST /events"):
        events_store.put(event.user_id, event.item_id)
        metrics.events_stored_total.inc()

        logger.info(f"POST /events user_id={event.user_id} item_id={event.item_id}")

        return {"result": "ok"}


@app.get("/events")
async def get_events(
    user_id: int = Query(..., ge=1),
    k: int = Query(DEFAULT_K, ge=1, le=MAX_K),
):
    """
    Возвращает список последних k событий пользователя user_id
    """
    with metrics.track("GET /events"):
        events = events_store.get(user_id, k)

        logger.info(f"GET /events user_id={user_id} k={k} returned={len(events)}")

        return {"events": events}
