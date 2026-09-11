"""Тестирование сервиса рекомендаций.

Проверяются сценарии:
1) сервис жив и артефакты загружены (/health);
2) пользователь с персональными рекомендациями получает именно их;
3) неизвестный пользователь получает топ популярных (холодный старт);
4) для популярного товара находятся похожие, скоры идут по убыванию;
5) события сохраняются и возвращаются (POST /events, GET /events);
6) после события выдача меняется — блендинг онлайн и офлайн работает;
7) негативные запросы (user_id=-1, k=1000) отклоняются с кодом 422.

Перед запуском сервис должен быть поднят (локально uvicorn или
docker compose), а артефакты скачаны в data/.

Запуск: python test_service.py
Вывод пишется в консоль и в файл test_service.log.
"""

import logging
import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

K = 10

service_url = f"http://127.0.0.1:{os.environ.get('MAIN_APP_PORT', '8010')}"
headers = {"Content-type": "application/json", "Accept": "application/json"}

PERSONAL_RECS_PATH = "data/recommendations.parquet"
DEFAULT_RECS_PATH = "data/top_popular.parquet"

# в артефактах этапа 3 колонки называются как в исходных данных Retailrocket,
# сервис приводит их к единым именам — здесь делаем то же самое
COLUMN_NAMES = {"visitorid": "user_id", "itemid": "item_id"}

# логируем и в консоль, и в файл test_service.log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_service.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

failed_checks = 0


def check(condition, description):
    global failed_checks
    if condition:
        logger.info(f"OK: {description}")
    else:
        failed_checks += 1
        logger.error(f"ОШИБКА: {description}")


def get_json(endpoint, params):
    """
    Дёргает GET-эндпоинт и возвращает разобранный ответ вместе со статусом
    """
    resp = requests.get(service_url + endpoint, headers=headers, params=params)
    if resp.status_code != 200:
        return resp.status_code, {}

    return resp.status_code, resp.json()


def get_recs(endpoint, user_id, k=K):
    """
    Возвращает список рекомендаций, отданный эндпоинтом
    """
    status, body = get_json(endpoint, {"user_id": user_id, "k": k})
    if status != 200:
        logger.error(f"{endpoint} вернул статус {status}")
        return []

    return body["recs"]


def put_event(user_id, item_id):
    """
    Отправляет событие пользователя в сервис
    """
    resp = requests.post(
        service_url + "/events",
        headers=headers,
        json={"user_id": user_id, "item_id": item_id},
    )
    check(resp.status_code == 200, f"событие ({user_id}, {item_id}) сохранено")


# подбираем пользователей и товары по тем же файлам, что читает сервис:
# один пользователь с персональными рекомендациями и один заведомо неизвестный
personal = pd.read_parquet(PERSONAL_RECS_PATH).rename(columns=COLUMN_NAMES)
top_popular = pd.read_parquet(DEFAULT_RECS_PATH).rename(columns=COLUMN_NAMES)

# в Retailrocket visitorid бывает и 0, а сервис принимает только user_id >= 1
# (см. main.py), поэтому берём первого подходящего пользователя из файла
USER_PERSONAL = int(personal.loc[personal["user_id"] >= 1, "user_id"].iloc[0])
USER_UNKNOWN = int(personal["user_id"].max()) + 1_000_000
top_k = top_popular.sort_values("rank")["item_id"].to_list()[:K]

# товар для онлайн-события берём из топа популярных: для него точно
# есть похожие товары в similar_items.parquet
EVENT_ITEM_ID = top_k[0]

# --- Тест 1. Сервис готов принимать запросы ---
logger.info("=" * 60)
logger.info("Тест 1. Проверка /health")

status, health = get_json("/health", {})
logger.info(f"ответ: {health}")

check(status == 200, "/health отвечает со статусом 200")
check(health.get("status") == "ok", "сервис сообщает status=ok")
check(health.get("model_loaded") == 1, "артефакты рекомендаций загружены")

# --- Тест 2. Пользователь с персональными рекомендациями ---
logger.info("=" * 60)
logger.info(f"Тест 2. Пользователь {USER_PERSONAL} с персональными рекомендациями")

recs_offline = get_recs("/recommendations_offline", USER_PERSONAL)
logger.info(f"офлайн-рекомендации: {recs_offline}")

expected = (
    personal[personal["user_id"] == USER_PERSONAL]
    .sort_values("rank")["item_id"]
    .to_list()[:K]
)

check(len(recs_offline) == K, f"получено {K} рекомендаций")
check(recs_offline == expected, "выдача совпадает с персональными из parquet")
check(recs_offline != top_k, "выдача не равна топу популярных")

# --- Тест 3. Неизвестный пользователь, холодный старт ---
logger.info("=" * 60)
logger.info(f"Тест 3. Неизвестный пользователь {USER_UNKNOWN}")

recs_cold = get_recs("/recommendations", USER_UNKNOWN)
logger.info(f"рекомендации: {recs_cold}")

check(len(recs_cold) == K, f"получено {K} рекомендаций")
check(recs_cold == top_k, "неизвестный пользователь получил топ популярных")

# --- Тест 4. Похожие товары ---
logger.info("=" * 60)
logger.info(f"Тест 4. Похожие товары для товара {EVENT_ITEM_ID}")

status, i2i = get_json("/similar_items", {"item_id": EVENT_ITEM_ID, "k": K})
logger.info(f"похожие товары: {i2i.get('sim_item_id')}")

scores = i2i.get("score", [])
check(len(i2i.get("sim_item_id", [])) == K, f"найдено {K} похожих товаров")
check(
    scores == sorted(scores, reverse=True),
    "скоры похожих товаров идут по убыванию",
)
check(EVENT_ITEM_ID not in i2i.get("sim_item_id", []), "сам товар в выдачу не попал")

# --- Тест 5. Сохранение и чтение событий ---
logger.info("=" * 60)
logger.info(f"Тест 5. События пользователя {USER_PERSONAL}")

logger.info(f"добавляем онлайн-событие: {EVENT_ITEM_ID}")
put_event(USER_PERSONAL, EVENT_ITEM_ID)

status, events = get_json("/events", {"user_id": USER_PERSONAL, "k": K})
logger.info(f"события пользователя: {events.get('events')}")

stored = events.get("events", [])
check(len(stored) > 0 and stored[0] == EVENT_ITEM_ID, "событие вернулось из хранилища")

# --- Тест 6. Блендинг онлайн и офлайн ---
logger.info("=" * 60)
logger.info(f"Тест 6. Выдача пользователя {USER_PERSONAL} после события")

recs_after = get_recs("/recommendations", USER_PERSONAL)
logger.info(f"рекомендации после события: {recs_after}")

check(len(recs_after) == K, f"получено {K} рекомендаций")
check(recs_after != recs_offline, "выдача отличается от офлайновой: онлайн учтён")
# при смешивании онлайн занимают нечётные места, офлайн — чётные
check(
    recs_after[0] in i2i.get("sim_item_id", []),
    "первое место занял онлайн-кандидат (похожий товар)",
)
check(recs_after[1] == recs_offline[0], "второе место занял офлайн-кандидат")

# --- Тест 7. Негативные сценарии ---
logger.info("=" * 60)
logger.info("Тест 7. Некорректные запросы")

status, _ = get_json("/recommendations", {"user_id": -1, "k": K})
check(status == 422, "user_id=-1 отклонён со статусом 422")

status, _ = get_json("/recommendations", {"user_id": USER_PERSONAL, "k": 1000})
check(status == 422, "k=1000 отклонён со статусом 422")

status, _ = get_json("/similar_items", {"item_id": 0, "k": K})
check(status == 422, "item_id=0 отклонён со статусом 422")

resp = requests.post(
    service_url + "/events", headers=headers, json={"user_id": "abc", "item_id": 1}
)
check(resp.status_code == 422, "событие с нечисловым user_id отклонено со статусом 422")

resp = requests.post(
    service_url + "/events", headers=headers, json={"user_id": -1, "item_id": 1}
)
check(resp.status_code == 422, "событие с user_id=-1 отклонено со статусом 422")

# --- Итог ---
logger.info("=" * 60)
if failed_checks == 0:
    logger.info("Все тесты пройдены")
else:
    logger.error(f"Провалено проверок: {failed_checks}")
    raise SystemExit(1)
