"""Метрики Prometheus сервиса рекомендаций.

Здесь объявлены все собственные метрики: трафик и ошибки по эндпоинтам,
задержка ответа, источник рекомендаций, длина выдачи, холодные старты,
принятые события и индикатор загруженных артефактов. Отдаются на /metrics
вместе со стандартными метриками prometheus_fastapi_instrumentator
и process_* из prometheus_client.
"""

import time
from contextlib import contextmanager

from prometheus_client import Counter, Gauge, Histogram

requests_total = Counter(
    "recsys_requests_total",
    "Запросы к эндпоинтам сервиса в разрезе статуса ответа",
    ["endpoint", "status"],
)

request_duration = Histogram(
    "recsys_request_duration_seconds",
    "Время обработки запроса эндпоинтом",
    ["endpoint"],
)

recs_source_total = Counter(
    "recsys_recs_source_total",
    "Источник выдачи: personal, default или online",
    ["source"],
)

recs_returned = Histogram(
    "recsys_recs_returned",
    "Длина выдачи рекомендаций",
    buckets=(0, 1, 3, 5, 10, 20, 50, 100),
)

cold_start_total = Counter(
    "recsys_cold_start_total",
    "Запросы по пользователям без персональных рекомендаций",
)

events_stored_total = Counter(
    "recsys_events_stored_total",
    "Принятые сервисом события пользователей",
)

model_loaded = Gauge(
    "recsys_model_loaded",
    "Загружены ли артефакты рекомендаций: 1 — готов, 0 — нет",
)


@contextmanager
def track(endpoint: str):
    """
    Замеряет длительность запроса и считает его в разрезе статуса:
    "200" при нормальном завершении и "500", если вылетело исключение
    """
    started = time.perf_counter()
    status = "200"
    try:
        yield
    except Exception:
        status = "500"
        raise
    finally:
        request_duration.labels(endpoint=endpoint).observe(
            time.perf_counter() - started
        )
        requests_total.labels(endpoint=endpoint, status=status).inc()
