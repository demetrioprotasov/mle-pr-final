"""Шаг extract_events: выборка событий из Postgres за интервал дообучения.

События читаются хуком PostgresHook по соединению destination_db тем же
запросом, что и в recsys.data. Конец интервала берётся из контекста Airflow
(data_interval_end) и ограничивается сверху параметром data_max_date:
датасет Retailrocket заморожен на сентябре 2015, поэтому ручной запуск
сегодня даёт ровно то же окно, что и запуск по расписанию. Начало окна —
на train_window_days раньше конца. В реальной эксплуатации ограничение
сверху убирается.

Запуск: вызывается задачей extract_events DAG recsys_retrain
"""

import logging
import os

import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook

from recsys.data import TABLE_NAME
from steps.common import load_params, run_dir

logger = logging.getLogger(__name__)

# тот же запрос, что в recsys.data.load_events, но с плейсхолдерами psycopg2:
# пакет читает события через SQLAlchemy, а в образе Airflow стоит SQLAlchemy
# 1.4 (её требует сам Airflow), и pandas 2.2 такое соединение не принимает.
# в DAG данные берутся хуком курса, подключение — AIRFLOW_CONN_DESTINATION_DB
EVENTS_SQL = f"""
select event_ts, visitorid, event, itemid, transactionid
from {TABLE_NAME}
where event_ts >= %(start)s and event_ts < %(end)s
order by event_ts
"""


def load_events(start, end) -> pd.DataFrame:
    """
    Читает события из учебной Postgres за полуинтервал [start, end)
    """
    hook = PostgresHook("destination_db")
    conn = hook.get_conn()
    try:
        events = pd.read_sql(
            EVENTS_SQL, conn, params={"start": str(start), "end": str(end)}
        )
    finally:
        conn.close()
    events["event_ts"] = pd.to_datetime(events["event_ts"])
    return events


def extract_events(**kwargs) -> dict:
    """
    Читает события за полуинтервал [start, end) и складывает их в parquet
    каталога прогона; возвращает путь к файлу и границы окна
    """
    params = load_params()
    nominal_end = pd.Timestamp(kwargs["data_interval_end"].date())
    max_date = pd.Timestamp(params["data_max_date"])

    # верхняя граница строгая: событие ровно на границе в окно не попадает
    end = min(nominal_end, max_date)
    start = end - pd.Timedelta(days=params["train_window_days"])
    logger.info("номинальный конец интервала: %s", nominal_end.date())
    logger.info(
        "фактическое окно: %s — %s (%d дней)",
        start.date(),
        end.date(),
        params["train_window_days"],
    )

    events = load_events(start, end)
    path = os.path.join(run_dir(kwargs["run_id"]), "events.parquet")
    events.to_parquet(path, index=False)

    logger.info(
        "событий в окне: %d, визитёров %d, товаров %d",
        len(events),
        events["visitorid"].nunique(),
        events["itemid"].nunique(),
    )
    logger.info("события сохранены в %s", path)
    return {"events": path, "start": str(start.date()), "end": str(end.date())}
