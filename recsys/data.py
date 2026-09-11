"""Загрузка исходных данных и разбиение событий по времени.

События берутся из таблицы retailrocket_events в Postgres, каталог товаров —
из parquet-файла в S3. Разбиение по датам сделано одной функцией, чтобы одни
и те же границы использовали ноутбук, стадии DVC и шаги DAG.

Запуск: используется как библиотека (from recsys.data import load_events)
"""

import logging

import pandas as pd
from sqlalchemy import create_engine, text

from recsys.config import ITEMS_PATH, postgres_credentials
from recsys.s3_io import read_parquet

logger = logging.getLogger(__name__)

TABLE_NAME = "retailrocket_events"

EVENTS_SQL = f"""
select event_ts, visitorid, event, itemid, transactionid
from {TABLE_NAME}
where (:start is null or event_ts >= cast(:start as timestamp))
  and (:end is null or event_ts < cast(:end as timestamp))
order by event_ts
"""


def get_engine():
    """
    Собирает SQLAlchemy engine к учебной Postgres из переменных окружения
    """
    p = postgres_credentials
    url = (
        f"postgresql+psycopg2://{p['user']}:{p['password']}"
        f"@{p['host']}:{p['port']}/{p['dbname']}"
    )
    return create_engine(url, connect_args={"sslmode": "require"})


def load_events(start=None, end=None) -> pd.DataFrame:
    """
    Читает события из Postgres за полуинтервал [start, end);
    любая из границ может быть None — тогда она не ограничивает выборку
    """
    engine = get_engine()
    params = {
        "start": None if start is None else str(pd.Timestamp(start)),
        "end": None if end is None else str(pd.Timestamp(end)),
    }
    # text() + именованные параметры: одинаково работает на SQLAlchemy 1.4 и 2.0,
    # в образе Airflow стоит 1.4, в venv проекта — 2.0
    with engine.connect() as conn:
        events = pd.read_sql(text(EVENTS_SQL), conn, params=params)
    events["event_ts"] = pd.to_datetime(events["event_ts"])
    logger.info("загружено событий: %d", len(events))
    return events


def load_items() -> pd.DataFrame:
    """
    Читает каталог товаров items.parquet из S3
    """
    return read_parquet(ITEMS_PATH)


def split_events(events: pd.DataFrame, dates: dict) -> dict:
    """
    Режет события на окна train_fit / valid / labels / test по границам из
    params.yaml; верхняя граница везде строгая, чтобы событие попало ровно
    в одно окно. train = train_fit + valid
    """
    ts = events["event_ts"]
    b = {name: pd.Timestamp(value) for name, value in dates.items()}

    train_fit = events[(ts >= b["split_train_fit"]) & (ts < b["split_valid"])]
    valid = events[(ts >= b["split_valid"]) & (ts < b["split_labels"])]
    labels = events[(ts >= b["split_labels"]) & (ts < b["split_test"])]
    test = events[(ts >= b["split_test"]) & (ts < b["split_end"])]
    train = events[(ts >= b["split_train_fit"]) & (ts < b["split_labels"])]

    parts = {
        "train_fit": train_fit,
        "valid": valid,
        "labels": labels,
        "test": test,
        "train": train,
    }
    for name, part in parts.items():
        logger.info("%s: %d событий", name, len(part))
    return parts
