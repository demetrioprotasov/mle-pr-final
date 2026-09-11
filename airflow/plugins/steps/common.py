"""Общее для шагов DAG: параметры, каталог прогона, обёртки над recsys.

Здесь лежит только клей между Airflow и пакетом recsys: чтение params.yml,
каталог с промежуточными файлами прогона, настройка MLflow и несколько
коротких функций, которые в ноутбуке были локальными (перевод выдачи ALS
в сырые идентификаторы, купленные пары в кодах матрицы, популярность товара,
сборка пула кандидатов). Содержательная логика — в пакете recsys.

Запуск: используется как библиотека (from steps.common import load_params)
"""

import logging
import os
import socket
from urllib.parse import urlparse

import mlflow
import pandas as pd
import yaml

from recsys.features import (
    build_candidates,
    decode,
    encode,
    ranker_features,
    user_history_purchases,
)
from recsys.models import als_recommend

logger = logging.getLogger(__name__)

PARAMS_PATH = "/opt/airflow/params.yml"
# каталог смонтирован из airflow/data, он в .gitignore проекта
DATA_DIR = "/opt/airflow/data"


def load_params() -> dict:
    """
    Читает параметры дообучения из params.yml
    """
    with open(PARAMS_PATH) as fd:
        return yaml.safe_load(fd)


def run_dir(run_id: str) -> str:
    """
    Возвращает каталог прогона под промежуточные файлы: между собой шаги
    передают только пути, сами таблицы лежат на диске
    """
    # в run_id есть двоеточия и плюс (manual__2026-09-11T18:00:00+00:00),
    # в имени каталога оставляем только буквы, цифры, дефис и подчёркивание
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_id)
    path = os.path.join(DATA_DIR, safe)
    os.makedirs(path, exist_ok=True)
    return path


def tracking_uri() -> str:
    """
    Адрес сервера MLflow с именем хоста, заменённым на его ip-адрес
    """
    uri = urlparse(os.environ["MLFLOW_TRACKING_URI"])
    # сервер MLflow 3.x защищается от DNS rebinding: он отдаёт 403 на любой
    # заголовок Host, кроме localhost и адресов из частных диапазонов.
    # host.docker.internal под это правило не подходит, а его ip — да
    address = socket.gethostbyname(uri.hostname)
    return f"{uri.scheme}://{address}:{uri.port}"


def setup_mlflow(params: dict) -> None:
    """
    Настраивает адрес сервера MLflow и эксперимент проекта
    """
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(params["mlflow_experiment"])


def item_popularity(events: pd.DataFrame) -> pd.Series:
    """
    Доля визитёров, взаимодействовавших с товаром, — нужна для novelty
    """
    return (
        events.groupby("itemid")["visitorid"].nunique()
        / events["visitorid"].nunique()
    )


def bought_enc(events: pd.DataFrame, id_maps: pd.DataFrame) -> pd.DataFrame:
    """
    Пары (user_enc, item_enc) с покупкой в истории в кодах матрицы
    """
    bought = user_history_purchases(events)
    enc = pd.DataFrame(
        {
            "user_enc": encode(id_maps, "user", bought["visitorid"]),
            "item_enc": encode(id_maps, "item", bought["itemid"]),
        }
    )
    return enc[(enc["user_enc"] >= 0) & (enc["item_enc"] >= 0)]


def als_recs(model, matrix, id_maps, users, bought, k) -> pd.DataFrame:
    """
    Рекомендации ALS для списка сырых visitorid, ответ тоже в сырых id
    """
    enc_users = encode(id_maps, "user", users)
    recs = als_recommend(model, matrix, enc_users, n=k, exclude=bought)
    return pd.DataFrame(
        {
            "visitorid": decode(id_maps, "user", recs["user_enc"]),
            "itemid": decode(id_maps, "item", recs["item_enc"]),
            "score": recs["score"].to_numpy(),
            "rank": recs["rank"].to_numpy(),
        }
    )


def candidate_pool(source: dict, users, n_als: int, n_pop: int) -> pd.DataFrame:
    """
    Пул кандидатов первой стадии: выдача ALS длиной n_als плюс топ
    популярных длиной n_pop, купленное из пула убирается
    """
    als_part = als_recs(
        source["als"],
        source["matrix"],
        source["id_maps"],
        users,
        source["bought"],
        n_als,
    )
    return build_candidates(
        als_part, source["top_pop"].head(n_pop), users, source["bought_raw"]
    )


def pool_features(candidates: pd.DataFrame, feats: dict) -> pd.DataFrame:
    """
    Приклеивает к пулу кандидатов признаки товара, пользователя и пары
    """
    return ranker_features(candidates, feats["item"], feats["user"], feats["cat"])
