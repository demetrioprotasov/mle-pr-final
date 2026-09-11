"""Подготовка данных для моделей: веса событий и user-item матрица.

Сырой вес пары (visitorid, itemid) — сумма весов её событий, значение в
матрице — log1p от этой суммы: медиана событий на визитёра равна 1, но
максимум доходит до нескольких тысяч, логарифм гасит вклад накрутчиков.
Соответствие сырых и закодированных идентификаторов хранится parquet-таблицей,
а не pickle-ом: venv проекта на Python 3.12, образ Airflow — на 3.11,
pickle между разными версиями scikit-learn ненадёжен.

Запуск: используется как библиотека (from recsys.features import build_matrix)
"""

import logging

import numpy as np
import pandas as pd
import scipy.sparse

logger = logging.getLogger(__name__)

EVENT_WEIGHTS = {"view": 1.0, "addtocart": 5.0, "transaction": 10.0}

# позитивы для оценки качества — только целевое действие и покупка
POSITIVE_EVENTS = ("addtocart", "transaction")


def weight_events(events: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """
    Сворачивает события в пары (visitorid, itemid) со значением log1p(сумма
    весов): view = 1, addtocart = 5, transaction = 10
    """
    weights = EVENT_WEIGHTS if weights is None else weights
    pairs = events[["visitorid", "itemid", "event"]].copy()
    pairs["weight"] = pairs["event"].map(weights).astype("float32")
    pairs = pairs.groupby(["visitorid", "itemid"], as_index=False)["weight"].sum()
    pairs["value"] = np.log1p(pairs["weight"]).astype("float32")
    logger.info("пар пользователь-товар: %d", len(pairs))
    return pairs


def build_matrix(pairs: pd.DataFrame):
    """
    Строит разреженную матрицу csr (пользователи x товары) и таблицу
    соответствия идентификаторов с колонками kind, raw_id, enc_id
    """
    user_ids = np.sort(pairs["visitorid"].unique())
    item_ids = np.sort(pairs["itemid"].unique())
    user_pos = pd.Series(np.arange(len(user_ids)), index=user_ids)
    item_pos = pd.Series(np.arange(len(item_ids)), index=item_ids)

    rows = user_pos.loc[pairs["visitorid"]].to_numpy()
    cols = item_pos.loc[pairs["itemid"]].to_numpy()
    matrix = scipy.sparse.csr_matrix(
        (pairs["value"].to_numpy(), (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float32,
    )

    id_maps = pd.concat(
        [
            pd.DataFrame(
                {
                    "kind": "user",
                    "raw_id": user_ids.astype("int64"),
                    "enc_id": np.arange(len(user_ids), dtype="int64"),
                }
            ),
            pd.DataFrame(
                {
                    "kind": "item",
                    "raw_id": item_ids.astype("int64"),
                    "enc_id": np.arange(len(item_ids), dtype="int64"),
                }
            ),
        ],
        ignore_index=True,
    )
    logger.info("матрица %s, ненулевых значений: %d", matrix.shape, matrix.nnz)
    return matrix, id_maps


def encode(id_maps: pd.DataFrame, kind: str, raw_ids) -> np.ndarray:
    """
    Переводит сырые visitorid/itemid в номера строк или столбцов матрицы;
    идентификаторы, которых нет в маппинге, получают -1
    """
    mapping = id_maps[id_maps["kind"] == kind].set_index("raw_id")["enc_id"]
    return mapping.reindex(np.asarray(raw_ids)).fillna(-1).to_numpy(dtype="int64")


def decode(id_maps: pd.DataFrame, kind: str, enc_ids) -> np.ndarray:
    """
    Обратный перевод: из номеров строк или столбцов матрицы в сырые
    visitorid/itemid
    """
    mapping = id_maps[id_maps["kind"] == kind].set_index("enc_id")["raw_id"]
    return mapping.reindex(np.asarray(enc_ids)).to_numpy(dtype="int64")


def user_history_purchases(events: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает пары (visitorid, itemid) с покупкой в истории — такие товары
    исключаются из рекомендаций: купленный холодильник второй раз не нужен
    """
    bought = events.loc[events["event"] == "transaction", ["visitorid", "itemid"]]
    bought = bought.drop_duplicates().reset_index(drop=True)
    logger.info("купленных пар в истории: %d", len(bought))
    return bought


def positives(events: pd.DataFrame) -> pd.DataFrame:
    """
    Пары (visitorid, itemid) с addtocart или transaction в окне оценки —
    это релевантные товары, по которым считаются метрики
    """
    mask = events["event"].isin(POSITIVE_EVENTS)
    pos = events.loc[mask, ["visitorid", "itemid"]].drop_duplicates()
    return pos.reset_index(drop=True)
