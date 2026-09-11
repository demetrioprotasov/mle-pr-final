"""Подготовка данных для моделей: веса событий, user-item матрица, признаки.

Сырой вес пары (visitorid, itemid) — сумма весов её событий, значение в
матрице — log1p от этой суммы: медиана событий на визитёра равна 1, но
максимум доходит до нескольких тысяч, логарифм гасит вклад накрутчиков.
Соответствие сырых и закодированных идентификаторов хранится parquet-таблицей,
а не pickle-ом: venv проекта на Python 3.12, образ Airflow — на 3.11,
pickle между разными версиями scikit-learn ненадёжен.

Вторая половина модуля — двухстадийная схема: сборка пула кандидатов из
выдач ALS и топа популярных, разметка таргета по окну labels и признаки
ранжировщика. Все признаки считаются по истории, переданной аргументом,
поэтому одна и та же функция годится и для обучения, и для выкатки.

Запуск: используется как библиотека (from recsys.features import build_matrix)
"""

import logging

import numpy as np
import pandas as pd
import scipy.sparse

from recsys.config import SEED

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


# --- двухстадийная схема: кандидаты и признаки ранжировщика ---

RANKER_FEATURES = [
    "als_score",
    "als_rank",
    "pop_score",
    "pop_rank",
    "item_views",
    "item_addtocarts",
    "item_cr",
    "item_age_days",
    "categoryid",
    "user_events_cnt",
    "user_addtocart_cnt",
    "user_recency_days",
    "user_cat_affinity",
]

# CatBoost работает с категориальным признаком напрямую, кодировать не нужно
CAT_FEATURES = ["categoryid"]

NUMERIC_FILL = [
    "item_views",
    "item_addtocarts",
    "item_cr",
    "item_age_days",
    "user_events_cnt",
    "user_addtocart_cnt",
    "user_recency_days",
]


def item_features(history: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """
    Признаки товара по истории: просмотры, добавления в корзину, конверсия
    из просмотра в корзину, возраст в днях и категория из каталога
    """
    end_ts = history["event_ts"].max()
    counts = (
        history.groupby(["itemid", "event"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["view", "addtocart", "transaction"], fill_value=0)
    )
    feats = pd.DataFrame(index=counts.index)
    feats["item_views"] = counts["view"]
    feats["item_addtocarts"] = counts["addtocart"]
    # у товара, который попал в корзину без просмотра, знаменатель равен 1
    feats["item_cr"] = feats["item_addtocarts"] / feats["item_views"].clip(lower=1)
    first_seen = history.groupby("itemid")["event_ts"].min()
    feats["item_age_days"] = (end_ts - first_seen).dt.days

    category = items.set_index("itemid")["categoryid"]
    feats["categoryid"] = category.reindex(feats.index).fillna(-1).astype("int32")
    logger.info("признаки товаров: %d строк", len(feats))
    return feats.reset_index()


def user_features(history: pd.DataFrame) -> pd.DataFrame:
    """
    Признаки пользователя по истории: число событий, число добавлений
    в корзину и давность последнего события в днях
    """
    end_ts = history["event_ts"].max()
    by_user = history.groupby("visitorid")
    feats = pd.DataFrame(
        {
            "user_events_cnt": by_user.size(),
            "user_recency_days": (end_ts - by_user["event_ts"].max()).dt.days,
        }
    )
    carts = history[history["event"] == "addtocart"].groupby("visitorid").size()
    feats["user_addtocart_cnt"] = carts.reindex(feats.index).fillna(0).astype("int64")
    logger.info("признаки пользователей: %d строк", len(feats))
    return feats.reset_index()


def user_category_affinity(history: pd.DataFrame, items: pd.DataFrame):
    """
    Парный признак: доля событий пользователя, пришедшаяся на категорию
    товара. Товары без категории собраны в отдельную категорию -1
    """
    events = history[["visitorid", "itemid"]].merge(
        items[["itemid", "categoryid"]], on="itemid", how="left"
    )
    events["categoryid"] = events["categoryid"].fillna(-1).astype("int32")

    in_category = events.groupby(["visitorid", "categoryid"]).size().rename("in_cat")
    total = events.groupby("visitorid").size().rename("total")
    affinity = in_category.reset_index().merge(
        total.reset_index(), on="visitorid", how="left"
    )
    affinity["user_cat_affinity"] = (
        affinity["in_cat"] / affinity["total"]
    ).astype("float32")
    return affinity[["visitorid", "categoryid", "user_cat_affinity"]]


def build_candidates(
    als_recs: pd.DataFrame,
    popular: pd.DataFrame,
    users,
    bought: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Собирает пул кандидатов: выдача ALS плюс общий список популярного,
    объединённые по паре (visitorid, itemid). У кандидата, пришедшего
    только от одного генератора, скор второго остаётся пустым
    """
    users = np.asarray(users)
    als = als_recs.rename(columns={"score": "als_score", "rank": "als_rank"})
    items = popular["itemid"].to_numpy()
    pop = pd.DataFrame(
        {
            "visitorid": np.repeat(users, len(items)),
            "itemid": np.tile(items, len(users)),
            "pop_score": np.tile(popular["score"].to_numpy(), len(users)),
            "pop_rank": np.tile(popular["rank"].to_numpy(), len(users)),
        }
    )
    candidates = als.merge(pop, on=["visitorid", "itemid"], how="outer")

    if bought is not None and len(bought) > 0:
        marked = candidates.merge(
            bought[["visitorid", "itemid"]].drop_duplicates().assign(bought=1),
            on=["visitorid", "itemid"],
            how="left",
        )
        candidates = marked[marked["bought"].isna()].drop(columns="bought")

    candidates = candidates.reset_index(drop=True)
    logger.info(
        "кандидатов: %d на %d пользователей",
        len(candidates),
        candidates["visitorid"].nunique(),
    )
    return candidates


def add_target(candidates: pd.DataFrame, target_positives: pd.DataFrame):
    """
    Ставит кандидату target = 1, если в окне разметки по этой паре было
    добавление в корзину или покупка, иначе 0
    """
    pos = target_positives[["visitorid", "itemid"]].drop_duplicates().assign(target=1)
    marked = candidates.merge(pos, on=["visitorid", "itemid"], how="left")
    marked["target"] = marked["target"].fillna(0).astype("int8")
    return marked


def sample_negatives(candidates: pd.DataFrame, n_negatives: int, seed: int = SEED):
    """
    Оставляет все позитивы и не больше n_negatives случайных негативов
    на пользователя — иначе классы разъезжаются на два порядка
    """
    positive = candidates[candidates["target"] == 1]
    negative = candidates[candidates["target"] == 0].sample(frac=1.0, random_state=seed)
    negative = negative[negative.groupby("visitorid").cumcount() < n_negatives]

    sampled = pd.concat([positive, negative]).sort_values(["visitorid", "itemid"])
    logger.info(
        "обучающая выборка ранжировщика: %d строк, позитивов %d",
        len(sampled),
        len(positive),
    )
    return sampled.reset_index(drop=True)


def holdout_users(users, share: float, seed: int = SEED):
    """
    Делит пользователей на обучающих и отложенных: отложенные нужны
    ранжировщику для ранней остановки и честной проверки AUC
    """
    users = np.unique(np.asarray(users))
    rng = np.random.default_rng(seed)
    holdout = rng.choice(users, size=int(round(len(users) * share)), replace=False)
    return np.setdiff1d(users, holdout), np.sort(holdout)


def ranker_features(
    candidates: pd.DataFrame,
    item_feats: pd.DataFrame,
    user_feats: pd.DataFrame,
    cat_affinity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Приклеивает к пулу кандидатов признаки товара, пользователя и пары.
    Пустые скоры кандидатогенераторов остаются пустыми: CatBoost сам
    обрабатывает пропуски, а «нет кандидата от ALS» — это и есть сигнал
    """
    data = candidates.merge(item_feats, on="itemid", how="left")
    data = data.merge(user_feats, on="visitorid", how="left")
    data["categoryid"] = data["categoryid"].fillna(-1).astype("int32")
    data = data.merge(cat_affinity, on=["visitorid", "categoryid"], how="left")
    data["user_cat_affinity"] = data["user_cat_affinity"].fillna(0.0).astype("float32")

    for column in NUMERIC_FILL:
        data[column] = data[column].fillna(0).astype("float32")
    for column in ["als_score", "als_rank", "pop_score", "pop_rank"]:
        data[column] = data[column].astype("float32")
    return data
