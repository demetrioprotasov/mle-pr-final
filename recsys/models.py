"""Модели рекомендаций: топ популярных, ALS и похожие товары.

Топ популярных считается по числу добавлений в корзину — это целевое
действие кейса. ALS берётся из implicit и обучается на матрице весов
из recsys.features. Модель сохраняется нативным npz (не pickle), потому
что файл читают и ноутбук, и шаги DAG в другом образе.

Запуск: используется как библиотека (from recsys.models import fit_als)
"""

import logging

import numpy as np
import pandas as pd
import threadpoolctl
from implicit.cpu.als import AlternatingLeastSquares

from recsys.config import SEED

logger = logging.getLogger(__name__)

# 0 — по числу ядер машины; implicit сам распараллеливает ALS по пользователям
NUM_THREADS = 0


def top_popular(events: pd.DataFrame, k: int = 100) -> pd.DataFrame:
    """
    Считает топ популярных товаров по числу добавлений в корзину в окне
    обучения; score — доля визитёров окна, добавивших товар в корзину
    """
    carts = events[events["event"] == "addtocart"]
    n_users = events["visitorid"].nunique()
    pop = (
        carts.groupby("itemid")["visitorid"]
        .nunique()
        .rename("users")
        .reset_index()
        .sort_values("users", ascending=False)
        .head(k)
        .reset_index(drop=True)
    )
    pop["rank"] = pop.index + 1
    pop["score"] = (pop["users"] / n_users).astype("float32")
    logger.info("топ популярных: %d товаров из %d", len(pop), carts["itemid"].nunique())
    return pop[["itemid", "score", "rank"]]


def fit_als(matrix, factors=64, regularization=0.05, iterations=20, seed=SEED):
    """
    Обучает ALS из implicit на матрице весов пользователь-товар
    """
    # внутренний пул потоков OpenBLAS дерётся за ядра с распараллеливанием
    # самого ALS, поэтому на время обучения оставляем BLAS один поток
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        als = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=seed,
            num_threads=NUM_THREADS,
        )
        als.fit(matrix, show_progress=False)
    logger.info(
        "ALS обучен: factors=%d, regularization=%s, iterations=%d",
        factors,
        regularization,
        iterations,
    )
    return als


def als_recommend(model, matrix, user_enc_ids, n=10, exclude=None) -> pd.DataFrame:
    """
    Возвращает топ-n рекомендаций ALS для указанных пользователей в виде
    таблицы user_enc, item_enc, score, rank. В exclude передаются пары
    (user_enc, item_enc), которые пользователь уже купил, — их из выдачи
    убираем, а просмотренные и отложенные товары рекомендовать можно
    """
    user_enc_ids = np.asarray(user_enc_ids, dtype="int64")

    # запас позиций на выброшенные покупки: берём максимум покупок на
    # пользователя среди тех, кому считаем рекомендации
    pad = 0
    if exclude is not None and len(exclude) > 0:
        per_user = exclude[exclude["user_enc"].isin(user_enc_ids)]
        pad = int(per_user.groupby("user_enc").size().max()) if len(per_user) else 0
        pad = min(pad, 100)

    ids, scores = model.recommend(
        user_enc_ids,
        matrix[user_enc_ids],
        N=n + pad,
        filter_already_liked_items=False,
    )
    width = ids.shape[1]
    recs = pd.DataFrame(
        {
            "user_enc": np.repeat(user_enc_ids, width),
            "item_enc": ids.ravel().astype("int64"),
            "score": scores.ravel().astype("float32"),
        }
    )
    # implicit добивает выдачу значением -1, если кандидатов не хватило
    recs = recs[recs["item_enc"] >= 0]

    if pad > 0:
        marked = recs.merge(
            exclude.assign(bought=1), on=["user_enc", "item_enc"], how="left"
        )
        recs = marked[marked["bought"].isna()].drop(columns="bought")

    recs["rank"] = recs.groupby("user_enc").cumcount() + 1
    recs = recs[recs["rank"] <= n].reset_index(drop=True)
    return recs


def similar_items(model, item_enc_ids, n=10) -> pd.DataFrame:
    """
    Считает n похожих товаров для каждого товара по факторам ALS;
    запрашиваем n + 1, потому что первым в выдаче идёт сам товар
    """
    item_enc_ids = np.asarray(item_enc_ids, dtype="int64")
    ids, scores = model.similar_items(item_enc_ids, N=n + 1)
    width = ids.shape[1]

    sim = pd.DataFrame(
        {
            "item_enc": np.repeat(item_enc_ids, width),
            "similar_enc": ids.ravel().astype("int64"),
            "score": scores.ravel().astype("float32"),
        }
    )
    sim = sim[(sim["similar_enc"] >= 0) & (sim["item_enc"] != sim["similar_enc"])]
    sim["rank"] = sim.groupby("item_enc").cumcount() + 1
    sim = sim[sim["rank"] <= n].reset_index(drop=True)
    logger.info("похожих товаров: %d строк", len(sim))
    return sim


def save_als(model, path: str) -> None:
    """
    Сохраняет ALS в npz средствами implicit
    """
    model.save(path)
    logger.info("модель сохранена в %s", path)


def load_als(path: str):
    """
    Загружает ALS из npz
    """
    return AlternatingLeastSquares.load(path)
