"""Модели рекомендаций: топ популярных, ALS, похожие товары, ранжировщик.

Топ популярных считается по числу добавлений в корзину — это целевое
действие кейса. ALS берётся из implicit и обучается на матрице весов
из recsys.features. Модель сохраняется нативным npz (не pickle), потому
что файл читают и ноутбук, и шаги DAG в другом образе.

Вторая стадия — CatBoostClassifier, который переупорядочивает пул
кандидатов по вероятности добавления в корзину; он сохраняется в cbm,
тоже без pickle.

Запуск: используется как библиотека (from recsys.models import fit_als)
"""

import logging

import numpy as np
import pandas as pd
import threadpoolctl
from catboost import CatBoostClassifier
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


def fit_ranker(
    features,
    target,
    cat_features=None,
    eval_set=None,
    iterations=500,
    learning_rate=0.1,
    depth=6,
    seed=SEED,
):
    """
    Обучает ранжировщик второй стадии — бинарный классификатор CatBoost
    на пуле кандидатов. Если передан eval_set с отложенными
    пользователями, обучение останавливается по нему
    """
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="Logloss",
        random_seed=seed,
        verbose=100,
        cat_features=cat_features,
        # иначе CatBoost создаёт рядом с ноутбуком каталог catboost_info
        allow_writing_files=False,
    )
    model.fit(
        features,
        target,
        eval_set=eval_set,
        early_stopping_rounds=50 if eval_set is not None else None,
        use_best_model=eval_set is not None,
    )
    logger.info("ранжировщик обучен: деревьев %d", model.tree_count_)
    return model


def rank_candidates(model, features) -> np.ndarray:
    """
    Вероятность положительного класса для каждой строки пула кандидатов
    """
    return model.predict_proba(features)[:, 1].astype("float32")


def top_by_score(recs: pd.DataFrame, k: int, score_col: str = "score") -> pd.DataFrame:
    """
    Оставляет k лучших позиций на пользователя по указанному скору и
    проставляет rank; пустые скоры уходят в конец списка
    """
    ordered = recs.sort_values(["visitorid", score_col], ascending=[True, False])
    ordered["rank"] = ordered.groupby("visitorid").cumcount() + 1
    return ordered[ordered["rank"] <= k].reset_index(drop=True)


def save_ranker(model, path: str) -> None:
    """
    Сохраняет ранжировщик в формате cbm
    """
    model.save_model(path)
    logger.info("ранжировщик сохранён в %s", path)


def load_ranker(path: str):
    """
    Загружает ранжировщик из cbm
    """
    model = CatBoostClassifier()
    model.load_model(path)
    return model
