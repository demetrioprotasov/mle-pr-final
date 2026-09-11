"""Метрики качества рекомендаций на k позициях.

Считаются precision@k, recall@k, MAP@k, NDCG@k, coverage@k и novelty@k.
Релевантными считаются пары (visitorid, itemid), по которым в окне оценки
было добавление в корзину или покупка. База усреднения задаётся явно
параметром users и одинакова для всех сравниваемых моделей.

Запуск: используется как библиотека (from recsys.metrics import evaluate)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def discount_sums(k: int) -> np.ndarray:
    """
    Накопленные суммы 1 / log2(i + 1) по позициям — из них берётся IDCG@k
    для пользователя с m релевантными товарами: discount_sums(k)[m]
    """
    discount = 1.0 / np.log2(np.arange(1, k + 1) + 1)
    return np.concatenate([[0.0], np.cumsum(discount)])


def per_user_scores(recs: pd.DataFrame, positives: pd.DataFrame, k: int):
    """
    Возвращает по каждому пользователю число попаданий в топ-k, сумму
    членов AP@k и DCG@k, а также саму урезанную до k позиций выдачу
    """
    top = recs[recs["rank"] <= k][["visitorid", "itemid", "rank"]]
    top = top.sort_values(["visitorid", "rank"])

    marked = top.merge(
        positives[["visitorid", "itemid"]].drop_duplicates().assign(rel=1.0),
        on=["visitorid", "itemid"],
        how="left",
    )
    marked["rel"] = marked["rel"].fillna(0.0)

    by_user = marked.groupby("visitorid", sort=False)
    # precision@i считается нарастающим итогом по позициям внутри пользователя
    marked["prec_at_i"] = by_user["rel"].cumsum() / marked["rank"]
    marked["ap_term"] = marked["prec_at_i"] * marked["rel"]
    marked["dcg_term"] = marked["rel"] / np.log2(marked["rank"] + 1)

    scores = marked.groupby("visitorid").agg(
        hits=("rel", "sum"),
        ap_sum=("ap_term", "sum"),
        dcg=("dcg_term", "sum"),
    )
    return scores, top


def evaluate(
    recs: pd.DataFrame,
    positives: pd.DataFrame,
    k: int = 10,
    catalog_size: int = None,
    item_pop: pd.Series = None,
    users=None,
) -> dict:
    """
    Считает шесть метрик по рекомендациям recs (колонки visitorid, itemid,
    rank) относительно релевантных пар positives. В users передаётся база
    усреднения — список пользователей, по которым усредняются метрики
    """
    base = pd.Index(
        positives["visitorid"].unique() if users is None else np.unique(users),
        name="visitorid",
    )
    n_rel = (
        positives[positives["visitorid"].isin(base)]
        .groupby("visitorid")
        .size()
        .rename("n_rel")
    )

    scores, top = per_user_scores(recs, positives, k)
    df = pd.DataFrame(index=base).join(scores).join(n_rel)
    df[["hits", "ap_sum", "dcg"]] = df[["hits", "ap_sum", "dcg"]].fillna(0.0)
    df["n_rel"] = df["n_rel"].fillna(0).astype("int64")

    # AP@k и NDCG@k нормируются на идеальный случай: min(число релевантных, k)
    m = df["n_rel"].clip(upper=k).to_numpy()
    idcg = discount_sums(k)[m]
    ap = np.where(m > 0, df["ap_sum"].to_numpy() / np.maximum(m, 1), 0.0)
    ndcg = np.where(idcg > 0, df["dcg"].to_numpy() / np.where(idcg > 0, idcg, 1), 0.0)

    result = {
        "precision": float((df["hits"] / k).mean()),
        "recall": float((df["hits"] / df["n_rel"].clip(lower=1)).mean()),
        "map": float(np.mean(ap)),
        "ndcg": float(np.mean(ndcg)),
    }

    if catalog_size:
        result["coverage"] = float(top["itemid"].nunique() / catalog_size)
    if item_pop is not None:
        # novelty — средняя собственная информация -log2(p) рекомендованного
        # товара: чем реже товар встречался в обучении, тем он «новее»
        self_info = -np.log2(item_pop.clip(lower=1e-9))
        info = top["itemid"].map(self_info)
        result["novelty"] = float(info.mean())

    logger.info("метрики@%d: %s", k, {n: round(v, 4) for n, v in result.items()})
    return result
