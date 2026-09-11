"""Стадия DVC train: ALS на окне train, кандидаты, ранжировщик, пул теста.

Повторяет этап 3-4 ноутбука `notebooks/recommendations.ipynb`, но без
подбора гиперпараметров ALS Optuna: als_best уже найден и зафиксирован в
params.yaml. Кандидаты для ранжировщика собираются из выдачи ALS и топа
популярных на окне train, таргет берётся из окна labels. Модель ALS и
ранжировщик сохраняются как out-файлы пайплайна, а оценённый пул кандидатов
теста — во входной файл стадии evaluate.

Запуск: python scripts/train.py
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recsys.features import (
    CAT_FEATURES,
    RANKER_FEATURES,
    add_target,
    build_candidates,
    build_matrix,
    decode,
    encode,
    holdout_users,
    item_features,
    positives,
    ranker_features,
    sample_negatives,
    user_category_affinity,
    user_features,
    user_history_purchases,
    weight_events,
)
from recsys.models import (
    als_recommend,
    fit_als,
    fit_ranker,
    rank_candidates,
    save_als,
    save_ranker,
    top_popular,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "data"
MODELS_DIR = "models"


def bought_enc(id_maps: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """
    Пары (user_enc, item_enc) с покупкой в истории — ALS их не рекомендует
    """
    enc = pd.DataFrame(
        {
            "user_enc": encode(id_maps, "user", history["visitorid"]),
            "item_enc": encode(id_maps, "item", history["itemid"]),
        }
    )
    return enc[(enc["user_enc"] >= 0) & (enc["item_enc"] >= 0)]


def als_recs(model, matrix, id_maps, users, bought, k) -> pd.DataFrame:
    """
    Рекомендации ALS для списка сырых visitorid, в выдаче — сырые id
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


def main() -> None:
    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)

    seed = params["seed"]
    als_best = params["als_best"]
    cand_cfg = params["candidates"]
    cb_cfg = params["catboost"]

    events_train = pd.read_parquet(f"{DATA_DIR}/events_train.parquet")
    events_labels = pd.read_parquet(f"{DATA_DIR}/events_labels.parquet")
    events_test = pd.read_parquet(f"{DATA_DIR}/events_test.parquet")
    items = pd.read_parquet(f"{DATA_DIR}/items.parquet")

    # --- ALS на окне train ---
    pairs_train = weight_events(events_train, params["event_weights"])
    matrix_train, id_maps_train = build_matrix(pairs_train)
    model = fit_als(
        matrix_train,
        als_best["factors"],
        als_best["regularization"],
        als_best["iterations"],
        seed,
    )
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_als(model, f"{MODELS_DIR}/als_model.npz")

    top_pop_train = top_popular(events_train, k=params["n_candidates"])
    bought_train_raw = user_history_purchases(events_train)
    bought_train_enc = bought_enc(id_maps_train, bought_train_raw)

    # --- пул кандидатов для обучения ранжировщика: таргет из окна labels ---
    pos_labels = positives(events_labels)
    base_rank = np.intersect1d(
        pos_labels["visitorid"].unique(), events_train["visitorid"].unique()
    )
    als_cand_rank = als_recs(
        model, matrix_train, id_maps_train, base_rank, bought_train_enc,
        cand_cfg["als_train"],
    )
    cand_rank = build_candidates(
        als_cand_rank, top_pop_train.head(cand_cfg["popular_train"]),
        base_rank, bought_train_raw,
    )
    cand_rank = add_target(cand_rank, pos_labels)
    train_rank = sample_negatives(cand_rank, cand_cfg["negatives_per_user"], seed)

    item_feats_train = item_features(events_train, items)
    user_feats_train = user_features(events_train)
    cat_aff_train = user_category_affinity(events_train, items)
    x_rank = ranker_features(
        train_rank, item_feats_train, user_feats_train, cat_aff_train
    )

    fit_users, hold_users = holdout_users(base_rank, cand_cfg["holdout_share"], seed)
    x_fit = x_rank[x_rank["visitorid"].isin(fit_users)]
    x_hold = x_rank[x_rank["visitorid"].isin(hold_users)]

    ranker = fit_ranker(
        x_fit[RANKER_FEATURES],
        x_fit["target"],
        cat_features=CAT_FEATURES,
        eval_set=(x_hold[RANKER_FEATURES], x_hold["target"]),
        iterations=cb_cfg["iterations"],
        learning_rate=cb_cfg["learning_rate"],
        depth=cb_cfg["depth"],
        seed=seed,
    )
    save_ranker(ranker, f"{MODELS_DIR}/catboost_ranker.cbm")

    fit_pred = rank_candidates(ranker, x_fit[RANKER_FEATURES])
    hold_pred = rank_candidates(ranker, x_hold[RANKER_FEATURES])
    logger.info(
        "ранжировщик: train_auc=%.4f holdout_auc=%.4f train_logloss=%.4f "
        "holdout_logloss=%.4f деревьев=%d",
        roc_auc_score(x_fit["target"], fit_pred),
        roc_auc_score(x_hold["target"], hold_pred),
        log_loss(x_fit["target"], fit_pred),
        log_loss(x_hold["target"], hold_pred),
        ranker.tree_count_,
    )

    # --- тот же пул кандидатов на тестовых пользователях, для стадии evaluate ---
    pos_test = positives(events_test)
    base_test = np.intersect1d(
        pos_test["visitorid"].unique(), events_train["visitorid"].unique()
    )
    als_cand_test = als_recs(
        model, matrix_train, id_maps_train, base_test, bought_train_enc,
        cand_cfg["als_train"],
    )
    cand_test = build_candidates(
        als_cand_test, top_pop_train.head(cand_cfg["popular_train"]),
        base_test, bought_train_raw,
    )
    x_test = ranker_features(cand_test, item_feats_train, user_feats_train, cat_aff_train)
    x_test["cb_score"] = rank_candidates(ranker, x_test[RANKER_FEATURES])

    candidates_test = x_test[["visitorid", "itemid", "als_score", "pop_rank", "cb_score"]]
    candidates_test.to_parquet(f"{DATA_DIR}/candidates_test.parquet", index=False)
    logger.info(
        "пул кандидатов теста: %d строк, %d пользователей",
        len(candidates_test),
        candidates_test["visitorid"].nunique(),
    )


if __name__ == "__main__":
    main()
