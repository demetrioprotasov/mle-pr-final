"""Шаг train_models: обучение ALS и ранжировщика, оценка на тесте, MLflow.

Схема повторяет этапы 3 и 4 ноутбука: ALS учится на истории окна, пул
кандидатов собирается из выдачи ALS и топа популярных, таргет берётся из
окна labels, ранжировщик CatBoost переупорядочивает пул. Качество связки
меряется на окне test той же функцией recsys.metrics.evaluate, что и в
ноутбуке, и вместе с параметрами и моделью уходит в MLflow в прогон
retrain_<конец интервала>.

Запуск: вызывается задачей train_models DAG recsys_retrain
"""

import logging
import os

import mlflow
import numpy as np
import pandas as pd
import scipy.sparse
from mlflow.models import infer_signature
from sklearn.metrics import roc_auc_score

from recsys.config import SEED
from recsys.data import load_items
from recsys.features import (
    CAT_FEATURES,
    RANKER_FEATURES,
    add_target,
    holdout_users,
    item_features,
    positives,
    sample_negatives,
    user_category_affinity,
    user_features,
    user_history_purchases,
)
from recsys.metrics import evaluate
from recsys.models import (
    fit_als,
    fit_ranker,
    rank_candidates,
    save_ranker,
    top_by_score,
    top_popular,
)
from steps.common import (
    bought_enc,
    candidate_pool,
    item_popularity,
    load_params,
    pool_features,
    setup_mlflow,
)

logger = logging.getLogger(__name__)


def train_models(prepared: dict) -> dict:
    """
    Обучает ALS и ранжировщик на истории окна, считает метрики на тесте
    и пишет прогон в MLflow; возвращает метрики и ссылку на модель
    """
    params = load_params()
    candidates = params["candidates"]
    top_k = params["top_k"]

    history = pd.read_parquet(prepared["history"])
    labels = pd.read_parquet(prepared["labels"])
    test = pd.read_parquet(prepared["test"])
    matrix = scipy.sparse.load_npz(prepared["matrix"])
    id_maps = pd.read_parquet(prepared["id_maps"])
    items = load_items()

    als = fit_als(matrix, seed=SEED, **params["als"])
    source = {
        "als": als,
        "matrix": matrix,
        "id_maps": id_maps,
        "top_pop": top_popular(history, k=params["n_candidates"]),
        "bought": bought_enc(history, id_maps),
        "bought_raw": user_history_purchases(history),
    }
    feats = {
        "item": item_features(history, items),
        "user": user_features(history),
        "cat": user_category_affinity(history, items),
    }

    # учим ранжировщик на тех, у кого есть история и есть позитив в labels
    pos_labels = positives(labels)
    base_rank = np.intersect1d(
        pos_labels["visitorid"].unique(), history["visitorid"].unique()
    )
    pool = candidate_pool(
        source, base_rank, candidates["als_train"], candidates["popular_train"]
    )
    pool = add_target(pool, pos_labels)
    sampled = sample_negatives(pool, candidates["negatives_per_user"], SEED)

    X_rank = pool_features(sampled, feats)
    fit_users, hold_users = holdout_users(
        base_rank, candidates["holdout_share"], SEED
    )
    X_fit = X_rank[X_rank["visitorid"].isin(fit_users)]
    X_hold = X_rank[X_rank["visitorid"].isin(hold_users)]
    logger.info(
        "выборка ранжировщика: обучение %d строк, отложено %d строк",
        len(X_fit),
        len(X_hold),
    )

    ranker = fit_ranker(
        X_fit[RANKER_FEATURES],
        X_fit["target"],
        cat_features=CAT_FEATURES,
        eval_set=(X_hold[RANKER_FEATURES], X_hold["target"]),
        seed=SEED,
        **params["catboost"],
    )
    ranker_path = os.path.join(
        os.path.dirname(prepared["events"]), "catboost_ranker.cbm"
    )
    save_ranker(ranker, ranker_path)

    quality = {"trees": float(ranker.tree_count_)}
    for name, part in [("train", X_fit), ("holdout", X_hold)]:
        # AUC считается, только если в части есть оба класса
        if part["target"].nunique() > 1:
            predicted = rank_candidates(ranker, part[RANKER_FEATURES])
            quality[f"{name}_auc"] = float(roc_auc_score(part["target"], predicted))

    # оценка на тесте: база усреднения — визитёры с позитивом в test,
    # у которых есть история, как и в ноутбуке
    pos_test = positives(test)
    base_test = np.intersect1d(
        pos_test["visitorid"].unique(), history["visitorid"].unique()
    )
    pool_test = candidate_pool(
        source, base_test, candidates["als_train"], candidates["popular_train"]
    )
    X_test = pool_features(pool_test, feats)
    X_test["cb_score"] = rank_candidates(ranker, X_test[RANKER_FEATURES])
    recs_test = top_by_score(
        X_test[["visitorid", "itemid", "cb_score"]], top_k, "cb_score"
    )
    metrics = evaluate(
        recs_test,
        pos_test,
        top_k,
        matrix.shape[1],
        item_popularity(history),
        base_test,
    )

    setup_mlflow(params)
    with mlflow.start_run(run_name=f"retrain_{prepared['end']}") as run:
        mlflow.log_params(
            {
                "model": "als_candidates_plus_catboost",
                "window_start": prepared["start"],
                "window_end": prepared["end"],
                "train_window_days": params["train_window_days"],
                "labels_days": params["labels_days"],
                "test_days": params["test_days"],
                "als_factors": params["als"]["factors"],
                "als_regularization": params["als"]["regularization"],
                "als_iterations": params["als"]["iterations"],
                "candidates_als": candidates["als_train"],
                "candidates_popular": candidates["popular_train"],
                "negatives_per_user": candidates["negatives_per_user"],
                "features": ",".join(RANKER_FEATURES),
                "seed": SEED,
                **params["catboost"],
            }
        )
        mlflow.log_metrics(
            {
                "events_window": len(history) + len(labels) + len(test),
                "events_history": len(history),
                "users_rank": len(base_rank),
                "users_test": len(base_test),
                "train_rows": len(X_fit),
                "holdout_rows": len(X_hold),
                "matrix_users": matrix.shape[0],
                "matrix_items": matrix.shape[1],
                **quality,
                **{f"test_{name}_{top_k}": value for name, value in metrics.items()},
            }
        )
        # важность признаков — метриками, чтобы не тянуть в образ matplotlib
        mlflow.log_metrics(
            {
                f"fi_{name}": float(value)
                for name, value in zip(
                    RANKER_FEATURES, ranker.get_feature_importance()
                )
            }
        )

        example = X_fit[RANKER_FEATURES].head(10)
        signature = infer_signature(example, rank_candidates(ranker, example))
        model_info = mlflow.catboost.log_model(
            ranker,
            name="ranker",
            signature=signature,
            input_example=example,
            # список задан явно: вывод зависимостей запускает pip в подпроцессе,
            # а регистрируем мы модель отдельным шагом и только при успехе
            pip_requirements=["mlflow==3.15.1", "catboost==1.2.8"],
        )
        run_id = run.info.run_id

    logger.info("прогон MLflow %s: recall@%d = %.4f", run_id, top_k, metrics["recall"])
    return {
        "mlflow_run_id": run_id,
        "model_uri": model_info.model_uri,
        "ranker": ranker_path,
        "recall": float(metrics["recall"]),
        "users_test": int(len(base_test)),
    }
