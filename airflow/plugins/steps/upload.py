"""Шаг upload_recs: пересчёт рекомендаций на всём окне и выгрузка в S3.

Перед выкаткой ALS переобучается на всём окне прогона (история + labels +
test) с теми же гиперпараметрами, ранжировщик берётся тот, что обучен шагом
train_models. Считаются три артефакта, которые читает сервис: персональные
рекомендации, похожие товары и топ популярных. Модель ALS в бакет не
заливается: она весит сотни мегабайт, а канал наружу узкий.

Запуск: вызывается задачей upload_recs DAG recsys_retrain
"""

import logging
import os
import time

import mlflow
import numpy as np
import pandas as pd

from recsys.config import (
    RECS_DEFAULT_PATH,
    RECS_PERSONAL_PATH,
    SEED,
    SIMILAR_PATH,
)
from recsys.data import load_items
from recsys.features import (
    RANKER_FEATURES,
    build_matrix,
    decode,
    item_features,
    user_category_affinity,
    user_features,
    user_history_purchases,
    weight_events,
)
from recsys.models import (
    fit_als,
    load_ranker,
    rank_candidates,
    similar_items,
    top_by_score,
    top_popular,
)
from recsys.s3_io import upload_file
from steps.common import (
    bought_enc,
    candidate_pool,
    load_params,
    pool_features,
    setup_mlflow,
)

logger = logging.getLogger(__name__)


def personal_recommendations(source, feats, ranker, users, params) -> pd.DataFrame:
    """
    Считает персональные рекомендации пачками пользователей: пул кандидатов
    на всех сразу не помещается в память контейнера
    """
    candidates = params["candidates"]
    chunk = params["serving_chunk_users"]
    chunks = []
    for start in range(0, len(users), chunk):
        part_users = users[start:start + chunk]
        pool = candidate_pool(
            source,
            part_users,
            candidates["als_serving"],
            candidates["popular_serving"],
        )
        scored = pool_features(pool, feats)
        scored["score"] = rank_candidates(ranker, scored[RANKER_FEATURES])
        chunks.append(
            top_by_score(
                scored[["visitorid", "itemid", "score"]], params["recs_top_n"]
            )
        )
        done = start + len(part_users)
        logger.info("посчитано пользователей: %d из %d", done, len(users))

    recs = pd.concat(chunks, ignore_index=True).rename(
        columns={"visitorid": "user_id", "itemid": "item_id"}
    )
    return recs.astype(
        {"user_id": "int64", "item_id": "int64", "score": "float32", "rank": "int16"}
    )


def upload_recs(prepared: dict, trained: dict) -> dict:
    """
    Переобучает ALS на всём окне, считает выдачу для сервиса и заливает
    три parquet-файла в S3; возвращает размеры и время заливок
    """
    params = load_params()
    events = pd.read_parquet(prepared["events"])
    items = load_items()

    pairs = weight_events(events, params["event_weights"])
    matrix, id_maps = build_matrix(pairs)
    als = fit_als(matrix, seed=SEED, **params["als"])
    top_pop = top_popular(events, k=params["n_candidates"])

    sim = similar_items(als, np.arange(matrix.shape[1]), n=params["n_similar"])
    similar = pd.DataFrame(
        {
            "itemid": decode(id_maps, "item", sim["item_enc"]),
            "similar_itemid": decode(id_maps, "item", sim["similar_enc"]),
            "score": sim["score"].to_numpy(),
            "rank": sim["rank"].to_numpy(),
        }
    )

    source = {
        "als": als,
        "matrix": matrix,
        "id_maps": id_maps,
        "top_pop": top_pop,
        "bought": bought_enc(events, id_maps),
        "bought_raw": user_history_purchases(events),
    }
    feats = {
        "item": item_features(events, items),
        "user": user_features(events),
        "cat": user_category_affinity(events, items),
    }

    counts = events.groupby("visitorid").size()
    serve_users = np.sort(counts[counts >= params["min_user_events"]].index.to_numpy())
    logger.info(
        "пользователей с персональными рекомендациями: %d из %d",
        len(serve_users),
        events["visitorid"].nunique(),
    )
    recommendations = personal_recommendations(
        source, feats, load_ranker(trained["ranker"]), serve_users, params
    )

    directory = os.path.dirname(prepared["events"])
    recs_path = os.path.join(directory, "recommendations.parquet")
    similar_path = os.path.join(directory, "similar_items.parquet")
    top_pop_path = os.path.join(directory, "top_popular.parquet")
    # zstd вместо snappy: тот же файл выходит меньше, а при 0,1 МБ/с
    # каждый мегабайт экономит десяток секунд заливки
    recommendations.to_parquet(recs_path, index=False, compression="zstd")
    similar.to_parquet(similar_path, index=False)
    top_pop.to_parquet(top_pop_path, index=False)

    uploads = []
    for local_path, key in [
        (recs_path, RECS_PERSONAL_PATH),
        (similar_path, SIMILAR_PATH),
        (top_pop_path, RECS_DEFAULT_PATH),
    ]:
        # ключи зашиты в recsys.config — их же читает сервис; префикс из
        # params.yml проверяем, чтобы каталоги в бакете не разъехались
        if not key.startswith(params["s3_prefix"] + "/"):
            raise ValueError(f"ключ {key} вне префикса {params['s3_prefix']}")
        size_mb = os.path.getsize(local_path) / 1e6
        started = time.time()
        upload_file(local_path, key)
        spent = time.time() - started
        uploads.append(
            {"key": key, "size_mb": round(size_mb, 1), "seconds": round(spent, 1)}
        )
        logger.info("залито %s: %.1f МБ за %.1f с", key, size_mb, spent)

    setup_mlflow(params)
    with mlflow.start_run(run_id=trained["mlflow_run_id"]):
        mlflow.log_metrics(
            {
                "serve_users": len(serve_users),
                "recs_rows": len(recommendations),
                "recs_items": recommendations["item_id"].nunique(),
                "similar_rows": len(similar),
                **{
                    f"upload_seconds_{index}": upload["seconds"]
                    for index, upload in enumerate(uploads)
                },
            }
        )
    return {"serve_users": int(len(serve_users)), "uploads": uploads}
