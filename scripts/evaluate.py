"""Стадия DVC evaluate: топ-10 по трём моделям на одном пуле кандидатов теста.

Из пула кандидатов, посчитанного стадией train, строятся три выдачи топ-k:
топ популярных (по позиции в общем списке), ALS (по скору) и ранжировщик
(по скору CatBoost) — все три на одном и том же пуле и одной базе
пользователей, поэтому разница метрик объясняется только порядком
товаров. Метрики считает recsys.metrics.evaluate, результат — файл
cv_results/metrics.json, который DVC отслеживает как metrics пайплайна.

Запуск: python scripts/evaluate.py
"""

import json
import logging
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recsys.features import positives
from recsys.metrics import evaluate
from recsys.models import top_by_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "data"
RESULTS_DIR = "cv_results"


def item_popularity(events: pd.DataFrame) -> pd.Series:
    """
    Доля визитёров окна, взаимодействовавших с товаром — используется
    в качестве item_pop для novelty и не зависит от модели
    """
    return events.groupby("itemid")["visitorid"].nunique() / events["visitorid"].nunique()


def main() -> None:
    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)
    top_k = params["top_k"]

    candidates_test = pd.read_parquet(f"{DATA_DIR}/candidates_test.parquet")
    events_train = pd.read_parquet(f"{DATA_DIR}/events_train.parquet")
    events_test = pd.read_parquet(f"{DATA_DIR}/events_test.parquet")

    pos_test = positives(events_test)
    base_test = candidates_test["visitorid"].unique()
    catalog_train = events_train["itemid"].nunique()
    pop_train = item_popularity(events_train)

    # у популярных товаров скор совпадает при равном числе добавлений в
    # корзину, поэтому порядок задаём позицией в общем списке, а не скором
    pool_pop = candidates_test[["visitorid", "itemid", "pop_rank"]].copy()
    pool_pop["pop_order"] = -pool_pop["pop_rank"]

    recs_pool_pop = top_by_score(pool_pop, top_k, "pop_order")
    recs_pool_als = top_by_score(
        candidates_test[["visitorid", "itemid", "als_score"]], top_k, "als_score"
    )
    recs_ranker = top_by_score(
        candidates_test[["visitorid", "itemid", "cb_score"]], top_k, "cb_score"
    )

    metrics = {
        "top_k": top_k,
        "users_test": int(len(base_test)),
        "test": {
            "top_popular": evaluate(
                recs_pool_pop, pos_test, top_k, catalog_train, pop_train, base_test
            ),
            "als": evaluate(
                recs_pool_als, pos_test, top_k, catalog_train, pop_train, base_test
            ),
            "als_ranker": evaluate(
                recs_ranker, pos_test, top_k, catalog_train, pop_train, base_test
            ),
        },
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/metrics.json", "w", encoding="utf-8") as fd:
        json.dump(metrics, fd, indent=2, ensure_ascii=False)
    logger.info("метрики сохранены в %s/metrics.json", RESULTS_DIR)


if __name__ == "__main__":
    main()
