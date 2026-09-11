"""Шаг prepare_data: сплит окна по датам, веса событий и матрица.

Окно прогона режется на три части теми же правилами, что и в ноутбуке:
последние test_days — тест, перед ним labels_days — окно таргета для
ранжировщика, всё остальное — история, по которой учится ALS и считаются
признаки. Верхняя граница каждой части строгая, поэтому событие попадает
ровно в одну часть.

Запуск: вызывается задачей prepare_data DAG recsys_retrain
"""

import logging
import os

import pandas as pd
import scipy.sparse

from recsys.features import build_matrix, weight_events
from steps.common import load_params

logger = logging.getLogger(__name__)


def prepare_data(extracted: dict) -> dict:
    """
    Режет окно на историю, разметку и тест, считает веса событий и строит
    по истории матрицу пользователь-товар; возвращает пути к файлам
    """
    params = load_params()
    events = pd.read_parquet(extracted["events"])
    end = pd.Timestamp(extracted["end"])
    test_start = end - pd.Timedelta(days=params["test_days"])
    labels_start = test_start - pd.Timedelta(days=params["labels_days"])
    logger.info(
        "границы: история < %s, labels < %s, test < %s",
        labels_start.date(),
        test_start.date(),
        end.date(),
    )

    ts = events["event_ts"]
    parts = {
        "history": events[ts < labels_start],
        "labels": events[(ts >= labels_start) & (ts < test_start)],
        "test": events[(ts >= test_start) & (ts < end)],
    }

    directory = os.path.dirname(extracted["events"])
    paths = dict(extracted)
    for name, part in parts.items():
        paths[name] = os.path.join(directory, f"events_{name}.parquet")
        part.to_parquet(paths[name], index=False)
        logger.info(
            "%s: %d событий, визитёров %d",
            name,
            len(part),
            part["visitorid"].nunique(),
        )

    pairs = weight_events(parts["history"], params["event_weights"])
    matrix, id_maps = build_matrix(pairs)
    paths["matrix"] = os.path.join(directory, "matrix.npz")
    paths["id_maps"] = os.path.join(directory, "id_maps.parquet")
    scipy.sparse.save_npz(paths["matrix"], matrix)
    id_maps.to_parquet(paths["id_maps"], index=False)

    logger.info("матрица %s, ненулевых значений %d", matrix.shape, matrix.nnz)
    return paths
