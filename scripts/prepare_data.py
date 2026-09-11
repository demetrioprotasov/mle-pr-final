"""Стадия DVC prepare_data: события из Postgres, каталог из S3, сплит по датам.

Читает все события Retailrocket и режет их на окна train / labels / test
границами из params.yaml (recsys.data.split_events); train = train_fit +
valid, окна подбора ALS на этой стадии не нужны — гиперпараметры уже
зафиксированы в params.yaml (als_best). Результат — parquet-файлы,
которые читают стадии train и evaluate.

Запуск: python scripts/prepare_data.py
"""

import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recsys.data import load_events, load_items, split_events

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "data"

SPLIT_KEYS = [
    "split_train_fit",
    "split_valid",
    "split_labels",
    "split_test",
    "split_end",
]


def main() -> None:
    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)

    dates = {key: params[key] for key in SPLIT_KEYS}

    events = load_events()
    parts = split_events(events, dates)
    items = load_items()

    os.makedirs(DATA_DIR, exist_ok=True)
    parts["train"].to_parquet(f"{DATA_DIR}/events_train.parquet", index=False)
    parts["labels"].to_parquet(f"{DATA_DIR}/events_labels.parquet", index=False)
    parts["test"].to_parquet(f"{DATA_DIR}/events_test.parquet", index=False)
    items.to_parquet(f"{DATA_DIR}/items.parquet", index=False)
    logger.info(
        "сохранено: train %d, labels %d, test %d, товаров %d",
        len(parts["train"]),
        len(parts["labels"]),
        len(parts["test"]),
        len(items),
    )


if __name__ == "__main__":
    main()
