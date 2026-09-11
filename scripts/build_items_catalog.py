"""Сборка каталога товаров items.parquet из item_properties и category_tree.

Из item_properties_part1/2.csv читаются только свойства categoryid и
available — остальные анонимизированы числовыми id и не расшифровываются.
Для каждой пары (itemid, property) берётся последнее значение с
timestamp < SPLIT_DATE_TRAIN (2015-08-04, граница train_fit из плана
сплитов) — иначе в признаки подглядывает будущее. К результату
присоединяется parentid категории из category_tree.csv. Каталог строится
на всех itemid, встречающихся в events.csv, даже если у товара нет ни
одного свойства (тогда categoryid/available остаются пустыми). Итог
пишется в data/items.parquet и заливается в S3 по ключу
recsys_final/data/items.parquet.

Запуск: python scripts/build_items_catalog.py --data-dir data
"""

import argparse
import logging
import os
import sys

import pandas as pd

# скрипт запускается из корня репозитория (python scripts/build_items_catalog.py),
# добавляем корень в sys.path, чтобы был виден пакет recsys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recsys.config import ITEMS_PATH
from recsys.s3_io import upload_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SPLIT_DATE_TRAIN = pd.Timestamp("2015-08-04")
PROPERTIES_TO_KEEP = ("categoryid", "available")
CHUNK_SIZE = 1_000_000


def read_event_item_ids(events_path: str) -> set:
    """
    Возвращает множество itemid, встречающихся в events.csv
    """
    item_ids = set()
    for chunk in pd.read_csv(events_path, usecols=["itemid"], chunksize=CHUNK_SIZE):
        item_ids.update(chunk["itemid"].unique().tolist())
    return item_ids


def read_last_property_values(path: str) -> pd.DataFrame:
    """
    Читает item_properties чанками, оставляет только categoryid/available
    и записи до SPLIT_DATE_TRAIN
    """
    parts = []
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        chunk = chunk[chunk["property"].isin(PROPERTIES_TO_KEEP)].copy()
        if chunk.empty:
            continue
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], unit="ms")
        chunk = chunk[chunk["timestamp"] < SPLIT_DATE_TRAIN]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=["timestamp", "itemid", "property", "value"])
    return pd.concat(parts, ignore_index=True)


def build_catalog(event_item_ids: set, part1_path: str, part2_path: str) -> pd.DataFrame:
    """
    Собирает последнее значение categoryid/available на каждый itemid
    из events.csv (для товаров без свойств значения остаются пустыми)
    """
    props = pd.concat(
        [read_last_property_values(part1_path), read_last_property_values(part2_path)],
        ignore_index=True,
    )
    # для каждой пары (itemid, property) оставляем последнюю запись по времени
    props = props.sort_values("timestamp").drop_duplicates(
        subset=["itemid", "property"], keep="last"
    )

    pivoted = props.pivot(index="itemid", columns="property", values="value")
    pivoted = pivoted.rename_axis(None, axis=1).reset_index()
    for col in PROPERTIES_TO_KEEP:
        if col not in pivoted.columns:
            pivoted[col] = pd.NA

    items = pd.DataFrame({"itemid": sorted(event_item_ids)})
    items = items.merge(pivoted, how="left", on="itemid")
    for col in PROPERTIES_TO_KEEP:
        items[col] = pd.to_numeric(items[col], errors="coerce").astype("Int64")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="сборка каталога товаров items.parquet")
    parser.add_argument("--data-dir", default="data", help="каталог с исходными csv")
    args = parser.parse_args()

    events_path = os.path.join(args.data_dir, "events.csv")
    category_tree_path = os.path.join(args.data_dir, "category_tree.csv")
    part1_path = os.path.join(args.data_dir, "item_properties_part1.csv")
    part2_path = os.path.join(args.data_dir, "item_properties_part2.csv")

    logger.info("читаем itemid из events.csv")
    event_item_ids = read_event_item_ids(events_path)
    logger.info("уникальных itemid в events.csv: %d", len(event_item_ids))

    logger.info(
        "читаем item_properties (part1, part2), свойства %s, cutoff %s",
        PROPERTIES_TO_KEEP,
        SPLIT_DATE_TRAIN.date(),
    )
    items = build_catalog(event_item_ids, part1_path, part2_path)

    category_tree = pd.read_csv(category_tree_path)
    items = items.merge(category_tree, how="left", on="categoryid")

    with_category = items["categoryid"].notna().mean()
    logger.info("товаров в каталоге: %d", len(items))
    logger.info("доля товаров events с известной категорией: %.4f", with_category)

    os.makedirs(args.data_dir, exist_ok=True)
    out_path = os.path.join(args.data_dir, "items.parquet")
    items.to_parquet(out_path, index=False)
    logger.info("сохранено %s (%d строк)", out_path, len(items))

    upload_file(out_path, ITEMS_PATH)


if __name__ == "__main__":
    main()
