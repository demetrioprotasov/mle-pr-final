"""Заливка событий Retailrocket в Postgres.

Читает events.csv чанками (timestamp в мс от эпохи конвертируется в
timestamp), создаёт таблицу retailrocket_events, если её ещё нет, и
грузит данные через psycopg2 copy_expert (обычный to_sql на 2.76 млн
строк по сети идёт слишком долго). Повторный запуск с флагом
--truncate чистит таблицу перед загрузкой, чтобы не плодить дубли.

Запуск: python scripts/load_events_to_pg.py --data-dir data
"""

import argparse
import io
import logging
import os
import sys

import pandas as pd
import psycopg2

# скрипт запускается из корня репозитория (python scripts/load_events_to_pg.py),
# добавляем корень в sys.path, чтобы был виден пакет recsys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recsys.config import postgres_credentials

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TABLE_NAME = "retailrocket_events"
CHUNK_SIZE = 100_000

CREATE_TABLE_SQL = f"""
create table if not exists {TABLE_NAME} (
    event_id bigserial primary key,
    event_ts timestamp not null,
    visitorid bigint,
    event varchar(16),
    itemid bigint,
    transactionid bigint null
);
create index if not exists idx_events_ts on {TABLE_NAME} (event_ts);
"""

COPY_SQL = f"""
copy {TABLE_NAME} (event_ts, visitorid, event, itemid, transactionid)
from stdin with (format csv)
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="загрузка events.csv в Postgres")
    parser.add_argument("--data-dir", default="data", help="каталог с events.csv")
    parser.add_argument(
        "--truncate", action="store_true", help="очистить таблицу перед загрузкой"
    )
    return parser.parse_args()


def load_chunk(cur, chunk: pd.DataFrame) -> None:
    """
    Конвертирует timestamp в event_ts и грузит один чанк через copy_expert
    """
    chunk = chunk.copy()
    chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], unit="ms")
    buf = io.StringIO()
    chunk[["timestamp", "visitorid", "event", "itemid", "transactionid"]].to_csv(
        buf, index=False, header=False
    )
    buf.seek(0)
    cur.copy_expert(COPY_SQL, buf)


def main() -> None:
    args = parse_args()
    events_path = os.path.join(args.data_dir, "events.csv")

    conn = psycopg2.connect(**postgres_credentials, sslmode="require")
    try:
        cur = conn.cursor()
        cur.execute(CREATE_TABLE_SQL)
        if args.truncate:
            logger.info("очищаем таблицу %s перед загрузкой", TABLE_NAME)
            cur.execute(f"truncate table {TABLE_NAME} restart identity")
        conn.commit()

        total_rows = 0
        reader = pd.read_csv(
            events_path,
            dtype={
                "visitorid": "int64",
                "event": "string",
                "itemid": "int64",
                "transactionid": "Int64",
            },
            chunksize=CHUNK_SIZE,
        )
        for chunk in reader:
            load_chunk(cur, chunk)
            total_rows += len(chunk)
            conn.commit()
            logger.info("загружено %d строк (всего %d)", len(chunk), total_rows)

        logger.info("загрузка завершена, загружено строк: %d", total_rows)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
