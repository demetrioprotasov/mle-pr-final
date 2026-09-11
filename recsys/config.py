"""Конфигурация проекта.

Читает переменные окружения из .env (доступы к Postgres и S3), собирает их
в словари для psycopg2/boto3 и хранит константы путей в S3 и SEED,
общие для скриптов, ноутбуков и шагов DAG.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# --- доступы к Postgres (retailrocket_events и backend store MLflow) ---
postgres_credentials = {
    "host": os.environ["DB_DESTINATION_HOST"],
    "port": os.environ["DB_DESTINATION_PORT"],
    "dbname": os.environ["DB_DESTINATION_NAME"],
    "user": os.environ["DB_DESTINATION_USER"],
    "password": os.environ["DB_DESTINATION_PASSWORD"],
}
assert all([value != "" for value in postgres_credentials.values()])

# --- доступы к S3 (Yandex Object Storage) ---
s3_credentials = {
    "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
    "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
    "endpoint_url": os.environ.get(
        "MLFLOW_S3_ENDPOINT_URL", "https://storage.yandexcloud.net"
    ),
}
assert all([value != "" for value in s3_credentials.values()])

S3_BUCKET = os.environ["S3_BUCKET_NAME"]

# --- пути артефактов в S3 ---
S3_PREFIX = "recsys_final"
ITEMS_PATH = f"{S3_PREFIX}/data/items.parquet"
RECS_PERSONAL_PATH = f"{S3_PREFIX}/recommendations/recommendations.parquet"
RECS_DEFAULT_PATH = f"{S3_PREFIX}/recommendations/top_popular.parquet"
SIMILAR_PATH = f"{S3_PREFIX}/recommendations/similar_items.parquet"
ID_MAPS_PATH = f"{S3_PREFIX}/models/id_maps.parquet"
ALS_MODEL_PATH = f"{S3_PREFIX}/models/als_model.npz"

SEED = 42
