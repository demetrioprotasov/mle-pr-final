"""Скачивает готовые рекомендации (результаты этапа 3) из S3-бакета
в локальный каталог data/, откуда их читает сервис. Каталог тот же,
что монтируется в контейнер, поэтому запускать скрипт нужно и перед
локальным запуском uvicorn, и перед docker compose up.

Скрипт намеренно не импортирует пакет recsys: сервис живёт в отдельном
образе и знать про обучающий код не должен, поэтому boto3 собирается
здесь заново, но по тем же переменным окружения.

Запуск: python download_recommendations.py
Доступы к S3 берутся из .env в корне репозитория (см. .env.example).
"""

import os
import time

import boto3
from dotenv import load_dotenv

# файлы, которые нужны сервису рекомендаций
FILES = [
    "recsys_final/recommendations/recommendations.parquet",
    "recsys_final/recommendations/top_popular.parquet",
    "recsys_final/recommendations/similar_items.parquet",
]

LOCAL_DIR = "data"


def main():
    load_dotenv()
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get(
            "MLFLOW_S3_ENDPOINT_URL", "https://storage.yandexcloud.net"
        ),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    bucket = os.environ["S3_BUCKET_NAME"]

    os.makedirs(LOCAL_DIR, exist_ok=True)
    for key in FILES:
        local_path = os.path.join(LOCAL_DIR, os.path.basename(key))
        started = time.time()
        s3.download_file(bucket, key, local_path)
        size_mb = os.path.getsize(local_path) / 1024 / 1024
        print(f"{key} -> {local_path}, {size_mb:.1f} МБ, {time.time() - started:.1f} с")
    print("Готово.")


if __name__ == "__main__":
    main()
