"""Работа с S3 (Yandex Object Storage): чтение/запись parquet, выгрузка файлов.

Клиент boto3 собирается из переменных окружения через recsys.config.
Бакет один на весь проект (S3_BUCKET_NAME), все пути внутри него —
с префиксом recsys_final/ (см. recsys.config.S3_PREFIX).
"""

import io
import logging

import boto3
import pandas as pd

from recsys.config import S3_BUCKET, s3_credentials

logger = logging.getLogger(__name__)

s3 = boto3.client(
    "s3",
    endpoint_url=s3_credentials["endpoint_url"],
    aws_access_key_id=s3_credentials["aws_access_key_id"],
    aws_secret_access_key=s3_credentials["aws_secret_access_key"],
)


def read_parquet(key: str) -> pd.DataFrame:
    """
    Читает parquet-файл из S3-бакета в DataFrame
    """
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, key, buf)
    buf.seek(0)
    df = pd.read_parquet(buf)
    logger.info("прочитано s3://%s/%s (%d строк)", S3_BUCKET, key, len(df))
    return df


def write_parquet(df: pd.DataFrame, key: str) -> None:
    """
    Сохраняет DataFrame в parquet-файл в S3-бакете
    """
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.upload_fileobj(buf, S3_BUCKET, key)
    logger.info("сохранено s3://%s/%s (%d строк)", S3_BUCKET, key, len(df))


def upload_file(local_path: str, key: str) -> None:
    """
    Загружает локальный файл в S3-бакет по указанному ключу
    """
    s3.upload_file(local_path, S3_BUCKET, key)
    logger.info("загружено %s -> s3://%s/%s", local_path, S3_BUCKET, key)


def list_objects(prefix: str) -> list[str]:
    """
    Возвращает список ключей в S3-бакете с заданным префиксом
    """
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys
