"""Константы сервиса рекомендаций.

Пути к parquet-артефактам и ограничения на размер выдачи. Артефакты
кладёт рядом с сервисом download_recommendations.py, в контейнере тот же
каталог примонтирован как /app/data.
"""

import os

# каталог с артефактами: локально запускаем из services, в контейнере
# рабочий каталог /app, поэтому относительного пути хватает в обоих случаях
DATA_DIR = os.environ.get("RECS_DATA_DIR", "data")

PERSONAL_RECS_PATH = os.path.join(DATA_DIR, "recommendations.parquet")
DEFAULT_RECS_PATH = os.path.join(DATA_DIR, "top_popular.parquet")
SIMILAR_ITEMS_PATH = os.path.join(DATA_DIR, "similar_items.parquet")

# офлайн-рекомендации посчитаны на этапе 3 как top-20 на пользователя,
# поэтому больше двадцати товаров сервису отдавать просто неоткуда
DEFAULT_K = 10
MAX_K = 20

# сколько последних событий пользователя храним в памяти и по скольким
# из них строим онлайн-часть выдачи
MAX_EVENTS_PER_USER = 10
N_ONLINE_EVENTS = 3
