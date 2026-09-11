"""Генерация нагрузки на сервис рекомендаций.

Шлёт тысячу запросов со случайными пользователями и товарами, чтобы на
дашборде Grafana были данные на всех панелях. Пользователи берутся двух
сортов: из recommendations.parquet (персональная выдача) и заведомо
неизвестные (холодный старт, фолбэк на топ популярных). Часть запросов
идёт в /similar_items и /events, чтобы ожили метрики онлайн-части.

Перед запуском сервис должен быть поднят, а артефакты скачаны в data/.

Запуск: python simulate_load.py
"""

import os
import random
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

N_REQUESTS = 1000
MIN_DELAY = 0.02
MAX_DELAY = 0.08
SEED = 42

service_url = f"http://127.0.0.1:{os.environ.get('MAIN_APP_PORT', '8010')}"

PERSONAL_RECS_PATH = "data/recommendations.parquet"
DEFAULT_RECS_PATH = "data/top_popular.parquet"

COLUMN_NAMES = {"visitorid": "user_id", "itemid": "item_id"}

random.seed(SEED)

personal = pd.read_parquet(PERSONAL_RECS_PATH).rename(columns=COLUMN_NAMES)
known_users = personal["user_id"].unique().tolist()
unknown_user_from = int(max(known_users)) + 1_000_000

top_popular = pd.read_parquet(DEFAULT_RECS_PATH).rename(columns=COLUMN_NAMES)
popular_items = top_popular["item_id"].tolist()

statuses = {}

for i in range(N_REQUESTS):
    k = random.randint(1, 20)
    # каждый десятый пользователь неизвестен сервису — так на дашборде
    # видно долю холодных стартов, а не ровную нулевую линию
    if random.random() < 0.1:
        user_id = unknown_user_from + random.randint(1, 10_000)
    else:
        user_id = int(random.choice(known_users))

    roll = random.random()
    if roll < 0.1:
        # событие пользователя: дальше его выдача станет смешанной
        resp = requests.post(
            service_url + "/events",
            json={"user_id": user_id, "item_id": int(random.choice(popular_items))},
        )
    elif roll < 0.2:
        resp = requests.get(
            service_url + "/similar_items",
            params={"item_id": int(random.choice(popular_items)), "k": k},
        )
    elif roll < 0.35:
        resp = requests.get(
            service_url + "/recommendations_offline",
            params={"user_id": user_id, "k": k},
        )
    else:
        resp = requests.get(
            service_url + "/recommendations", params={"user_id": user_id, "k": k}
        )

    statuses[resp.status_code] = statuses.get(resp.status_code, 0) + 1
    if (i + 1) % 100 == 0:
        print(f"{i + 1}/{N_REQUESTS} -> {statuses}")

    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

print(f"Готово. Ответы по статусам: {statuses}")
