"""Хранилища данных сервиса рекомендаций.

Три класса с одним скелетом (__init__ / load / get / stats):
Recommendations отдаёт офлайн-рекомендации — персональные, а если
пользователя в них нет, то топ популярных; SimilarItems отдаёт похожие
товары для онлайн-части выдачи; EventStore держит в памяти последние
события пользователей.

Parquet-файлы читаются один раз при старте сервиса (см. recsys_app.main),
скачивает их download_recommendations.py.
"""

import logging

import pandas as pd

logger = logging.getLogger("uvicorn.error")

# в артефактах этапа 3 колонки названы так же, как в исходных данных
# Retailrocket (visitorid, itemid), внутри сервиса имена единые
COLUMN_NAMES = {
    "visitorid": "user_id",
    "itemid": "item_id",
    "similar_itemid": "sim_item_id",
    "similar_item_id": "sim_item_id",
}


class Recommendations:

    def __init__(self):

        self._recs = {"personal": None, "default": None}
        self._stats = {
            "request_personal_count": 0,
            "request_default_count": 0,
        }

    def load(self, type, path, **kwargs):
        """
        Загружает рекомендации из файла
        """

        logger.info(f"Loading recommendations, type: {type}")
        recs = pd.read_parquet(path, **kwargs).rename(columns=COLUMN_NAMES)
        if type == "personal":
            # сортируем один раз при загрузке: .loc по отсортированному индексу
            # ищет двоичным поиском, а товары внутри пользователя сразу идут
            # по возрастанию rank, и в get остаётся только взять первые k
            recs = recs.sort_values(["user_id", "rank"]).set_index("user_id")
        else:
            recs = recs.sort_values("rank")
        self._recs[type] = recs
        logger.info(f"Loaded {len(recs)} rows")

    def get(self, user_id: int, k: int = 10):
        """
        Возвращает список офлайн-рекомендаций для пользователя и источник
        выдачи: персональные ("personal"), а если их нет — топ популярных
        ("default")
        """
        try:
            # берём индекс списком, иначе для пользователя с одной строкой
            # pandas вернёт Series вместо DataFrame
            recs = self._recs["personal"].loc[[user_id]]
            recs = recs["item_id"].to_list()[:k]
            source = "personal"
            self._stats["request_personal_count"] += 1
        except KeyError:
            recs = self._recs["default"]
            recs = recs["item_id"].to_list()[:k]
            source = "default"
            self._stats["request_default_count"] += 1
        except Exception:
            logger.exception(f"No recommendations found for user_id={user_id}")
            recs, source = [], "default"

        return recs, source

    def is_loaded(self):
        """
        Загружены ли оба набора рекомендаций
        """
        return all(recs is not None for recs in self._recs.values())

    def stats(self):

        logger.info("Stats for recommendations")
        for name, value in self._stats.items():
            logger.info(f"{name:<30} {value}")


class SimilarItems:

    def __init__(self):

        self._similar_items = None
        self._stats = {
            "request_count": 0,
            "request_miss_count": 0,
        }

    def load(self, path, **kwargs):
        """
        Загружает похожие товары из файла
        """

        logger.info("Loading similar items")
        sim = pd.read_parquet(path, **kwargs).rename(columns=COLUMN_NAMES)
        sim = sim.sort_values(["item_id", "rank"]).set_index("item_id")
        self._similar_items = sim
        logger.info(f"Loaded {len(sim)} rows")

    def get(self, item_id: int, k: int = 10):
        """
        Возвращает список похожих товаров и их скоры
        """
        try:
            i2i = self._similar_items.loc[[item_id]].head(k)
            i2i = i2i[["sim_item_id", "score"]].to_dict(orient="list")
            self._stats["request_count"] += 1
        except KeyError:
            # товар не попал в матрицу ALS (слишком редкий), похожих для него нет
            logger.info(f"No similar items found for item_id={item_id}")
            i2i = {"sim_item_id": [], "score": []}
            self._stats["request_miss_count"] += 1
        except Exception:
            logger.exception(f"Failed to get similar items for item_id={item_id}")
            i2i = {"sim_item_id": [], "score": []}

        return i2i

    def is_loaded(self):
        """
        Загружены ли похожие товары
        """
        return self._similar_items is not None

    def stats(self):

        logger.info("Stats for similar items")
        for name, value in self._stats.items():
            logger.info(f"{name:<30} {value}")


class EventStore:

    def __init__(self, max_events_per_user=10):

        self._events = {}
        self._max_events_per_user = max_events_per_user
        self._stats = {
            "put_count": 0,
            "get_count": 0,
        }

    def put(self, user_id: int, item_id: int):
        """
        Сохраняет событие пользователя, оставляя не больше
        max_events_per_user последних
        """
        user_events = self._events.get(user_id, [])
        # срез на одну позицию короче лимита: сверху встаёт новое событие,
        # иначе история росла бы до max_events_per_user + 1
        keep = user_events[: self._max_events_per_user - 1]
        self._events[user_id] = [item_id] + keep
        self._stats["put_count"] += 1

    def get(self, user_id: int, k: int = 10):
        """
        Возвращает события пользователя (первыми — самые последние)
        """
        user_events = self._events.get(user_id, [])[:k]
        self._stats["get_count"] += 1

        return user_events

    def stats(self):

        logger.info("Stats for event store")
        for name, value in self._stats.items():
            logger.info(f"{name:<30} {value}")
        logger.info(f"{'users_with_events':<30} {len(self._events)}")
