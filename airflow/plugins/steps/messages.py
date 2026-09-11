"""Уведомления о результате прогона DAG.

Колбэки пишут итог прогона в лог планировщика и, если в окружении заданы
TELEGRAM_TOKEN и TELEGRAM_CHAT_ID, дублируют его сообщением в Telegram.
Без токена шаг не падает: уведомление остаётся только в логе, поэтому
запускать DAG можно и без настроенного бота.

Запуск: используется как библиотека (from steps.messages import ...)
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def send_message(text: str) -> None:
    """
    Отправляет сообщение в Telegram, если бот настроен переменными
    окружения; иначе просто оставляет запись в логе
    """
    logger.info(text)
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.info("TELEGRAM_TOKEN не задан, уведомление осталось в логе")
        return

    try:
        response = requests.post(
            TELEGRAM_API.format(token=token),
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
        response.raise_for_status()
    except Exception:
        # уведомление не должно ронять прогон, поэтому ошибку только логируем
        logger.exception("не удалось отправить уведомление в Telegram")


def send_success_message(context) -> None:
    """
    Колбэк успешного прогона: принимает словарь контекстных переменных
    """
    dag_id = context["dag"].dag_id
    run_id = context["run_id"]
    send_message(f"DAG {dag_id} с id={run_id} отработал успешно")


def send_failure_message(context) -> None:
    """
    Колбэк неудачного прогона: сообщает, какая задача упала
    """
    run_id = context["run_id"]
    task_key = context["task_instance_key_str"]
    send_message(
        f"DAG с id={run_id} завершился неудачно\nУпавшая задача: {task_key}"
    )
