"""DAG дообучения рекомендательной системы.

Раз в неделю берёт события за последние train_window_days, переобучает ALS и
ранжировщик, считает метрики на отложенном окне, сравнивает recall@10 с
последней версией модели в MLflow Registry и, если качество не просело,
регистрирует новую версию и обновляет рекомендации в S3. Логики здесь нет:
все шаги лежат в plugins/steps, параметры — в params.yml.

Запуск: docker compose up -d --build, затем в интерфейсе Airflow или
командой airflow dags trigger recsys_retrain
"""

from datetime import timedelta

import pendulum
from airflow.decorators import dag, task

from steps.messages import send_failure_message, send_success_message


@dag(
    dag_id="recsys_retrain",
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 9, 1, tz="Europe/Moscow"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["recsys", "final"],
    on_success_callback=send_success_message,
    on_failure_callback=send_failure_message,
)
def recsys_retrain():
    """
    Еженедельное дообучение: выборка событий за интервал, подготовка,
    обучение двух стадий, сверка с Registry и выгрузка выдачи в S3
    """

    @task()
    def extract_events(**kwargs):
        """
        Выбирает события из Postgres за интервал прогона
        """
        from steps.extract import extract_events as step

        return step(**kwargs)

    @task()
    def prepare_data(extracted: dict):
        """
        Режет окно на историю, разметку и тест, строит матрицу
        """
        from steps.prepare import prepare_data as step

        return step(extracted)

    @task()
    def train_models(prepared: dict):
        """
        Обучает ALS и ранжировщик, считает метрики, пишет прогон в MLflow
        """
        from steps.train import train_models as step

        return step(prepared)

    @task()
    def evaluate_models(trained: dict):
        """
        Сверяет качество с Registry и регистрирует новую версию модели
        """
        from steps.evaluate import evaluate_models as step

        return step(trained)

    @task()
    def upload_recs(prepared: dict, trained: dict):
        """
        Пересчитывает рекомендации на всём окне и заливает их в S3
        """
        from steps.upload import upload_recs as step

        return step(prepared, trained)

    extracted = extract_events()
    prepared = prepare_data(extracted)
    trained = train_models(prepared)
    checked = evaluate_models(trained)
    uploaded = upload_recs(prepared, trained)

    # выгрузка идёт только после успешной сверки с Registry: если качество
    # просело, evaluate_models уходит в skipped и выгрузка не выполняется
    checked >> uploaded


recsys_retrain()
