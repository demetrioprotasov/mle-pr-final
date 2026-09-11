"""Шаг evaluate_models: сравнение с Registry и регистрация новой версии.

За эталон берётся recall@10 последней версии модели в Model Registry:
у версии есть run_id, из метрик этого прогона и читается метрика. Если новая
модель набрала меньше degradation_tolerance от эталона, шаг завершается
AirflowSkipException — выгрузка рекомендаций в S3 не выполняется, в бакете
остаётся прежняя выдача. Первый прогон, когда в Registry ещё пусто,
регистрирует модель без сравнения.

Запуск: вызывается задачей evaluate_models DAG recsys_retrain
"""

import logging

import mlflow
from airflow.exceptions import AirflowSkipException
from mlflow.tracking import MlflowClient

from steps.common import load_params, setup_mlflow

logger = logging.getLogger(__name__)


def reference_recall(client: MlflowClient, name: str, metric: str):
    """
    Возвращает метрику последней версии модели в Registry и номер этой
    версии; если модель ещё не зарегистрирована — (None, None)
    """
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        return None, None
    latest = max(versions, key=lambda version: int(version.version))
    run = client.get_run(latest.run_id)
    return run.data.metrics.get(metric), latest.version


def evaluate_models(trained: dict) -> dict:
    """
    Сравнивает recall@k новой модели с эталоном из Registry и регистрирует
    новую версию, если качество не просело
    """
    params = load_params()
    setup_mlflow(params)
    client = MlflowClient()

    metric = f"test_recall_{params['top_k']}"
    tolerance = params["degradation_tolerance"]
    reference, version = reference_recall(client, params["registry_model"], metric)
    new_recall = trained["recall"]

    if reference is None:
        logger.info("в Registry нет версий, сравнивать не с чем")
        threshold = 0.0
    else:
        threshold = tolerance * reference
        logger.info(
            "эталон — версия %s: %s = %.4f, порог %.4f, новая модель %.4f",
            version,
            metric,
            reference,
            threshold,
            new_recall,
        )

    registered = new_recall >= threshold
    with mlflow.start_run(run_id=trained["mlflow_run_id"]):
        mlflow.log_metrics(
            {
                "reference_recall": reference if reference is not None else -1.0,
                "degradation_tolerance": tolerance,
                "registered": int(registered),
            }
        )

    if not registered:
        # downstream-задача выгрузки пропускается сама: у неё all_success
        logger.warning(
            "качество просело: %.4f < %.4f, новая версия не регистрируется "
            "и рекомендации в S3 не обновляются",
            new_recall,
            threshold,
        )
        raise AirflowSkipException("recall@%d ниже порога" % params["top_k"])

    model_version = mlflow.register_model(
        trained["model_uri"], params["registry_model"]
    )
    logger.info(
        "зарегистрирована версия %s модели %s",
        model_version.version,
        params["registry_model"],
    )
    return {
        "registry_version": str(model_version.version),
        "reference_recall": reference,
        "new_recall": new_recall,
    }
