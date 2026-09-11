FROM apache/airflow:2.7.3-python3.11

# libgomp1 нужен implicit/catboost/scikit-learn — они собраны с OpenMP,
# а в базовом образе этой системной библиотеки нет
USER root
RUN apt-get update -o Acquire::Check-Valid-Until=false \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-airflow.txt /requirements-airflow.txt

USER airflow
RUN pip install --no-cache-dir -r /requirements-airflow.txt
