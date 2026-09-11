"""Конфигурация Flask-AppBuilder для веб-сервера Airflow.

Переопределяет только AUTH_ROLE_PUBLIC — остальное берётся из значений
по умолчанию образа apache/airflow. AIRFLOW__WEBSERVER__AUTH_ROLE_PUBLIC
в docker-compose.yaml на этот файл не влияет: в 2.7.3 Flask-AppBuilder
читает роль для анонимного доступа только отсюда.
"""

from airflow.www.fab_security.manager import AUTH_DB

AUTH_TYPE = AUTH_DB

# анонимный доступ на чтение — чтобы снять скриншот Grid view без логина
AUTH_ROLE_PUBLIC = "Viewer"
