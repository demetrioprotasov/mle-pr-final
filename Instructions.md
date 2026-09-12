# Инструкции по запуску проекта

Все команды выполняются из корня репозитория `mle-pr-final`, если не указано
иное.

## 1. Виртуальное окружение и доступы

```bash
# создание виртуального окружения и установка библиотек
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# ядро Jupyter для ноутбуков
python -m ipykernel install --user --name mle-pr-final
```

Доступы к Postgres и S3 — в `.env`, который не коммитится:

```bash
cp .env.example .env
```

В `.env` нужно заполнить `DB_DESTINATION_*` (учебная Postgres),
`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`S3_BUCKET_NAME` (Yandex Object
Storage) и `GRAFANA_USER`/`GRAFANA_PASS`. `MLFLOW_TRACKING_URI` и
`MAIN_APP_PORT` уже заполнены значениями по умолчанию.

## 2. Данные

Распаковать `events.csv`, `item_properties_part1.csv`,
`item_properties_part2.csv` и `category_tree.csv` в `data/` в корне
репозитория, затем залить события и собрать каталог товаров:

```bash
# заливка events.csv в Postgres, таблица retailrocket_events
python scripts/load_events_to_pg.py --data-dir data

# при повторном запуске (например, после смены данных) — с очисткой таблицы
python scripts/load_events_to_pg.py --data-dir data --truncate

# сборка items.parquet (только categoryid/available) и заливка в S3
python scripts/build_items_catalog.py --data-dir data
```

## 3. MLflow

```bash
# запуск tracking server: backend и registry — Postgres, артефакты — S3
bash run_mlflow_server.sh
```

Держать команду в отдельном терминале. Проверка:

```bash
curl -s http://127.0.0.1:5000/health
```

Веб-интерфейс — `http://127.0.0.1:5000`, эксперимент `final_project_recsys`
(id 9).

## 4. Ноутбуки

```bash
jupyter lab --ip=0.0.0.0 --no-browser
```

Сначала `notebooks/eda_retailrocket.ipynb` (этап 1, EDA), затем
`notebooks/recommendations.ipynb` (этапы 3-4: сплиты, ALS, Optuna, i2i,
ранжировщик, метрики, выгрузка артефактов в S3 и MLflow). Оба ноутбука
выполняются сверху вниз при поднятом MLflow-сервере и заполненном `.env`.

## 5. Сервис рекомендаций в Docker

```bash
cd services

# скачать посчитанные в ноутбуке артефакты из S3 в services/data/
python download_recommendations.py

# поднять сервис, Prometheus и Grafana (.env — в корне репозитория)
docker compose --env-file ../.env up --build -d
```

Проверка вручную:

```bash
curl -s "http://127.0.0.1:8010/health"
curl -s "http://127.0.0.1:8010/recommendations_offline?user_id=2&k=10"
curl -s "http://127.0.0.1:8010/similar_items?item_id=461686&k=10"
```

Остановка:

```bash
docker compose down
```

## 6. Тестирование и нагрузка

Сервис (локально или в Docker) должен быть поднят, артефакты — скачаны в
`services/data/`.

```bash
cd services

# сценарии проверки API: /health, персональные и офлайн-рекомендации,
# похожие товары, события, блендинг, негативные запросы
python test_service.py

# 1000 запросов со случайными пользователями и паузами 20-80 мс —
# чтобы на дашборде Grafana были данные
python simulate_load.py
```

Вывод `test_service.py` дублируется в `test_service.log` и завершается
ошибкой, если хоть одна проверка не прошла.

## 7. Мониторинг

Grafana — `http://localhost:3000`, вход не нужен (анонимный доступ с ролью
Viewer), логин и пароль администратора — `GRAFANA_USER`/`GRAFANA_PASS` из
`.env`, нужны только для редактирования. Дашборд «Мониторинг сервиса
рекомендаций» провижинится автоматически.

Открыть дашборд сразу после `simulate_load.py` и выставить в правом верхнем
углу интервал **Last 15 minutes** и автообновление **5s** — иначе минута
нагрузки сожмётся в несколько пикселей и панели покажутся пустыми.

Prometheus — `http://localhost:9090/targets`, состояние сборщика
`scrapping-recsys-app` должно быть `State=UP`. Обоснование метрик и панелей
— в [Monitoring.md](Monitoring.md).

## 8. Airflow (дообучение по расписанию)

```bash
cd airflow

# id пользователя хоста — иначе файлы в volume будут принадлежать root
cp .env.example .env
echo "AIRFLOW_UID=$(id -u)" >> .env

# инициализация служебной базы Airflow
docker compose up airflow-init

# сборка образа и запуск postgres/webserver/scheduler
docker compose up -d --build
```

Проверка и запуск DAG:

```bash
# в списке DAG не должно быть ошибок импорта
docker compose exec airflow-scheduler airflow dags list-import-errors

# DAG recsys_retrain должен быть в списке
docker compose exec airflow-scheduler airflow dags list | grep recsys_retrain

# ручной прогон: окно выборки всё равно упирается в data_max_date из params.yml
docker compose exec airflow-scheduler airflow dags trigger recsys_retrain
```

Веб-интерфейс — `http://localhost:8080` (`airflow`/`airflow`, доступен и
анонимный просмотр); прогон смотреть в Grid view DAG `recsys_retrain`.
Пять задач: `extract_events → prepare_data → train_models →
evaluate_models → upload_recs`. Если качество новой модели просело больше
чем на 5 % от версии в Model Registry, `evaluate_models` и `upload_recs`
помечаются пропущенными (skipped) — это штатное поведение, а не ошибка.

Остановка:

```bash
docker compose down
```

## 9. DVC

Пайплайн работает поверх тех же функций пакета `recsys`, что и ноутбук, и
шаги DAG, при заполненном `.env` (доступ к Postgres и S3).

```bash
# прогнать все три стадии: prepare_data → train → evaluate
dvc repro

# метрики последнего прогона
dvc metrics show

# ничего не изменилось — повторный прогон ничего не пересчитывает
dvc status
```

Ремоут — `s3://s3-student-mle-20260317-efc01cb482-freetrack/dvc_final`.
Ключи доступа задаются локально и не попадают в git:

```bash
dvc remote modify --local s3_storage access_key_id <ключ>
dvc remote modify --local s3_storage secret_access_key <секрет>

dvc push
```

Выходы `models/als_model.npz`, `models/catboost_ranker.cbm` и
`cv_results/metrics.json` помечены в `dvc.yaml` как `cache: false`, то есть
`dvc push` их не отправляет: модель ALS слишком тяжёлая для узкого канала
и нужна только локально, а два остальных файла лежат в git.
