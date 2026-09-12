# Рекомендательная система товаров в электронной коммерции — итоговый проект

Итоговый проект курса «ML-инженер с опытом» (кейс 2, Retailrocket):
рекомендации товаров для интернет-магазина по логу событий пользователей.

## Цель

Перевести бизнес-метрику рекомендательного блока (конверсию в добавление
товара в корзину) в офлайн-метрики качества, сравнить топ-популярную,
персональную (ALS) и двухстадийную (ALS + ранжировщик CatBoost) модели и
довести лучшую до продакшена: веб-сервис в Docker с мониторингом, дообучение
по расписанию в Airflow, воспроизводимый пайплайн экспериментов на DVC.

## Таблица «Задача — Артефакт — Файл»

| № | Задача | Артефакт по заданию | Файлы в репозитории |
|---|---|---|---|
| 1 | Исследование данных | Jupyter Notebook с EDA | `notebooks/eda_retailrocket.ipynb` |
| 2 | Подготовка инфраструктуры | `.sh`-скрипт запуска и настройки MLflow | `run_mlflow_server.sh`, «Этап 2» ниже, `screenshots/mlflow_experiment.png` |
| 3 | Трансляция бизнес-задачи | Описано в README | «Бизнес-задача и её техническая трансляция», «Результаты экспериментов» |
| 4 | Моделирование | Notebook с экспериментами, bin-файл модели | `notebooks/recommendations.ipynb`, `recsys/*.py`, `models/catboost_ranker.cbm`, `models/model_params.json`, `cv_results/metrics.json` |
| 5 | Продуктивизация | Python-проект с Dockerfile и описанной структурой API | `services/`, «API» ниже, `Instructions.md` |
| 6 | Пайплайн дообучения | Граф Airflow (`.py`) | `airflow/dags/recsys_retrain.py`, `airflow/plugins/steps/*.py`, `airflow/params.yml`, `screenshots/airflow_dag.png` |
| 7 | Мониторинг | `.md` с метриками, метрики из кода проекта | `Monitoring.md`, `services/recsys_app/metrics.py`, `services/prometheus/prometheus.yml`, `services/grafana/provisioning/` |
| 8 | Документация | Заполненный README | `README.md`, `Instructions.md` |
| 9 | Требования и среда | `requirements.txt` | `requirements.txt`, `services/requirements.txt`, `airflow/requirements-airflow.txt`, `params.yaml`, «Воспроизводимость» ниже |

## Технологии

- Python 3.12; pandas 2.2.1, numpy 1.26.4, scipy 1.11.4, pyarrow 15.0.2
- implicit 0.7.3 (ALS), CatBoost 1.2.8, Optuna 4.4.0 — модели и подбор параметров
- MLflow 3.15.1 — трекинг экспериментов и Model Registry
- FastAPI 0.115.6, Uvicorn 0.24, Docker, Docker Compose — веб-сервис
- Prometheus v2.53.0, Grafana 11.2.0 — мониторинг сервиса
- Apache Airflow 2.7.3 — дообучение модели по расписанию
- DVC 3.48.1 (remote s3) — воспроизводимый пайплайн
- boto3 1.43.73, psycopg2-binary 2.9.9, SQLAlchemy 2.0.25 — Object Storage и Postgres

## Бизнес-задача и её техническая трансляция

Бизнес-метрика рекомендательного блока интернет-магазина — конверсия показов
рекомендаций в добавление товара в корзину (Add-to-Cart CR): по условию
задания ориентир — именно это действие, а не покупка напрямую. Замерить её
онлайн, A/B-тестом, в рамках проекта нельзя, поэтому качество моделей
оценивается офлайн-прокси-метриками на отложенном по времени окне.

Целевое действие — `addtocart`; `transaction` не выбрасывается, а считается
более сильным позитивом: покупка тоже подсвечивает интерес, хотя и означает,
что тот же товар рекомендовать повторно смысла мало. Оба типа событий
считаются релевантными парами (visitorid, itemid) при расчёте метрик; `view`
в релевантные пары не входит, но используется как более слабый сигнал
интереса при построении матрицы для ALS (см. «Этап 3»).

Основные офлайн-метрики — `precision@10` и `recall@10`: доля релевантных
товаров в выдаче из 10 позиций и доля релевантных товаров пользователя,
попавших в эту выдачу. Глубина 10 выбрана потому, что именно столько позиций
помещается в блок рекомендаций на витрине. Дополнительно считаются `MAP@10`
(усреднение precision по позициям релевантных товаров, учитывает их порядок),
`NDCG@10` (то же с логарифмическим дисконтом по позиции), `coverage@10` (доля
каталога, попавшая хоть кому-нибудь в топ-10) и `novelty@10` (средняя
`-log2(p)` рекомендованных товаров, `p` — доля визитёров, видевших товар в
обучающей истории).

Данные разбиты по времени, а не случайно: рекомендательная система
предсказывает будущее поведение, и случайный сплит дал бы моделям подсмотреть
будущую популярность товаров и будущую историю самого пользователя. Схема —
Global Time Split на четыре непересекающихся окна (см. «Этап 3»).

База усреднения метрик одинакова для всех сравниваемых моделей: визитёры,
у которых в тестовом окне есть хотя бы один `addtocart`/`transaction` **и**
хотя бы одно событие в истории обучения (`train` = `train_fit` + `valid`).
Визитёра без истории ни одна модель не может персонально ранжировать, поэтому
включать его в среднее бессмысленно; база — 383 пользователя (поле
`users_test` в `cv_results/metrics.json`).

## Данные

Датасет Retailrocket: `events.csv` (2 756 101 событие за 2015-05-03 —
2015-09-18, 1 407 580 визитёров, 235 061 товар),
`item_properties_part1/2.csv` (свойства товаров, 20.3 млн строк) и
`category_tree.csv` (1 669 категорий, 25 корневых, глубина дерева 6).

Каталог товаров (`scripts/build_items_catalog.py`) строится только из
товаров, встречающихся в `events.csv`, и только по двум свойствам,
`categoryid` и `available`: остальные анонимизированы числовыми id без
словаря. Значение берётся на момент до 2015-08-04 (граница `train_fit`),
чтобы не подглядывать в будущее: у 6.22 % товаров со временем меняется
`categoryid`, у 17.26 % — `available`. Разбор — в
`notebooks/eda_retailrocket.ipynb`, ключевые выводы — ниже.

## Структура репозитория

```
mle-pr-final/
├── README.md                       этот файл
├── Instructions.md                 пошаговый запуск проекта
├── Monitoring.md                   обоснование метрик и панелей Grafana
├── requirements.txt                зависимости проекта (venv)
├── .env.example, .gitignore        имена переменных окружения, игнор-лист
├── params.yaml                     границы сплитов, веса событий, гиперпараметры (DVC)
├── dvc.yaml, dvc.lock              3 стадии: prepare_data → train → evaluate
├── .dvc/config                     ремоут DVC (ключи — в config.local, не в git)
├── run_mlflow_server.sh            запуск MLflow tracking server
├── notebooks/
│   ├── eda_retailrocket.ipynb      этап 1: EDA по событиям, товарам, категориям
│   ├── recommendations.ipynb       этапы 3-4: сплиты, ALS, Optuna, i2i, ранжирование, MLflow
│   └── scratch.ipynb               черновик проверки подключений к S3/Postgres
├── recsys/                         пакет, общий для ноутбука, стадий DVC и шагов DAG
│   ├── config.py, data.py          .env и SEED; события из Postgres, каталог, сплит
│   ├── features.py, models.py      веса и признаки; топ популярных, ALS, i2i, CatBoost
│   └── metrics.py, s3_io.py        precision/recall/MAP/NDCG/coverage/novelty; S3
├── scripts/
│   ├── load_events_to_pg.py, build_items_catalog.py   разовая подготовка данных
│   └── prepare_data.py, train.py, evaluate.py         стадии DVC
├── models/
│   ├── als_model.npz               ALS на всей истории, не в git (см. «Ограничения»)
│   └── catboost_ranker.cbm, model_params.json   ранжировщик и его параметры
├── cv_results/metrics.json         метрики моделей (выход стадии evaluate)
├── services/
│   ├── recsys_app/                 main.py, stores.py, metrics.py, constants.py
│   ├── download_recommendations.py скачивание артефактов из S3 в data/
│   ├── test_service.py             сценарии проверки API, вывод — в test_service.log
│   ├── simulate_load.py            нагрузка, чтобы дашборд не был пустым
│   ├── requirements.txt, Dockerfile
│   ├── docker-compose.yaml         recsys-app + prometheus + grafana
│   ├── prometheus/prometheus.yml
│   └── grafana/provisioning/        datasource.yml и recsys_dashboard.json
├── airflow/
│   ├── Dockerfile                  apache/airflow:2.7.3-python3.11 + libgomp1
│   ├── docker-compose.yaml         LocalExecutor: postgres + init + webserver + scheduler
│   ├── .env.example                AIRFLOW_UID для прав на volumes
│   ├── config/webserver_config.py  анонимный просмотр интерфейса (роль Viewer)
│   ├── requirements-airflow.txt, params.yml   зависимости и параметры дообучения
│   ├── dags/recsys_retrain.py      DAG @weekly, 5 задач
│   └── plugins/steps/              common / extract / prepare / train / evaluate / upload / messages
└── screenshots/                     mlflow_experiment, mlflow_registry, airflow_dag,
                                     grafana_dashboard, prometheus_targets — png
```

### Этап 1. Исследование данных

EDA — `notebooks/eda_retailrocket.ipynb`. Ключевые выводы:

- **Объём и период.** 2 756 101 событие за 138 дней, 1 407 580 визитёров,
  235 061 товар. Поток по дням ровный (коэффициент вариации 0.21), поэтому
  сплит можно делать равными окнами.
- **Воронка.** `view` 96.67 %, `addtocart` 2.52 %, `transaction` 0.81 %;
  конверсия из корзины в покупку — 30.64 % на уровне пар (визитёр, товар).
  Хотя бы одно добавление в корзину сделали 37 722 визитёра (2.68 %), хотя бы
  одну покупку — 11 719 (0.83 %). Целевой сигнал редкий: обучать модель
  только на целевых событиях нельзя.
- **Разреженность.** 2 145 179 уникальных пар (визитёр, товар) на матрицу
  1 407 580 × 235 061 — плотность 6.48e-06, типично для implicit ALS, но
  задаёт низкий потолок абсолютных значений precision/recall.
- **Холодный старт.** 31.31 % товаров получили за весь период ровно одно
  событие; большинство визитёров приходят один раз и не имеют истории —
  сервис обязан отдавать осмысленный ответ и без персонализации.
- **Контентные признаки.** У самого частого свойства `888` — 281 842
  уникальных значения без словаря, расшифровать нельзя; значения свойств
  меняются во времени у 4.67 % пар (товар, свойство), поэтому каталог
  собирается на дату отсечки.

Полный список выводов — в самом ноутбуке, раздел «Итоговые выводы».

### Этап 2. Инфраструктура обучения (MLflow)

MLflow tracking server поднимается локально скриптом `run_mlflow_server.sh`:
читает переменные `.env`, backend store и registry — учебная Postgres,
artifact root — `s3://$S3_BUCKET_NAME` через `--no-serve-artifacts`, порт
5000. Версия `mlflow==3.15.1` зафиксирована по ревизии `alembic_version` этой
базы — в ней же лежат эксперименты прошлых спринтов, поэтому
`mlflow db upgrade` не запускался. Эксперимент проекта —
`final_project_recsys`, id 9 (0-8 заняты прошлыми спринтами). Скриншоты —
`screenshots/mlflow_experiment.png` (список запусков) и
`screenshots/mlflow_registry.png` (зарегистрированные версии).

### Этап 3. Моделирование

Моделирование — в `notebooks/recommendations.ipynb`, логика вынесена в пакет
`recsys` (его же используют стадии DVC и шаги DAG). Даты сплитов, веса
событий и гиперпараметры не задаются в ноутбуке руками — все читаются из
`params.yaml`.

Сплит по времени (границы строгие, `<`, окна не пересекаются):

```
2015-05-03 ──── 2015-08-04 ──── 2015-08-19 ──── 2015-09-03 ──── 2015-09-19
      train_fit          valid            labels             test
   1 950 977 соб.     269 662 соб.      267 469 соб.       267 993 соб.
```

`train_fit` — обучение ALS при подборе гиперпараметров, `valid` — оценка
`recall@10` в Optuna (тест при подборе не используется), `train` =
`train_fit + valid` (2 220 639 событий) — обучение финальной модели, `labels`
— источник таргета ранжировщика, `test` — единственное окно итоговых метрик.

**Веса событий.** Сырой вес пары (визитёр, товар) — сумма весов её событий
(`view` 1, `addtocart` 5, `transaction` 10), значение в матрице — `log1p` от
этой суммы: медиана событий на визитёра равна 1, но максимум доходит до
нескольких тысяч, логарифм гасит вклад накрутчиков. Матрица `train`:
1 131 714 × 213 612, ненулевых значений 1 725 395. Соответствие сырых и
закодированных идентификаторов хранится в `id_maps.parquet`, а не pickle-ом
(venv на Python 3.12, образ Airflow — на 3.11).

**ALS.** `implicit.als.AlternatingLeastSquares` со стартовыми параметрами
(`factors=64, regularization=0.05, iterations=20`) даёт на тесте `recall@10`
0.0235 против 0.0070 у топа популярных. Гиперпараметры подбираются Optuna
(`TPESampler(seed=42)`, 12 попыток, `factors ∈ {32, 64, 128}`,
`regularization ∈ loguniform(1e-3, 1e-1)`, `iterations ∈ {10, 15, 20}`,
целевая функция — `recall@10` на `valid`, обучение только на `train_fit`):
лучшие параметры — `factors=128, regularization=0.008168, iterations=10`,
`recall@10` на валидации — 0.0816. Каждая попытка — вложенный run `trial_N`
в MLflow (run `Stage_3_ALS_Optuna`), найденные параметры зафиксированы в
`params.yaml` (`als_best`) и `airflow/params.yml` (`als`). Похожие товары
(i2i) — 10 ближайших соседей по факторам ALS на товар,
`similar_items.parquet`, 2 136 120 строк на 213 612 товаров.

**Ранжировщик.** Финальная модель — двухстадийная: кандидаты первой стадии —
топ-100 ALS + топ-50 популярных, таргет — `addtocart`/`transaction` из окна
`labels`, признаки (13 штук: скор и позиция ALS, скор и позиция топа
популярных, счётчики и конверсия товара, его возраст и категория, активность
и давность пользователя, доля событий пользователя в категории кандидата)
считаются только по `train`. Пул кандидатов на обучающей выборке — 74 642
строки на 525 пользователей, позитивов в пуле 176 из 1 452 пар окна `labels`;
до 30 негативов на пользователя, пятая часть пользователей
(`holdout_share=0.2`) отложена для ранней остановки.
`CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6,
random_seed=42, cat_features=["categoryid"])` останавливается на 95 деревьях
(`train_auc=0.9728, holdout_auc=0.9136`), файл — 0.21 МБ. На тесте пул
кандидатов — 54 492 строки на 383 пользователя, в него попадает лишь 71 из
1 053 позитивных пар (6.7 %) — это потолок схемы: даже идеальный ранжировщик
не поднял бы `recall@10` выше 0.09.

**Артефакты для сервиса.** Перед выкаткой ALS и признаки пересчитываются на
всей истории (`train + labels + test`, матрица 1 407 580 × 235 061) с теми же
гиперпараметрами — иначе сервис стартовал бы с моделью, не знающей последний
месяц; ранжировщик не переобучается. Персональные рекомендации считаются для
406 020 визитёров — тех, у кого в истории есть хотя бы 2 события
(`min_user_events`); у остальных запросы обслуживает топ популярных. Итоговая
`recommendations.parquet` — top-20 на пользователя, 8 120 400 строк,
10 733 уникальных товара.

### Этап 4. Сервис и мониторинг

Один процесс FastAPI вместо трёх учебных микросервисов — код в
`services/recsys_app/`. Хранилища `Recommendations`, `SimilarItems`,
`EventStore` (`stores.py`) загружаются в `lifespan` при старте, блендинг
вызывает их напрямую, без HTTP-запросов к самому себе. Эндпоинты — в разделе
«API» ниже, метрики и панели — в [Monitoring.md](Monitoring.md). Образ —
`services/Dockerfile` (`python:3.12-slim`), `services/docker-compose.yaml`
поднимает сервис вместе с Prometheus (`prom/prometheus:v2.53.0`) и Grafana
(`grafana/grafana:11.2.0`).

### Этап 5. Дообучение по расписанию (Airflow)

DAG `recsys_retrain` (`airflow/dags/recsys_retrain.py`, TaskFlow API,
`schedule="@weekly"`) — тонкий: только импорты, декораторы и вызовы функций
из `airflow/plugins/steps/*.py`, вся логика — в шагах, параметры — в
`airflow/params.yml`. Пять задач: `extract_events` (события из Postgres
хуком `PostgresHook` — образ Airflow держит SQLAlchemy 1.4, а `recsys.data`
рассчитан на 2.0, поэтому в шаге отдельный SQL-запрос через сам хук),
`prepare_data` (режет окно на историю, labels и test, строит матрицу),
`train_models` (ALS + ранжировщик, метрики, запись в MLflow),
`evaluate_models` (сверка с Model Registry), `upload_recs` (переобучение на
всём окне, заливка в S3).

Датасет Retailrocket заморожен на сентябре 2015: верхняя граница окна выборки
ограничена параметром `data_max_date: "2015-09-18"` из `airflow/params.yml`,
даже если `data_interval_end` прогона больше, — в реальной эксплуатации эта
строка убирается. Глубина истории — `train_window_days: 120` дней, что шире,
чем `train` ноутбука (108 дней), поэтому числа прогона DAG отличаются от
чисел ноутбука — это ожидаемо, а не рассинхрон.

`evaluate_models` сравнивает `test_recall_10` новой модели с той же метрикой
последней версии в Model Registry (`recsys_ranker_model`): если новая модель
набрала меньше 95 % (`degradation_tolerance`) от эталона, шаг завершается
`AirflowSkipException`, версия не регистрируется, и `upload_recs` из-за
зависимости по успеху пропускается — в бакете остаётся прежняя выдача. На
реальном прогоне новая модель показала `recall@10` 0.0453 против эталона
0.0397 (версия 1, обученная в ноутбуке) — деградации нет, зарегистрирована
версия 2, и рекомендации в S3 обновились. Скриншот прогона в Grid view —
`screenshots/airflow_dag.png`.

Модель ALS не заливается ни в S3, ни в MLflow (см. «Ограничения»). Адрес
сервера MLflow резолвится по IP, а не по имени `host.docker.internal`:
MLflow 3.x проверяет заголовок `Host` (`steps/common.py: tracking_uri`).

### Этап 6. Воспроизводимость (DVC)

Пайплайн `dvc.yaml` из трёх стадий поверх тех же функций пакета `recsys`,
что использует ноутбук и шаги DAG, — дублирования логики нет:

| Стадия | Скрипт | Выход |
|---|---|---|
| `prepare_data` | `scripts/prepare_data.py` | `data/events_train.parquet`, `events_labels.parquet`, `events_test.parquet`, `items.parquet` |
| `train` | `scripts/train.py` | `models/als_model.npz`, `models/catboost_ranker.cbm`, `data/candidates_test.parquet` |
| `evaluate` | `scripts/evaluate.py` | `cv_results/metrics.json` |

Стадия `train` не повторяет подбор Optuna — ALS обучается сразу с найденными
параметрами (`als_best` в `params.yaml`), поэтому `dvc repro` воспроизводит
уже готовую модель, а не поиск заново. Ремоут —
`s3://s3-student-mle-20260317-efc01cb482-freetrack/dvc_final` (`.dvc/config`),
ключи — в `.dvc/config.local` (не в git). Три выхода помечены `cache: false`
и потому не попадают в `dvc push`: `als_model.npz` не заливается по той же
причине, что и в DAG, а `catboost_ranker.cbm` и `metrics.json` лежат в git.

## Результаты экспериментов

Метрики на тесте (`top_k=10`), база усреднения — 383 пользователя с позитивом
в `test` и историей в `train` (см. «Бизнес-задача»):

| Модель | `precision@10` | `recall@10` | `map@10` | `ndcg@10` | `coverage@10` | `novelty@10` |
|---|---|---|---|---|---|---|
| Топ популярных | 0.0021 | 0.0070 | 0.0050 | 0.0075 | 0.0001 | 10.8550 |
| ALS (стартовые параметры) | 0.0060 | 0.0235 | 0.0070 | 0.0141 | 0.0025 | 11.6955 |
| ALS + Optuna | 0.0052 | 0.0257 | 0.0119 | 0.0180 | 0.0034 | 12.0094 |
| ALS + ранжировщик (финальная модель) | 0.0084 | 0.0397 | 0.0249 | 0.0327 | 0.0046 | 11.9801 |

Числа — из `notebooks/recommendations.ipynb`. `cv_results/metrics.json`,
который пересчитывает `dvc repro`, воспроизводит строки «Топ популярных» и
«ALS + ранжировщик» один в один, а строку `als` в нём — уже с параметрами
`als_best`, то есть она соответствует «ALS + Optuna», а не «ALS (стартовые
параметры)».

Финальная модель обходит и топ популярных, и одиночный ALS по всем метрикам
качества и по `coverage`; прирост от ранжирования объясняется порядком
выдачи — набор товаров тот же, что у ALS-кандидатов и топа популярных.
Абсолютные значения при этом низкие: добавления в корзину делают около 3 %
визитёров, каталог — 235 061 товар, окно оценки — 15 дней, а база усреднения
— несколько сотен пользователей, поэтому метрики шумные. Сравнивать модели
между собой это не мешает — база одна и та же; потолок пула кандидатов
(6.7 % позитивов теста, см. «Этап 3») показывает, что дальнейший рост
качества упирается в кандидатогенерацию, а не в ранжирование.

## Идентификация эксперимента в MLflow

* **Название эксперимента (Experiment Name):** `final_project_recsys`
* **ID эксперимента (Experiment ID):** `9`
* **Модель в Model Registry:** `recsys_ranker_model` — версия 1
  зарегистрирована ноутбуком, версия 2 — прогоном DAG `recsys_retrain` после
  успешной сверки качества (см. «Этап 5»)

## Имя бакета S3

`s3-student-mle-20260317-efc01cb482-freetrack`, эндпоинт
`https://storage.yandexcloud.net`. Артефакты проекта — под префиксом
`recsys_final/` (данные, модели, рекомендации), ремоут DVC — под
`dvc_final/`.

## Быстрый старт

```bash
git clone https://github.com/demetrioprotasov/mle-pr-final.git
cd mle-pr-final
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # заполнить своими доступами к Postgres и S3

# исходные csv датасета — в data/; события в Postgres, каталог товаров в S3
python scripts/load_events_to_pg.py --data-dir data
python scripts/build_items_catalog.py --data-dir data
bash run_mlflow_server.sh    # отдельный терминал, порт 5000

# дальше — оба ноутбука из notebooks/ сверху вниз, затем сервис
cd services
python download_recommendations.py
docker compose --env-file ../.env up --build -d
python test_service.py && python simulate_load.py
```

Сервис отвечает на `http://127.0.0.1:8010`, Grafana — `http://localhost:3000`
(анонимный просмотр), дашборд «Мониторинг сервиса рекомендаций». Airflow
поднимается из `airflow/`, DVC-пайплайн — из корня командой `dvc repro`.
Пошаговый запуск с проверками на каждом шаге — в
[Instructions.md](Instructions.md).

## API

| Метод | Путь | Поведение |
|---|---|---|
| GET | `/health` | `{"status": "ok", "model_loaded": 1}` |
| GET | `/metrics` | метрики Prometheus |
| GET | `/recommendations?user_id=&k=10` | блендинг: онлайн на нечётных позициях, офлайн на чётных, дедуп, обрезка до `k` |
| GET | `/recommendations_offline?user_id=&k=10` | персональные из `recommendations.parquet`, при отсутствии — топ популярных |
| GET | `/similar_items?item_id=&k=10` | похожие товары из `similar_items.parquet` |
| POST | `/events` | `{"user_id": int, "item_id": int}` — сохраняет событие в `EventStore` (в памяти, до 10 последних на пользователя) |
| GET | `/events?user_id=&k=10` | последние события пользователя |

`user_id`/`item_id` — `≥ 1`, `k` — от 1 до 20 (`recommendations.parquet`
содержит top-20 на пользователя, больше отдавать неоткуда); некорректные
значения отклоняются кодом 422.

```bash
curl -s "http://127.0.0.1:8010/health"
# {"status":"ok","model_loaded":1}

curl -s "http://127.0.0.1:8010/recommendations_offline?user_id=2&k=10"
# {"recs":[29940,259884,216305,449391,140853,48072,441734,342816,342264,154034]}

curl -s "http://127.0.0.1:8010/similar_items?item_id=461686&k=10"
# {"sim_item_id":[417316,462473,87085,425923,108561,102918,...],"score":[...]}
```

Ответы — реальные, из прогона `test_service.py` (`services/test_service.log`),
там же — негативные сценарии и проверка блендинга.

## Мониторинг

Обоснование каждой метрики и каждой панели — в
[Monitoring.md](Monitoring.md). Коротко: шесть панелей и три слоя метрик —
инфраструктурный (`process_resident_memory_bytes`,
`process_cpu_seconds_total`), реального времени (`recsys_requests_total`,
`recsys_request_duration_seconds`, `recsys_model_loaded`) и ML-прикладной
(`recsys_recs_source_total`, `recsys_cold_start_total`,
`recsys_recs_returned`, `recsys_events_stored_total`). Все собственные
метрики объявлены в `services/recsys_app/metrics.py`, Prometheus собирает их
каждые 5 секунд, источник данных и дашборд провижинятся автоматически.

Чтобы дашборд не был пустым: поднять сервис, прогнать `simulate_load.py` и
открыть `http://localhost:3000` с интервалом **Last 15 minutes** и
автообновлением **5s**. Скриншоты — `screenshots/grafana_dashboard.png` и
`screenshots/prometheus_targets.png` (страница Targets, `State=UP`).

## Воспроизводимость

Случайность зафиксирована во всех местах, где она есть:

- `SEED = 42` в `recsys/config.py` — общая константа проекта;
- `random_state=SEED` у `implicit.als.AlternatingLeastSquares` и
  `random_seed=SEED` у `CatBoostClassifier` (`recsys/models.py: fit_als`,
  `fit_ranker`);
- `optuna.samplers.TPESampler(seed=SEED)` при подборе гиперпараметров ALS;
- `random_state=seed` в `sample()` при негативном семплировании и
  `np.random.default_rng(seed)` при отборе отложенных пользователей
  (`recsys/features.py: sample_negatives, holdout_users`).

Гиперпараметры и границы сплитов не хардкодятся: для ноутбука и
DVC-пайплайна источник — `params.yaml`, для DAG — `airflow/params.yml`
(`als` в нём — это `als_best` из `params.yaml`). Версии всех библиотек
зафиксированы `==` в `requirements.txt`, `services/requirements.txt` и
`airflow/requirements-airflow.txt`.

## Ограничения

`EventStore` держит события в памяти процесса (до 10 последних на
пользователя) и теряет их при перезапуске сервиса — онлайн-часть блендинга не
переживает рестарт. Из свойств товаров используются только `categoryid` и
`available`: остальные анонимизированы числовыми id без словаря. Датасет
заморожен на 2015-09-18, поэтому DAG всегда переобучается на одном и том же
историческом окне. Модель ALS на всей истории (841 МБ) не заливается ни в S3,
ни в MLflow: канал наружу — около 0.1 МБ/с, заливка заняла бы больше двух
часов ради файла, который нужен только для расчёта рекомендаций; вместо неё
логируется манифест с путём и размером. Персональные рекомендации считаются
только для пользователей с историей от двух событий (406 020 из 1 407 580
визитёров), у остальных выдача — топ популярных. Оценка качества идёт на
383 пользователях — метрики от такой базы шумные, хотя и сопоставимы между
моделями. Grafana и веб-интерфейс Airflow открыты для анонимного просмотра
(роль Viewer) без входа — решение для демонстрации, не для продакшена.
