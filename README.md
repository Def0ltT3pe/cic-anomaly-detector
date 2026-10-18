# Детектор аномалий сетевого трафика

**Версия 1.0.0** · Python 3.12 · MIT

Поиск аномалий в сетевом трафике (CIC-IDS2017): IsolationForest для общих
аномалий и отдельный supervised-детектор для BruteForce. Есть веб-интерфейс
на Streamlit, CLI для пакетной обработки и опциональные пояснения на русском
через Gemini.

## Основа проекта

Проект вырос из ранее написанного учебного проекта
[**Practik-2**](https://github.com/Def0ltT3pe/Practik-2). Из него взяты:

- подготовка датасета CIC-IDS2017 и исходный `dataset.py`;
- обученная модель IsolationForest (`models/isolation_forest.pkl`,
  `scaler.pkl`, `selector.pkl`, `feature_cols_filtered.pkl`);
- первые версии Streamlit-приложения (`App.py`) и CLI (`detector.py`).

В этой версии исходный код переработан: добавлены калибровка порога,
supervised-детектор BruteForce, Docker, пользователи и роли, аудит,
Ed25519-подпись моделей, тесты и CI. История изменений — в
[CHANGELOG.md](CHANGELOG.md).

## Состав

```
App.py                      Streamlit-приложение
detector.py                 CLI: CSV/Parquet -> текстовый отчёт
calibrate_model.py          расчёт порога по benign-трафику (FPR)
train_model_v2.py           эксперимент: обучение на benign нескольких дней
train_bruteforce.py         обучение supervised-детектора BruteForce
make_checksums.py           SHA-256 + Ed25519-подпись моделей
manage_users.py             пользователи и роли (PBKDF2)
modules/model_loader.py     загрузка модели, статистик и каталога моделей
modules/integrity.py        контроль целостности и подпись моделей
modules/auth.py             пользователи, роли, пароли
modules/audit.py            журнал аудита JSONL
modules/calibration.py      порог, FPR и percentile
modules/bruteforce.py       supervised-детектор BruteForce
modules/pipeline.py         валидация данных и детекция (общая логика)
modules/explainer.py        z-оценки и текстовые отклонения
modules/llm_explain.py      Gemini (ключ из env/secrets, не из кода)
models/                     рабочая модель, статистики, калибровка, BruteForce
models_v2/                  экспериментальная модель (по умолчанию не используется)
trust/                      публичный ключ подписи моделей
keys/                       приватный ключ (не коммитится)
config/                     users.json (не коммитится)
audit/                      audit.jsonl (не коммитится)
```

## Пользователи и роли

Пользователи хранятся в `config/users.json` (PBKDF2-HMAC-SHA256, 600 000
итераций). Файл создаётся командой:

```bash
python manage_users.py add admin --role admin
python manage_users.py add alice --role analyst
python manage_users.py list
python manage_users.py passwd alice
python manage_users.py remove alice
```

| Роль | Анализ | Пояснения ИИ | Журнал аудита |
|---|---|---|---|
| `admin` | да | да | да |
| `analyst` | да | да | нет |
| `viewer` | да | нет | нет |

Если `users.json` нет, но задан `APP_PASSWORD` — вход по общему паролю
(роль `admin`). Если нет ни того, ни другого — приложение работает в
открытом режиме `anonymous/admin` и показывает предупреждение. Для
production задайте пользователей или `APP_PASSWORD`.

После 5 неудачных попыток вход блокируется на 60 секунд в рамках сессии;
все попытки пишутся в аудит.

## Аудит

События пишутся в `audit/audit.jsonl` по строке на событие:

```json
{"ts":"2026-10-18T22:40:03+00:00","user":"alice","role":"analyst",
 "action":"analyze","rows":389714,"anomalies":6449,"brute":9513,"threshold":-0.0638}
```

Пишутся: `login`, `login_failed`, `logout`, `upload`, `analyze`,
`llm_explain`, `model_warning`, изменения пользователей из CLI. Пароли,
токены и ключи вырезаются. Админ видит последние 20 событий в сайдбаре.

`AUDIT_LOG` задаёт путь, `AUDIT_ENABLED=0` отключает запись. Файл создаётся
с правами 0600; ротации нет — для долгой эксплуатации настройте logrotate.

## Быстрый старт (Docker, Linux)

```bash
cp .env.example .env      # при желании впишите GEMINI_API_KEY и APP_PASSWORD
docker compose up --build -d
```

Откройте `http://<host>:8501`.

Пользователи читаются из `./config/users.json` (создайте заранее командой
`python manage_users.py add ...`), аудит лежит в named volume `audit_data`.
Если `users.json` нет и `APP_PASSWORD` пуст — приложение откроется в
режиме `anonymous/admin` с предупреждением.

```bash
docker compose ps
docker inspect --format '{{.State.Health.Status}}' cic_anomaly_detector
```

Остановить: `docker compose down`.

### Без docker compose

```bash
docker build -t cic-anomaly-detector .
docker run -d --name cic_anomaly_detector -p 8501:8501 \
  -v "$PWD/config:/app/config:ro" -v cic_audit:/app/audit \
  -e GEMINI_API_KEY= -e APP_PASSWORD= cic-anomaly-detector
```

## Локальный запуск

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:     .venv\Scripts\activate
pip install -r requirements.txt
streamlit run App.py
```

Модели и `normal_stats.json` уже лежат в `models/`, тренировочный parquet
для работы не нужен.

## Переменные окружения

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `GEMINI_API_KEY` | пусто | Ключ Gemini. Без него детекция работает, пояснения ИИ — нет |
| `APP_PASSWORD` | пусто | Общий пароль, если нет `users.json` |
| `USERS_FILE` | `config/users.json` | Файл пользователей |
| `AUDIT_LOG` | `audit/audit.jsonl` | Журнал аудита |
| `AUDIT_ENABLED` | `1` | `0` — не писать аудит |
| `MODEL_DIR` | `models` | Каталог модели: `models` или `models_v2` |
| `VERIFY_MODEL_CHECKSUMS` | `1` | `0` — отключить проверку подписи/хэшей |
| `MODEL_PUBLIC_KEY` | `trust/model_public_key.pem` | Публичный ключ Ed25519 |
| `MODEL_PRIVATE_KEY` | `keys/model_signing_key.pem` | Приватный ключ (только подпись) |
| `ALLOW_UNSIGNED_MODELS` | `0` | `1` — разрешить неподписанные checksums |
| `APP_BIND` | `0.0.0.0` | Адрес публикации порта (`127.0.0.1` — только локально) |
| `APP_PORT` | `8501` | Внешний порт |

Для локального Streamlit ключ можно положить в `.streamlit/secrets.toml`
(шаблон — `.streamlit/secrets.toml.example`). Файл не коммитится.

## Формат данных

CSV или Parquet. Нужны те же **67 признаков**, на которых обучалась модель
(список — `models/feature_cols_filtered.pkl`), например:
`Protocol`, `Flow Duration`, `Total Fwd Packets`, ..., `Idle Min`.

- Колонка `Label` необязательна; если есть — показывается как истинная метка.
- Лишние колонки игнорируются.
- `inf` и `NaN` заменяются средними значениями нормы, количество замен
  выводится в интерфейсе и отчёте.
- Текстовые значения в признаках — ошибка с перечнем колонок.
- Ограничение: 2 000 000 строк на файл.

## Калибровка порога

`models/calibration.json` задаёт порог по benign-трафику: аномалия — это
`score <= threshold`, где `score = decision_function`. Порог выбран по
целевому FPR, а не по `contamination` модели. В интерфейсе и отчёте
показывается score и «аномальнее N% нормы».

Замеры на 120 000 benign-строк каждого дня:

| Целевой FPR | FPR Пн | FPR Вт | FPR Ср | Recall DoS | Recall BruteForce |
|---|---|---|---|---|---|
| 1% | 0.66% | 0.64% | 1.70% | 0.210 | 0.000 |
| **2% (по умолчанию)** | 1.66% | 1.71% | 2.63% | **0.708** | 0.000 |
| 5% | 4.46% | 5.09% | 5.45% | 0.762 | 0.000 |

Итог на полном DoS-Wednesday при пороге 2%: precision 0.930, recall 0.708,
FPR 2.60%. Против встроенного `contamination=0.05` (FPR 5.9%, recall 0.77)
это меньше ложных срабатываний при небольшой потере recall.

Пересчитать на своих данных:

```bash
python calibrate_model.py --data Benign-Monday-no-metadata.parquet \
    Bruteforce-Tuesday-no-metadata.parquet DoS-Wednesday-no-metadata.parquet \
    --target-fpr 0.02
```

Порог 1% брать не стоит: recall DoS падает до 0.21.

## Качество модели

- **DoS-атаки**: IsolationForest ловит recall ~0.74 при FPR 2% на отложенном дне.
- **BruteForce**: IsolationForest не ловит (recall 0), но отдельный
  supervised-детектор даёт recall 0.999 при FPR 0.1%; см. ниже.
- `VarianceThreshold` в исходном обучении применялся после `StandardScaler`,
  поэтому не отбрасывал ничего (67 из 67). Применять его до scaler с порогом
  0.01 тоже плохо: он выкидывает редкие, но важные `RST/ECE/FIN`-флаги.
  Поэтому в v2 селектор убран; на метрики это не влияет.
- Переобучение на benign трёх дней **не улучшает** IsolationForest: cross-day
  recall 0.743–0.746 при том же FPR, что и у модели только на понедельнике.

### BruteForce-детектор (supervised)

`models/bruteforce.pkl` — RandomForest на размеченных потоках CICIDS Tuesday.
Он не заменяет IsolationForest, а добавляет второй, независимый сигнал:
в интерфейсе и отчёте есть отдельная секция «Подозрения на BruteForce».

| Метрика | Значение |
|---|---|
| AUC на holdout | 1.000 |
| Recall (порог 0.354) | 0.999 |
| Precision | 0.960 |
| FPR на Tuesday holdout | 0.101% |
| FPR на Monday benign | 0.103% |
| FPR на Wednesday benign | 0.180% |
| Размер модели | 0.7 МБ |

Порог 0.354 выбран по целевому FPR 0.1% на benign. Модель специализированная:
DoS она почти не видит (recall 0.002), и наоборот — это ожидаемо.

**Важно:** модель обучена на CICIDS и для другой сети требует переобучения
на её размеченных данных. Без `Label` обучить нельзя. Recall на независимом
дне проверить нечем: брутфорс в CICIDS есть только во вторник; FPR проверен
на понедельнике и среде.

```bash
python train_bruteforce.py \
    --data Bruteforce-Tuesday-no-metadata.parquet \
    --eval Benign-Monday-no-metadata.parquet DoS-Wednesday-no-metadata.parquet \
    --target-fpr 0.001
```

### Экспериментальная модель v2

`models_v2/` — обучена на benign понедельника, вторника и среды, без
селектора, с калибровкой 2%. Метрики IsolationForest: DoS recall 0.740,
FPR 2.0%; BruteForce по-прежнему 0, его закрывает отдельный supervised-детектор.
По качеству v2 эквивалентна `models/`. Включить:

```bash
MODEL_DIR=models_v2 streamlit run App.py
# или
python detector.py --model-dir models_v2 --input traffic.csv
```

Переобучить: `python train_model_v2.py --data ... --eval ...`.

## CLI

```bash
python detector.py --input traffic.csv --output report.txt
python detector.py --input traffic.parquet --output report.txt --max_anomalies 10
python detector.py --input traffic.csv --output report.txt --use_llm
```

`--use_llm` без ключа не падает: отчёт формируется без пояснений ИИ.
Аномалии в отчёте отсортированы по score (сильнейшие первыми).

## Статистики нормы

`models/normal_stats.json` — средние и стандартные отклонения benign-трафика
(458 831 строка, 67 признаков). Пересчитать из parquet:

```python
from modules.model_loader import load_model_objects, load_normal_stats

_, _, _, cols = load_model_objects()
load_normal_stats(cols, parquet_path="Benign-Monday-no-metadata.parquet", rebuild=True)
```

Parquet в образ не входит: для инференса достаточно JSON.

## Целостность и подпись моделей

`models/checksums.json` хранит SHA-256 файлов модели и Ed25519-подпись самого
файла проверок. При загрузке `model_loader` и `bruteforce` сначала проверяют
подпись, затем хэши. Подмена `.pkl`, подмена хэша и неподписанный
`checksums.json` (при наличии публичного ключа) блокируются.

```bash
python make_checksums.py --generate-key   # один раз: keys/ + trust/
python make_checksums.py                  # пересчитать и подписать
python make_checksums.py --model-dir models
```

- Приватный ключ `keys/model_signing_key.pem` — 0600, в git и Docker не
  попадает. Публичный `trust/model_public_key.pem` едет с приложением.
- Если переобучаете модели, подпишите их тем же ключом. Если ключа нет —
  сгенерируйте свою пару и замените публичный ключ в `trust/`.
- Отпечаток ключа можно сравнить с `signature.fingerprint` в
  `checksums.json`.
- `VERIFY_MODEL_CHECKSUMS=0` отключает проверку, `ALLOW_UNSIGNED_MODELS=1`
  разрешает неподписанные проверки.

Граница доверия: подпись защищает каталог `models/` от подмены, пока
приватный ключ хранится отдельно. Если злоумышленник может менять и код,
и `trust/`, подпись не поможет — это уже компрометация хоста.

## Пояснения Gemini

Основной SDK — `google-genai` (ставится из `requirements.txt`). Если он не
установлен, код автоматически использует legacy `google-generativeai`.
Ключ берётся из `GEMINI_API_KEY` или `.streamlit/secrets.toml`; в исходниках
ключа нет. Без ключа детекция и отчёты работают, пояснения ИИ отключаются.

## Безопасность

- Ключей в репозитории быть не должно. Если ключ попал в git — **отзовите его**,
  удаление файла не помогает: ключ остаётся в истории.
- Пользователи и роли — в `config/users.json` (PBKDF2, 600k итераций).
  Без пользователей и `APP_PASSWORD` приложение открыто; для публичного
  сервера задайте пользователей, пароль и reverse proxy с TLS.
- Аудит пишется в `audit/audit.jsonl`; админ видит последние события.
  Блокировка после 5 неудачных попыток действует в рамках сессии.
- Модели — pickle-файлы: подменённый `.pkl` может выполнить код. Держите
  `models/` только на чтение и не отключайте подпись без причины.
- Данные трафика при включённом `--use_llm` уходят в Google. Для
  конфиденциального трафика LLM лучше отключить.

## Проверено

- `AppTest`: приложение стартует без исключений, 67 признаков, порог из
  `calibration.json` и BruteForce-модель подхватываются для `models/` и
  `models_v2/` (для v2 модель BruteForce берётся из `models/`).
- `streamlit run` + `GET /_stcore/health` → 200.
- Вход по `users.json`: неверный пароль, `analyst`, `viewer`, `admin`;
  fallback на `APP_PASSWORD`; открытый режим с предупреждением.
- `manage_users.py add/list/passwd` и аудит `user_add`.
- Аудит: `login`, `login_failed`, `logout`; админ видит журнал.
- Подпись: подмена `.pkl`, подмена хэша и неподписанный `checksums.json`
  блокируются; валидные подписи `models/` и `models_v2/` проходят.
- CLI: CSV и Parquet, запуск из другого каталога, `--model-dir models_v2`.
- Неполный набор колонок, текст в признаках, пустой файл, `inf`/`NaN`,
  `std=0` — обрабатываются без трейсбеков.
- Полный прогон на DoS-Wednesday: precision 0.930, recall 0.708, FPR 2.60%.
- Полный прогон на BruteForce-Tuesday: 9 513 подозрений из 389 714 строк
  (recall 0.999, precision ≈ 0.96, FPR ≈ 0.10%).
- Подмена `scaler.pkl` обнаружена проверкой SHA-256; `bruteforce.pkl` и
  `bruteforce.json` также входят в `checksums.json`.
- Новый путь `google-genai` проверен на mock-модуле (single и stream);
  fallback `google-generativeai` выбран локально.

## Тесты

```bash
python -m unittest discover -s tests -v
```

29 тестов: валидация признаков, explainer, калибровка, пароли/роли,
Ed25519-подпись (включая подмену файла, хэша и подписи), версия.
Тесты не требуют parquet, API-ключей и Docker.

CI: `.github/workflows/ci.yml` — Python 3.12, установка зависимостей,
`compileall` и `unittest` на каждый push/PR.

## Документы

- [CHANGELOG.md](CHANGELOG.md) — изменения по версиям.
- [SECURITY.md](SECURITY.md) — политика безопасности и развёртывание.
- [LICENSE](LICENSE) — MIT.

## Лицензия

MIT. Copyright (c) 2026 Def0ltT3pe.
