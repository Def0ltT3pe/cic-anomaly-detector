# Changelog

## [1.0.0] - 2026-10-19

Первый публичный релиз. Проект основан на ранее написанном
[Practik-2](https://github.com/Def0ltT3pe/Practik-2): оттуда взяты
подготовка CIC-IDS2017, исходная модель IsolationForest и первые версии
Streamlit/CLI. В этой версии пайплайн переработан и дополнен.

### Added

- Гибридная детекция: IsolationForest для общих аномалий и supervised
  RandomForest для BruteForce.
- Калибровка порога по benign-трафику (`calibration.json`, целевой FPR 2%),
  score и percentile в UI/CLI.
- Веб-интерфейс Streamlit: загрузка CSV/Parquet, метрики, таблицы аномалий,
  пояснения Gemini.
- CLI `detector.py` с отчётом, `--use_llm`, `--model-dir`.
- Пользователи и роли (`admin`/`analyst`/`viewer`), PBKDF2, блокировка
  после 5 неудачных попыток.
- Журнал аудита JSONL (`audit/audit.jsonl`) с санитайзингом секретов.
- Подпись моделей Ed25519 (`checksums.json`), проверка при загрузке.
- Dockerfile (non-root, healthcheck) и docker-compose.
- Тесты на `unittest`, CI GitHub Actions.

### Fixed

- Пути к моделям и статистикам больше не зависят от текущего каталога.
- `inf`/`NaN`/текстовые значения и пустые файлы обрабатываются с понятными
  ошибками, а не трейсбеками.
- Убран обязательный тренировочный parquet: статистики лежат в JSON.
- Исправлен `CMD` Docker (`App.py`), добавлен `pyarrow`.
- Состояние анализа сбрасывается при загрузке нового файла.
- `detector.py` больше не пишет отчёт в `~/Desktop` и не падает без LLM.

### Security

- Удалены захардкоженные ключи Gemini; ключ читается из `GEMINI_API_KEY`
  или `.streamlit/secrets.toml`.
- Добавлены `.gitignore`/`.dockerignore`, приватный ключ подписи не
  попадает в репозиторий и образ.
- Модели проверяются по SHA-256 и Ed25519-подписи.
- Пароли не хранятся в открытом виде; аудит не пишет секреты.

### Known limitations

- BruteForce-модель обучена на CICIDS Tuesday; для другой сети нужно
  переобучение на размеченных данных.
- Нет MFA/SSO и ротации журнала аудита.
- Подпись Ed25519 хранится на хосте, не в HSM.
