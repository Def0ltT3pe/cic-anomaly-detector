"""Проверка новых захватов с серверов и уведомления в Telegram об инцидентах.

Запускается после pull_captures.ps1 (он вызывает его сам) или вручную:

    python monitor.py                      # проверить новые файлы, отправить уведомления
    python monitor.py --dry-run            # проверить, но ничего не отправлять
    python monitor.py --test-notify        # проверить настройки Telegram

Серверы и папки берутся из pull_captures.json; модель по умолчанию —
models_flagship (поле "model" у сервера её переопределяет). Токен бота и
чат — TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID в окружении или в .env проекта.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.alerts import META_COLUMNS, detect_local_ip, has_meta  # noqa: E402
from modules.bruteforce import load_bruteforce  # noqa: E402
from modules.calibration import load_calibration  # noqa: E402
from modules.model_loader import load_model_objects, load_normal_stats, resolve_model_dir  # noqa: E402
from modules.monitor import FLAGGED_COLUMNS, load_servers, run, state_dir  # noqa: E402
from modules.notify import NotifyError, send_telegram, telegram_settings  # noqa: E402
from modules.pipeline import detect_anomalies  # noqa: E402
from pcap_to_flows import ConversionError, convert_pcap, find_tshark  # noqa: E402

DEFAULT_MODEL = "models_flagship"


def load_dotenv(path: str) -> None:
    """KEY=VALUE из .env; уже заданные переменные окружения не перезаписываются.
    Нужно Планировщику задач: у его задачи нет переменных из .env."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value


class Analyzer:
    """pcap -> тревожные потоки классификатора атак; модели грузятся один раз."""

    def __init__(self, default_model: str, tshark: str):
        self.default_model = default_model
        self.tshark = tshark
        self._models = {}

    def _assets(self, model_dir: str):
        model_dir = resolve_model_dir(model_dir)
        if model_dir not in self._models:
            model, scaler, selector, feature_cols = load_model_objects(model_dir)
            means, _ = load_normal_stats(feature_cols, model_dir=model_dir)
            brute = load_bruteforce(model_dir)
            if brute is None:
                raise RuntimeError(f"в {model_dir} нет классификатора атак (bruteforce.pkl)")
            self._models[model_dir] = (model, scaler, selector, feature_cols, means,
                                       load_calibration(model_dir), brute)
        return self._models[model_dir]

    def __call__(self, path: str, server):
        model, scaler, selector, feature_cols, means, calibration, brute = self._assets(
            server.model_dir or self.default_model
        )
        flows, _ = convert_pcap(path, self.tshark)
        if flows.empty or not has_meta(flows):
            return flows.reindex(columns=FLAGGED_COLUMNS).iloc[:0], None
        result = detect_anomalies(flows, model, scaler, selector, feature_cols,
                                  fill_values=means, calibration=calibration)
        proba = brute.predict_proba(result.X)
        flagged = result.df.loc[proba >= brute.threshold, list(META_COLUMNS)].copy()
        flagged["proba"] = proba[proba >= brute.threshold].astype(np.float64)
        return flagged.reset_index(drop=True), detect_local_ip(flows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=os.path.join(BASE_DIR, "pull_captures.json"),
                        help="Список серверов (как у pull_captures.ps1)")
    parser.add_argument("--state-dir", default=None, help="Куда писать историю (по умолчанию monitor/)")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL, help="Модель по умолчанию")
    parser.add_argument("--tshark", default=None, help="Путь к tshark")
    parser.add_argument("--backfill-days", type=float, default=1.0,
                        help="Первый запуск: проверять файлы не старше N суток")
    parser.add_argument("--dry-run", action="store_true", help="Не отправлять уведомления")
    parser.add_argument("--log-file", default=None, help="Дописывать журнал в этот файл (UTF-8)")
    parser.add_argument("--test-notify", action="store_true", help="Отправить тестовое сообщение и выйти")
    return parser


LOG_FILE = None


def log(message: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}"
    print(line, flush=True)
    if LOG_FILE:
        # Журнал pull_captures.ps1: пишем сами в UTF-8, иначе PowerShell
        # перекодирует вывод Python в кодовую страницу консоли
        with open(LOG_FILE, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def main() -> int:
    global LOG_FILE
    args = build_parser().parse_args()
    LOG_FILE = args.log_file
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    settings = telegram_settings()

    if args.test_notify:
        if settings is None:
            print("Telegram не настроен: задайте TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID (.env).")
            return 2
        try:
            send_telegram("Мониторинг трафика: проверка уведомлений.", *settings)
        except NotifyError as exc:
            print(exc)
            return 1
        print("Тестовое сообщение отправлено.")
        return 0

    try:
        servers = load_servers(args.config, os.environ.get("TRUSTED_IPS", ""))
    except (OSError, ValueError) as exc:
        log(f"Не прочитан список серверов: {exc}")
        return 2

    notify = None
    if args.dry_run:
        log("Пробный прогон: уведомления не отправляются.")
    elif settings is None:
        log("Telegram не настроен (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID): только история.")
    else:
        notify = lambda text: send_telegram(text, *settings)  # noqa: E731

    try:
        tshark = find_tshark(args.tshark or os.environ.get("TSHARK") or None)
    except ConversionError as exc:
        log(str(exc))
        return 2
    analyzer = Analyzer(args.model_dir, tshark)
    summary = run(servers, analyzer, notify, directory=args.state_dir or state_dir(),
                  backfill_days=args.backfill_days, log=log)
    log(f"Проверено файлов: {summary.files}, тревожных потоков: {summary.flagged}, "
        f"инцидентов: {summary.incidents} (новых {summary.new_incidents}, "
        f"отправлено {summary.notified})")
    return 1 if summary.errors else 0


if __name__ == "__main__":
    sys.exit(main())
