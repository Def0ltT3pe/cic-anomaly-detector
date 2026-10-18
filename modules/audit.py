"""Журнал аудита в JSONL.

Одна строка — одно событие: время, пользователь, роль, действие, детали.
Секреты (пароли, токены, ключи) вырезаются перед записью.
Ошибки записи не ломают приложение: пишем warning в лог и продолжаем.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_AUDIT_PATH = os.path.join(BASE_DIR, "audit", "audit.jsonl")

_SENSITIVE_PARTS = ("password", "secret", "token", "api_key", "apikey", "authorization")
_lock = threading.Lock()
logger = logging.getLogger(__name__)


def audit_path() -> str:
    path = os.environ.get("AUDIT_LOG") or DEFAULT_AUDIT_PATH
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    return os.path.abspath(path)


def audit_enabled() -> bool:
    return os.environ.get("AUDIT_ENABLED", "1").strip() != "0"


def _sanitize(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if any(part in str(key).lower() for part in _SENSITIVE_PARTS):
                result[key] = "***"
            else:
                result[key] = _sanitize(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def log_event(user: str | None, role: str | None, action: str, **details) -> None:
    if not audit_enabled():
        return

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "user": user or "anonymous",
        "role": role or "-",
        "action": action,
    }
    entry.update(_sanitize(details))

    try:
        line = json.dumps(entry, ensure_ascii=False, default=str)
        path = audit_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _lock:
            descriptor = os.open(
                path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600
            )
            with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except OSError as exc:
        logger.warning("Не удалось записать аудит в %s: %s", audit_path(), exc)


def read_recent(limit: int = 20) -> list[dict]:
    path = audit_path()
    if limit < 1 or not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as handle:
            lines = deque(handle, maxlen=limit)
    except OSError as exc:
        logger.warning("Не удалось прочитать аудит %s: %s", path, exc)
        return []

    entries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except ValueError:
            continue
    return entries
