"""Уведомления в Telegram о новых инцидентах.

Бот и чат задаются переменными TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID (в
окружении или в .env проекта). Токен не попадает ни в журнал, ни в текст
ошибок: он входит в адрес запроса, поэтому сообщения об ошибках чистятся.
Текст уходит без разметки (parse_mode не задаётся): адреса и порты из
захвата не должны превращаться в ссылки или форматирование.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

TELEGRAM_URL = "https://api.telegram.org"
TIMEOUT_SECONDS = 20
MAX_MESSAGE = 4000  # лимит Telegram — 4096 символов


class NotifyError(RuntimeError):
    """Сообщение не отправлено (сеть, неверный токен или чат)."""


def telegram_settings() -> tuple[str, str] | None:
    """(токен, chat_id) или None, если уведомления не настроены."""
    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
    return (token, chat_id) if token and chat_id else None


def _clean(text: str, token: str) -> str:
    return text.replace(token, "***") if token else text


def send_telegram(text: str, token: str, chat_id: str, opener=urllib.request.urlopen) -> None:
    """Одно сообщение в чат; длинный текст обрезается до лимита Telegram."""
    if len(text) > MAX_MESSAGE:
        text = text[: MAX_MESSAGE - 1] + "…"
    request = urllib.request.Request(
        f"{TELEGRAM_URL}/bot{token}/sendMessage",
        data=json.dumps(
            {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(body).get("description") or body
        except ValueError:
            detail = body
        raise NotifyError(_clean(f"Telegram: HTTP {exc.code}: {detail[:300]}", token)) from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise NotifyError(_clean(f"Telegram недоступен: {exc}", token)) from None
    if not payload.get("ok"):
        raise NotifyError(_clean(f"Telegram отказал: {payload.get('description', payload)}", token))
