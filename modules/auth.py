"""Пользователи, роли и пароли.

Пароли хранятся как PBKDF2-HMAC-SHA256 (600 000 итераций, соль 16 байт).
Файл пользователей — JSON вне репозитория (config/users.json), права 0600.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROLES = ("admin", "analyst", "viewer")
DEFAULT_ROLE = "analyst"
ITERATIONS = 600_000
SALT_BYTES = 16
MIN_PASSWORD_LENGTH = 8
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{2,32}$")

_DUMMY_HASH: str | None = None


class AuthError(RuntimeError):
    """Ошибка конфигурации пользователей."""


@dataclass(frozen=True)
class User:
    username: str
    role: str


def users_path() -> str:
    path = os.environ.get("USERS_FILE") or os.path.join(BASE_DIR, "config", "users.json")
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    return os.path.abspath(path)


def _b64encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"), validate=True)


def hash_password(password: str, iterations: int = ITERATIONS) -> str:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Пароль должен быть не короче {MIN_PASSWORD_LENGTH} символов.")
    salt = secrets.token_bytes(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64encode(salt)}${_b64encode(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations_text, salt_text, digest_text = encoded.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_text)
        if not 10_000 <= iterations <= 5_000_000:
            return False
        salt = _b64decode(salt_text)
        expected = _b64decode(digest_text)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError, binascii.Error):
        return False


def _dummy_hash() -> str:
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password("dummy-password-for-timing")
    return _DUMMY_HASH


def load_users(path: str | None = None) -> dict[str, dict]:
    path = path or users_path()
    if not os.path.isfile(path):
        return {}

    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        users = payload["users"]
        if not isinstance(users, dict):
            raise TypeError("users должен быть объектом")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise AuthError(f"Не удалось прочитать {path}: {exc}") from exc

    for username, record in users.items():
        if not isinstance(record, dict) or "password" not in record:
            raise AuthError(f"Некорректная запись пользователя {username!r}")
        if record.get("role") not in ROLES:
            record["role"] = DEFAULT_ROLE
    return users


def save_users(users: dict[str, dict], path: str | None = None) -> str:
    path = path or users_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"version": 1, "users": users}

    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    return path


def add_user(
    username: str,
    password: str,
    role: str = DEFAULT_ROLE,
    path: str | None = None,
) -> str:
    username = username.strip()
    if not USERNAME_RE.match(username):
        raise AuthError("Логин: 2-32 символа, только A-Z, a-z, 0-9, _, ., -")
    if role not in ROLES:
        raise AuthError(f"Роль должна быть одной из: {', '.join(ROLES)}")

    users = load_users(path)
    if username in users:
        raise AuthError(f"Пользователь {username!r} уже существует.")
    users[username] = {
        "role": role,
        "password": hash_password(password),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return save_users(users, path)


def set_password(username: str, password: str, path: str | None = None) -> str:
    users = load_users(path)
    if username not in users:
        raise AuthError(f"Пользователь {username!r} не найден.")
    users[username]["password"] = hash_password(password)
    users[username]["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return save_users(users, path)


def remove_user(username: str, path: str | None = None) -> str:
    users = load_users(path)
    if username not in users:
        raise AuthError(f"Пользователь {username!r} не найден.")
    del users[username]
    return save_users(users, path)


def authenticate(username: str, password: str, path: str | None = None) -> User | None:
    users = load_users(path)
    username = (username or "").strip()
    record = users.get(username)

    if record is None:
        # Выравниваем время ответа, чтобы нельзя было перебирать логины
        verify_password(password, _dummy_hash())
        return None

    if not verify_password(password, record["password"]):
        return None

    role = record.get("role", DEFAULT_ROLE)
    if role not in ROLES:
        role = DEFAULT_ROLE
    return User(username=username, role=role)


def can_use_llm(role: str | None) -> bool:
    return role in ("admin", "analyst")


def can_view_audit(role: str | None) -> bool:
    return role == "admin"
