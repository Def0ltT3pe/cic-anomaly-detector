"""Контроль целостности и подпись файлов модели.

SHA-256 защищает от случайной подмены/повреждения `.pkl` (joblib выполняет
код при загрузке). Ed25519-подпись `checksums.json` защищает от подмены
самих хэшей: приватный ключ хранится вне образа (keys/), в приложении лежит
только публичный (trust/model_public_key.pem).

Если публичный ключ есть, а `checksums.json` не подписан — загрузка
блокируется (ALLOW_UNSIGNED_MODELS=1 отключает это требование).
VERIFY_MODEL_CHECKSUMS=0 полностью отключает проверку.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
from datetime import datetime, timezone

CHECKSUMS_FILE = "checksums.json"
MODEL_ARTIFACTS = [
    "isolation_forest.pkl",
    "scaler.pkl",
    "selector.pkl",
    "feature_cols_filtered.pkl",
    "bruteforce.pkl",
    "bruteforce.json",
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PRIVATE_KEY = os.path.join(BASE_DIR, "keys", "model_signing_key.pem")
DEFAULT_PUBLIC_KEY = os.path.join(BASE_DIR, "trust", "model_public_key.pem")


class IntegrityError(RuntimeError):
    """Хэш/подпись не совпали или ключи недоступны."""


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: dict) -> bytes:
    body = {
        "algorithm": payload.get("algorithm", "sha256"),
        "created_at": payload.get("created_at", ""),
        "files": payload.get("files", {}),
    }
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _resolve(path: str | None, default: str) -> str:
    path = path or default
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    return os.path.abspath(path)


def private_key_path() -> str:
    return _resolve(os.environ.get("MODEL_PRIVATE_KEY"), DEFAULT_PRIVATE_KEY)


def public_key_path() -> str:
    return _resolve(os.environ.get("MODEL_PUBLIC_KEY"), DEFAULT_PUBLIC_KEY)


def _fingerprint(public_key) -> str:
    from cryptography.hazmat.primitives import serialization

    pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(pem).hexdigest()


def generate_keypair(
    private_path: str | None = None,
    public_path: str | None = None,
    force: bool = False,
) -> str:
    """Создаёт Ed25519-пару. Приватный ключ — 0600, публичный — 0644."""
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as exc:
        raise IntegrityError(
            "Для подписи нужен пакет cryptography: pip install cryptography"
        ) from exc

    private_path = _resolve(private_path, DEFAULT_PRIVATE_KEY)
    public_path = _resolve(public_path, DEFAULT_PUBLIC_KEY)

    if not force and (os.path.exists(private_path) or os.path.exists(public_path)):
        raise IntegrityError(
            "Ключи уже существуют. Используйте --force, чтобы перезаписать."
        )

    private_key = Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    for path in (private_path, public_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(private_path, "wb") as handle:
        handle.write(private_pem)
    os.chmod(private_path, 0o600)

    with open(public_path, "wb") as handle:
        handle.write(public_pem)
    os.chmod(public_path, 0o644)

    return _fingerprint(private_key.public_key())


def _sign_payload(payload: dict, key_path: str) -> dict:
    if not os.path.isabs(key_path):
        key_path = os.path.join(BASE_DIR, key_path)
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as exc:
        raise IntegrityError(
            "Для подписи нужен пакет cryptography: pip install cryptography"
        ) from exc

    try:
        with open(key_path, "rb") as handle:
            private_key = serialization.load_pem_private_key(
                handle.read(), password=None
            )
    except OSError as exc:
        raise IntegrityError(f"Не удалось прочитать ключ {key_path}: {exc}") from exc

    if not isinstance(private_key, Ed25519PrivateKey):
        raise IntegrityError(f"{key_path} — не Ed25519-ключ.")

    signature = private_key.sign(_canonical(payload))
    return {
        "algorithm": "ed25519",
        "fingerprint": _fingerprint(private_key.public_key()),
        "value": base64.b64encode(signature).decode("ascii"),
    }


def write_checksums(
    model_dir: str,
    filenames: list[str] | None = None,
    sign: bool | None = None,
    key_path: str | None = None,
) -> dict:
    """Считает checksums.json и подписывает его, если доступен ключ."""
    model_dir = os.path.abspath(model_dir)
    names = filenames or MODEL_ARTIFACTS

    files = {}
    for name in names:
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            files[name] = sha256_file(path)

    if not files:
        raise IntegrityError(f"В {model_dir} нет файлов модели для проверки.")

    payload = {
        "algorithm": "sha256",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": files,
    }

    if sign is None:
        sign = os.path.isfile(key_path or private_key_path())
    if sign:
        payload["signature"] = _sign_payload(payload, key_path or private_key_path())

    with open(
        os.path.join(model_dir, CHECKSUMS_FILE), "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def _verify_signature(payload: dict, signature: dict, public_path: str) -> None:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        raise IntegrityError(
            "Для проверки подписи нужен пакет cryptography: pip install cryptography"
        ) from exc

    try:
        signature_bytes = base64.b64decode(signature["value"], validate=True)
    except (KeyError, TypeError, ValueError, binascii.Error) as exc:
        raise IntegrityError(f"Некорректная подпись в {CHECKSUMS_FILE}: {exc}") from exc

    try:
        with open(public_path, "rb") as handle:
            public_key = serialization.load_pem_public_key(handle.read())
    except OSError as exc:
        raise IntegrityError(f"Не удалось прочитать публичный ключ {public_path}: {exc}") from exc

    if not isinstance(public_key, Ed25519PublicKey):
        raise IntegrityError(f"{public_path} — не Ed25519-ключ.")

    expected_fingerprint = signature.get("fingerprint")
    if expected_fingerprint and not hmac.compare_digest(
        expected_fingerprint, _fingerprint(public_key)
    ):
        raise IntegrityError("Отпечаток публичного ключа не совпадает с подписью.")

    try:
        public_key.verify(signature_bytes, _canonical(payload))
    except InvalidSignature as exc:
        raise IntegrityError(
            f"Подпись {CHECKSUMS_FILE} недействительна — файлы изменены."
        ) from exc


def verify_checksums(model_dir: str, filenames: list[str]) -> dict | None:
    """Проверяет подпись и хэши. None, если проверка отключена/нет файла."""
    if os.environ.get("VERIFY_MODEL_CHECKSUMS", "1").strip() == "0":
        return None

    model_dir = os.path.abspath(model_dir)
    checksums_path = os.path.join(model_dir, CHECKSUMS_FILE)
    if not os.path.isfile(checksums_path):
        return None

    try:
        with open(checksums_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        expected = payload["files"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise IntegrityError(f"Не удалось прочитать {checksums_path}: {exc}") from exc

    public_path = public_key_path()
    signature = payload.get("signature")

    if signature:
        if not os.path.isfile(public_path):
            raise IntegrityError(
                f"Есть подпись, но нет публичного ключа {public_path}. "
                "Задайте MODEL_PUBLIC_KEY."
            )
        _verify_signature(payload, signature, public_path)
    elif os.path.isfile(public_path) and os.environ.get(
        "ALLOW_UNSIGNED_MODELS", "0"
    ).strip() != "1":
        raise IntegrityError(
            f"{CHECKSUMS_FILE} не подписан, а публичный ключ есть. "
            "Подпишите модели: python make_checksums.py"
        )

    problems = []
    for path in filenames:
        full_path = path if os.path.isabs(path) else os.path.join(model_dir, path)
        name = os.path.basename(full_path)
        if not os.path.isfile(full_path):
            problems.append(f"{name}: файл отсутствует")
            continue
        if name not in expected:
            problems.append(f"{name}: нет записи в {CHECKSUMS_FILE}")
            continue
        actual = sha256_file(full_path)
        if actual != expected[name]:
            problems.append(f"{name}: SHA-256 не совпадает")

    if problems:
        raise IntegrityError(
            "Проверка целостности модели не пройдена: "
            + "; ".join(problems)
            + ". Если модели заменены намеренно, пересчитайте и подпишите checksums: "
            "python make_checksums.py"
        )
    return payload
