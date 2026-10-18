"""Генерация ключей и пересчёт подписанных checksums.json.

Первичная настройка:
    python make_checksums.py --generate-key
    python make_checksums.py

После изменения моделей:
    python make_checksums.py

Приватный ключ (keys/model_signing_key.pem) в репозиторий не коммитится.
Публичный (trust/model_public_key.pem) едет вместе с приложением.
"""

from __future__ import annotations

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.integrity import (
    IntegrityError,
    generate_keypair,
    private_key_path,
    public_key_path,
    write_checksums,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Подписанные checksums.json")
    parser.add_argument("--model-dir", nargs="+", default=["models", "models_v2"])
    parser.add_argument("--generate-key", action="store_true", help="Создать Ed25519-пару")
    parser.add_argument("--private-key", default=None)
    parser.add_argument("--public-key", default=None)
    parser.add_argument("--force", action="store_true", help="Перезаписать существующие ключи")
    parser.add_argument("--no-sign", action="store_true", help="Не подписывать")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.generate_key:
        try:
            fingerprint = generate_keypair(
                args.private_key, args.public_key, force=args.force
            )
        except IntegrityError as exc:
            print(f"Ошибка: {exc}")
            return 1
        print(f"Приватный ключ: {args.private_key or private_key_path()}")
        print(f"Публичный ключ: {args.public_key or public_key_path()}")
        print(f"Отпечаток: {fingerprint}")
        return 0

    failed = False
    unsigned = []
    for model_dir in args.model_dir:
        path = model_dir
        if not os.path.isabs(path):
            path = os.path.join(BASE_DIR, path)
        try:
            payload = write_checksums(
                path,
                sign=False if args.no_sign else None,
                key_path=args.private_key,
            )
        except (IntegrityError, OSError) as exc:
            print(f"{path}: ошибка — {exc}")
            failed = True
            continue

        signature = payload.get("signature")
        if signature:
            status = f"ed25519 {signature['fingerprint'][:16]}…"
        else:
            status = "БЕЗ ПОДПИСИ"
            unsigned.append(path)
        print(f"{path}: {len(payload['files'])} файлов, подпись: {status}")
        for name, digest in payload["files"].items():
            print(f"  {name}: {digest[:16]}…")

    if unsigned:
        print(
            "\nВНИМАНИЕ: часть checksums.json без подписи. "
            "Создайте ключ: python make_checksums.py --generate-key"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
