"""Управление пользователями приложения.

Примеры:
    python manage_users.py add alice --role analyst
    python manage_users.py passwd alice
    python manage_users.py list
    python manage_users.py remove alice --yes

Пароль вводится скрыто; флаг --password нужен только для автоматизации.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.audit import log_event
from modules.auth import (
    AuthError,
    ROLES,
    add_user,
    load_users,
    remove_user,
    set_password,
    users_path,
)


def _read_password(args) -> str:
    if args.password:
        return args.password
    first = getpass.getpass("Пароль: ")
    second = getpass.getpass("Повторите пароль: ")
    if first != second:
        raise AuthError("Пароли не совпадают.")
    return first


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Пользователи детектора аномалий")
    parser.add_argument("--users-file", default=None, help="Путь к users.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add = subparsers.add_parser("add", help="Добавить пользователя")
    add.add_argument("username")
    add.add_argument("--role", choices=ROLES, default="analyst")
    add.add_argument("--password", default=None, help="Не рекомендуется: пароль в истории shell")

    passwd = subparsers.add_parser("passwd", help="Сменить пароль")
    passwd.add_argument("username")
    passwd.add_argument("--password", default=None)

    remove = subparsers.add_parser("remove", help="Удалить пользователя")
    remove.add_argument("username")
    remove.add_argument("--yes", action="store_true", help="Без подтверждения")

    subparsers.add_parser("list", help="Список пользователей")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        if args.command == "add":
            password = _read_password(args)
            path = add_user(args.username, password, role=args.role, path=args.users_file)
            log_event("cli", "admin", "user_add", target=args.username, target_role=args.role)
            print(f"Пользователь {args.username!r} добавлен ({args.role}). Файл: {path}")

        elif args.command == "passwd":
            password = _read_password(args)
            path = set_password(args.username, password, path=args.users_file)
            log_event("cli", "admin", "user_passwd", target=args.username)
            print(f"Пароль {args.username!r} обновлён. Файл: {path}")

        elif args.command == "remove":
            if not args.yes:
                answer = input(f"Удалить {args.username!r}? [y/N]: ").strip().lower()
                if answer not in ("y", "yes", "д", "да"):
                    print("Отменено.")
                    return 0
            path = remove_user(args.username, path=args.users_file)
            log_event("cli", "admin", "user_remove", target=args.username)
            print(f"Пользователь {args.username!r} удалён. Файл: {path}")

        elif args.command == "list":
            users = load_users(args.users_file)
            print(f"Файл: {users_path() if not args.users_file else args.users_file}")
            if not users:
                print("Пользователей нет.")
                return 0
            print(f"{'Логин':20s} {'Роль':10s} Создан")
            for username, record in sorted(users.items()):
                created = record.get("created_at", "-")
                print(f"{username:20s} {record.get('role', '-'):10s} {created}")

    except AuthError as exc:
        print(f"Ошибка: {exc}")
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nПрервано.")
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
