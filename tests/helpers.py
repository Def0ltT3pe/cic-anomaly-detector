"""Вспомогательные функции для тестов."""

import os
import shutil
import uuid


def make_temp_dir(prefix: str = "test-") -> str:
    """Создаёт временный каталог в проекте (не в системном temp)."""
    base = os.environ.get("TEST_TMP_DIR") or os.path.join(os.getcwd(), ".test-tmp")
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, prefix + uuid.uuid4().hex)
    os.makedirs(path, exist_ok=False)
    return path


def remove_temp_dir(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)
