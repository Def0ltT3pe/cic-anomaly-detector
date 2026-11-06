"""Файловые мелочи, общие для модулей."""

from __future__ import annotations

import os
import time

REPLACE_ATTEMPTS = 6
REPLACE_DELAY = 0.05


def replace_file(source: str, target: str, attempts: int = REPLACE_ATTEMPTS, sleep=time.sleep) -> None:
    """os.replace с повторами: на Windows только что записанный файл ненадолго
    держит антивирус или индексатор, и замена падает с PermissionError
    («Отказано в доступе»), хотя через доли секунды проходит."""
    for attempt in range(1, attempts + 1):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            sleep(REPLACE_DELAY * attempt)
