"""Список моделей, из которых можно выбрать в приложении.

Модели ищутся в папках models* проекта и в корнях из переменной MODELS_ROOT
(несколько путей через os.pathsep), не глубже двух уровней: так находятся
и `serverA/model2`. Произвольный путь из интерфейса не принимается — только
подписанные каталоги (есть checksums.json), найденные здесь.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from modules.integrity import CHECKSUMS_FILE
from modules.model_loader import BASE_DIR, MODEL_FILES, resolve_model_dir

METADATA_FILE = "metadata.json"
MAX_DEPTH = 2


@dataclass(frozen=True)
class ModelEntry:
    name: str
    path: str
    metadata: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def label(self) -> str:
        version = self.metadata.get("version")
        return f"{self.name} ({version})" if version else self.name


def is_model_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, MODEL_FILES["model"])) and os.path.isfile(
        os.path.join(path, CHECKSUMS_FILE)
    )


def read_metadata(path: str) -> dict:
    try:
        with open(os.path.join(path, METADATA_FILE), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _walk(root: str, depth: int):
    if is_model_dir(root):
        yield root
        return
    if depth >= MAX_DEPTH:
        return
    try:
        children = sorted(entry.path for entry in os.scandir(root) if entry.is_dir())
    except OSError:
        return
    for child in children:
        yield from _walk(child, depth + 1)


def default_roots() -> list[tuple[str, list[str]]]:
    """(база для имени, каталоги для обхода)."""
    roots = [(BASE_DIR, sorted(
        os.path.join(BASE_DIR, name)
        for name in os.listdir(BASE_DIR)
        if name.startswith("models") and os.path.isdir(os.path.join(BASE_DIR, name))
    ))]
    for root in (os.environ.get("MODELS_ROOT") or "").split(os.pathsep):
        root = root.strip()
        if root and os.path.isdir(root):
            root = os.path.abspath(root)
            roots.append((os.path.dirname(root), [root]))
    return roots


def list_models(roots: list[tuple[str, list[str]]] | None = None) -> list[ModelEntry]:
    """Подписанные каталоги моделей; каталог по умолчанию (MODEL_DIR) — первым."""
    found: dict[str, ModelEntry] = {}
    for base, dirs in roots if roots is not None else default_roots():
        for top in dirs:
            for path in _walk(os.path.abspath(top), 0):
                key = os.path.normcase(path)
                if key not in found:
                    name = os.path.relpath(path, base).replace(os.sep, "/")
                    found[key] = ModelEntry(name, path, read_metadata(path))

    default = resolve_model_dir()
    entries = list(found.values())
    key = os.path.normcase(default)
    if key in found:
        entries.sort(key=lambda entry: os.path.normcase(entry.path) != key)
    elif os.path.isfile(os.path.join(default, MODEL_FILES["model"])):
        # MODEL_DIR вне корней: показываем, даже если он не подписан —
        # загрузчик сам решит, можно ли его открыть
        entries.insert(0, ModelEntry(os.path.basename(default), default, read_metadata(default)))
    return entries
