"""Supervised-детектор BruteForce.

IsolationForest по per-flow признакам не отличает брутфорс от обычных
коротких сессий (recall 0). Отдельный классификатор, обученный на размеченных
потоках, решает эту задачу: на CICIDS Tuesday recall 0.998 при FPR 0.1%.

Модель специализированная: она не заменяет IsolationForest, а дополняет его.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from modules.integrity import IntegrityError, verify_checksums
from modules.model_loader import DEFAULT_MODEL_DIR, resolve_model_dir

MODEL_FILE = "bruteforce.pkl"
META_FILE = "bruteforce.json"


class BruteForceError(RuntimeError):
    """Модель BruteForce не загрузилась или данные не подходят."""


@dataclass
class BruteForceDetector:
    model: object
    threshold: float
    feature_cols: list[str]
    metadata: dict

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        missing = [col for col in self.feature_cols if col not in X.columns]
        if missing:
            raise BruteForceError(
                "Для BruteForce-модели не хватает колонок: " + ", ".join(missing[:10])
            )
        proba = self.model.predict_proba(X[self.feature_cols])
        return np.asarray(proba[:, 1], dtype=float)

    def mask(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X) >= self.threshold


def load_bruteforce(model_dir: str | None = None) -> BruteForceDetector | None:
    """Загружает bruteforce.pkl/json. None, если модели нет.

    Если в выбранном каталоге модели нет, используется models/ (например,
    при MODEL_DIR=models_v2).
    """
    directory = resolve_model_dir(model_dir)
    model_path = os.path.join(directory, MODEL_FILE)
    meta_path = os.path.join(directory, META_FILE)

    if not (os.path.isfile(model_path) and os.path.isfile(meta_path)):
        default_dir = DEFAULT_MODEL_DIR
        fallback_model = os.path.join(default_dir, MODEL_FILE)
        fallback_meta = os.path.join(default_dir, META_FILE)
        if (
            os.path.abspath(default_dir) != os.path.abspath(directory)
            and os.path.isfile(fallback_model)
            and os.path.isfile(fallback_meta)
        ):
            directory = default_dir
            model_path = fallback_model
            meta_path = fallback_meta
        else:
            return None

    try:
        verify_checksums(directory, [MODEL_FILE, META_FILE])
    except IntegrityError as exc:
        raise BruteForceError(str(exc)) from exc

    try:
        with open(meta_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        threshold = float(metadata["threshold"])
        feature_cols = list(metadata["feature_cols"])
        model = joblib.load(model_path)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise BruteForceError(f"Не удалось загрузить BruteForce-модель: {exc}") from exc

    if not 0.0 < threshold < 1.0:
        raise BruteForceError("Некорректный threshold в bruteforce.json")
    if not feature_cols:
        raise BruteForceError("Пустой feature_cols в bruteforce.json")

    return BruteForceDetector(
        model=model,
        threshold=threshold,
        feature_cols=feature_cols,
        metadata=metadata,
    )
