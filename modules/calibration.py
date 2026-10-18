"""Калибровка порога аномалий по benign-трафику.

IsolationForest.predict использует offset_, привязанный к contamination,
поэтому доля ложных срабатываний на новых данных непредсказуема. Здесь
порог выбирается по квантилю score на отложенном benign-трафике: это даёт
явный целевой FPR и позволяет показывать percentile вместо «сырого» score.

score = decision_function: чем меньше, тем аномальнее.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from modules.model_loader import resolve_model_dir

CALIBRATION_FILE = "calibration.json"


class CalibrationError(RuntimeError):
    """Файл калибровки повреждён или несовместим."""


@dataclass
class Calibration:
    threshold: float
    target_fpr: float
    achieved_fpr: dict[str, float]
    quantiles: np.ndarray
    rows: int = 0
    source: str = ""

    def percentile(self, score: float) -> float:
        """Доля benign-трафика со score <= данного, %."""
        if score is None or not np.isfinite(score):
            return 0.0
        index = int(np.searchsorted(self.quantiles, score, side="left"))
        return float(np.clip(index / (len(self.quantiles) - 1) * 100.0, 0.0, 100.0))

    def anomaly_percentile(self, score: float) -> float:
        """Насколько запись аномальнее нормы, % (100 — аномальнее всей нормы)."""
        return 100.0 - self.percentile(score)

    def anomaly_percentiles(self, scores) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        index = np.searchsorted(self.quantiles, scores, side="left")
        cdf = index / (len(self.quantiles) - 1)
        return np.clip((1.0 - cdf) * 100.0, 0.0, 100.0)


def load_calibration(model_dir: str | None = None) -> Calibration | None:
    """Читает calibration.json рядом с моделью. None, если файла нет."""
    path = os.path.join(resolve_model_dir(model_dir), CALIBRATION_FILE)
    if not os.path.isfile(path):
        return None

    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)

        threshold = float(payload["threshold"])
        target_fpr = float(payload.get("target_fpr", 0.02))
        quantiles = np.asarray(payload["quantiles"], dtype=float)
        achieved = {
            str(key): float(value)
            for key, value in (payload.get("achieved_fpr") or {}).items()
        }
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(f"Не удалось прочитать {path}: {exc}") from exc

    if len(quantiles) < 2 or not np.isfinite(quantiles).all():
        raise CalibrationError(f"Некорректные квантили в {path}")
    if np.any(np.diff(quantiles) < 0):
        raise CalibrationError(f"Квантили в {path} не отсортированы")
    if not np.isfinite(threshold):
        raise CalibrationError(f"Некорректный порог в {path}")
    if not 0.0 < target_fpr < 1.0:
        raise CalibrationError(f"Некорректный target_fpr в {path}")

    return Calibration(
        threshold=threshold,
        target_fpr=target_fpr,
        achieved_fpr=achieved,
        quantiles=quantiles,
        rows=int(payload.get("rows", 0)),
        source=str(payload.get("source", "")),
    )
