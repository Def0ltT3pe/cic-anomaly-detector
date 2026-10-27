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

from modules.integrity import IntegrityError, verify_if_listed
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
    # False — calibration.json не покрыт checksums.json (подмена не обнаружится)
    protected: bool = True

    def _cdf(self, scores) -> np.ndarray:
        """Эмпирическая CDF benign-score с линейной интерполяцией между квантилями.

        quantiles[k] — квантиль уровня k/(n-1), поэтому интерполяция даёт
        точное значение в узлах (score на пороге 2% -> 2%), а не ступеньку
        до следующего узла, как searchsorted. Нефинитный score считается
        аномальнее всей нормы (CDF = 0).
        """
        scores = np.asarray(scores, dtype=float)
        levels = np.linspace(0.0, 1.0, len(self.quantiles))
        cdf = np.interp(scores, self.quantiles, levels, left=0.0, right=1.0)
        return np.where(np.isfinite(scores), cdf, 0.0)

    def percentile(self, score: float) -> float:
        """Доля benign-трафика со score <= данного, %."""
        if score is None:
            return 0.0
        return float(self._cdf([score])[0] * 100.0)

    def anomaly_percentile(self, score: float) -> float:
        """Насколько запись аномальнее нормы, % (100 — аномальнее всей нормы)."""
        return 100.0 - self.percentile(score)

    def anomaly_percentiles(self, scores) -> np.ndarray:
        return (1.0 - self._cdf(scores)) * 100.0


def load_calibration(model_dir: str | None = None) -> Calibration | None:
    """Читает calibration.json рядом с моделью. None, если файла нет."""
    directory = resolve_model_dir(model_dir)
    path = os.path.join(directory, CALIBRATION_FILE)
    if not os.path.isfile(path):
        return None

    try:
        unprotected = verify_if_listed(directory, [CALIBRATION_FILE])
    except IntegrityError as exc:
        raise CalibrationError(str(exc)) from exc

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
        protected=not unprotected,
    )
