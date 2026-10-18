"""Общая логика подготовки признаков и поиска аномалий.

Используется и Streamlit-приложением, и CLI (detector.py), чтобы правила
валидации не расходились между интерфейсами.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class FeatureValidationError(ValueError):
    """Входной файл не подходит для модели (понятное пользователю сообщение)."""


@dataclass
class DetectionResult:
    """Результат детекции. Все индексы — позиционные внутри df/X."""

    df: pd.DataFrame
    X: pd.DataFrame
    predictions: np.ndarray
    mask: np.ndarray
    indices: np.ndarray
    scores: np.ndarray
    info: dict = field(default_factory=dict)
    threshold: float | None = None
    percentiles: np.ndarray | None = None

    @property
    def total(self) -> int:
        return int(len(self.df))

    @property
    def n_anomalies(self) -> int:
        return int(self.mask.sum())

    @property
    def share(self) -> float:
        return self.n_anomalies / self.total if self.total else 0.0

    def top_anomalies(self, limit: int = 5) -> np.ndarray:
        """Индексы самых сильных аномалий: чем ниже score, тем аномальнее."""
        if self.n_anomalies == 0 or limit <= 0:
            return np.array([], dtype=int)
        candidates = self.indices
        order = np.argsort(self.scores[candidates], kind="stable")
        return candidates[order][:limit]


def _missing_columns_message(missing: list[str], expected: int) -> str:
    shown = ", ".join(missing[:10])
    if len(missing) > 10:
        shown += f" и ещё {len(missing) - 10}"
    return (
        f"В файле нет {len(missing)} из {expected} нужных колонок: {shown}. "
        "Загрузите файл с тем же набором признаков, что и при обучении."
    )


def prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    fill_values: pd.Series | None = None,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Проверяет данные и возвращает матрицу признаков + сведения о чистке."""
    if df is None or len(df) == 0:
        raise FeatureValidationError("Файл пуст: в нём нет ни одной строки данных.")

    if max_rows is not None and len(df) > max_rows:
        raise FeatureValidationError(
            f"Слишком много строк: {len(df):,}. Максимум — {max_rows:,}."
        )

    duplicates = df.columns[df.columns.duplicated()].unique().tolist()
    if duplicates:
        raise FeatureValidationError(
            "В файле повторяются колонки: " + ", ".join(map(str, duplicates))
        )

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise FeatureValidationError(_missing_columns_message(missing, len(feature_cols)))

    non_numeric = []
    for col in feature_cols:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            continue
        converted = pd.to_numeric(series, errors="coerce")
        if (converted.isna() & series.notna()).fillna(False).any():
            non_numeric.append(col)
    if non_numeric:
        shown = ", ".join(non_numeric[:10])
        if len(non_numeric) > 10:
            shown += f" и ещё {len(non_numeric) - 10}"
        raise FeatureValidationError(
            f"Колонки должны содержать числа, но найдены текстовые значения: {shown}."
        )

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").astype("float64")

    n_inf = int(np.isinf(X.to_numpy()).sum())
    if n_inf:
        X = X.replace([np.inf, -np.inf], np.nan)

    n_nan = int(X.isna().to_numpy().sum())
    filled = 0
    if n_nan:
        if fill_values is None:
            raise FeatureValidationError(
                f"В признаках {n_nan} пропусков (NaN/inf), а заполнять их нечем: "
                "не переданы средние значения нормы."
            )
        fill = pd.to_numeric(fill_values, errors="coerce").reindex(feature_cols)
        if fill.isna().any():
            raise FeatureValidationError(
                "Не удалось заполнить пропуски: нет средних значений для части признаков."
            )
        X = X.fillna(fill)
        filled = n_nan
        if X.isna().to_numpy().any():
            raise FeatureValidationError("После заполнения в данных остались пропуски.")

    info = {
        "rows": int(len(X)),
        "inf_replaced": n_inf,
        "nan_filled": filled,
        "text_columns": [],
    }
    return X, info


def detect_anomalies(
    df: pd.DataFrame,
    model,
    scaler,
    selector,
    feature_cols: list[str],
    fill_values: pd.Series | None = None,
    max_rows: int | None = None,
    calibration=None,
) -> DetectionResult:
    """Полный проход: валидация -> масштабирование -> селекция -> score.

    Если передана calibration, аномалии определяются её порогом (явный FPR),
    иначе используется встроенный predict модели (contamination).
    """
    X, info = prepare_features(df, feature_cols, fill_values=fill_values, max_rows=max_rows)

    try:
        X_scaled = scaler.transform(X)
        X_filtered = selector.transform(X_scaled) if selector is not None else X_scaled
        predictions = np.asarray(model.predict(X_filtered))
    except Exception as exc:  # sklearn бросает разные типы исключений
        raise FeatureValidationError(
            "Модель не смогла обработать данные "
            f"(проверьте типы и диапазоны признаков): {exc}"
        ) from exc

    try:
        scores = np.asarray(model.decision_function(X_filtered), dtype=float)
    except Exception:
        # decision_function есть не у всех моделей — не критично для детекции
        scores = np.zeros(len(X), dtype=float)

    percentiles = None
    if calibration is not None:
        mask = scores <= calibration.threshold
        percentiles = calibration.anomaly_percentiles(scores)
    else:
        mask = predictions == -1

    indices = np.flatnonzero(mask)

    return DetectionResult(
        df=df,
        X=X,
        predictions=predictions,
        mask=mask,
        indices=indices,
        scores=scores,
        info=info,
        threshold=calibration.threshold if calibration is not None else None,
        percentiles=percentiles,
    )
