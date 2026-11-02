"""Текстовое объяснение аномалии по z-оценкам относительно нормы."""

from __future__ import annotations

import numpy as np
import pandas as pd


def feature_deviations(record, means, stds, top_n: int = 3) -> pd.DataFrame:
    """top_n признаков с наибольшим |z| относительно benign-нормы.

    Колонки: feature, value, mean, std, z, ratio (value/mean, NaN при mean ≤ 0).
    Устойчиво к NaN/inf и нулевым std: такие признаки просто пропускаются,
    вместо падения с ZeroDivisionError/ValueError.
    """
    columns = ["feature", "value", "mean", "std", "z", "ratio"]
    if top_n is None or top_n < 1:
        top_n = 1

    features = [col for col in record.index if col in means.index and col in stds.index]
    if not features:
        return pd.DataFrame(columns=columns)

    values = pd.to_numeric(record[features], errors="coerce").astype(float)
    mu = pd.to_numeric(means[features], errors="coerce").astype(float)
    sd = pd.to_numeric(stds[features], errors="coerce").astype(float)

    valid = (
        pd.Series(np.isfinite(values.to_numpy()), index=values.index)
        & pd.Series(np.isfinite(mu.to_numpy()), index=mu.index)
        & pd.Series(np.isfinite(sd.to_numpy()), index=sd.index)
        & (sd > 0)
    )
    if not bool(valid.any()):
        return pd.DataFrame(columns=columns)

    z_scores = (values[valid] - mu[valid]) / sd[valid]
    top = z_scores.abs().nlargest(min(top_n, len(z_scores))).index
    mean_top = mu[top]
    return pd.DataFrame({
        "feature": list(top),
        "value": values[top].to_numpy(),
        "mean": mean_top.to_numpy(),
        "std": sd[top].to_numpy(),
        "z": z_scores[top].to_numpy(),
        "ratio": (values[top] / mean_top.where(mean_top > 0)).to_numpy(),
    })


def explain_anomaly(record, means, stds, top_n: int = 3) -> str:
    """Описывает top_n самых сильных отклонений записи от benign-нормы."""
    features = [col for col in record.index if col in means.index and col in stds.index]
    if not features:
        return "  - отклонения не рассчитаны: нет общих признаков со статистиками."

    deviations = feature_deviations(record, means, stds, top_n)
    if deviations.empty:
        return "  - отклонения не рассчитаны: статистики для признаков некорректны."

    lines = []
    for row in deviations.itertuples(index=False):
        direction = "выше" if row.z > 0 else "ниже"
        lines.append(
            f"  - {row.feature}: {row.value:.2f} (норма {row.mean:.2f} ± {row.std:.2f}, "
            f"отклонение {direction} на {abs(row.z):.1f} сигм)"
        )
    return "\n".join(lines)
