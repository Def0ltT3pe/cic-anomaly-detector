"""Текстовое объяснение аномалии по z-оценкам относительно нормы."""

from __future__ import annotations

import numpy as np
import pandas as pd


def explain_anomaly(record, means, stds, top_n: int = 3) -> str:
    """Описывает top_n самых сильных отклонений записи от benign-нормы.

    Устойчиво к NaN/inf и нулевым std: такие признаки просто пропускаются,
    вместо падения с ZeroDivisionError/ValueError.
    """
    if top_n is None or top_n < 1:
        top_n = 1

    features = [col for col in record.index if col in means.index and col in stds.index]
    if not features:
        return "  - отклонения не рассчитаны: нет общих признаков со статистиками."

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
        return "  - отклонения не рассчитаны: статистики для признаков некорректны."

    z_scores = (values[valid] - mu[valid]) / sd[valid]
    top_features = z_scores.abs().nlargest(min(top_n, len(z_scores))).index

    lines = []
    for feature in top_features:
        value = float(values[feature])
        mean_value = float(mu[feature])
        std_value = float(sd[feature])
        z_value = float(z_scores[feature])
        direction = "выше" if z_value > 0 else "ниже"
        lines.append(
            f"  - {feature}: {value:.2f} (норма {mean_value:.2f} ± {std_value:.2f}, "
            f"отклонение {direction} на {abs(z_value):.1f} сигм)"
        )

    if not lines:
        return "  - отклонения не рассчитаны: значимых признаков не найдено."
    return "\n".join(lines)
