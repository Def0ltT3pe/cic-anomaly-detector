"""Графики приложения на Altair: тревоги по времени, топ-списки, отклонения признаков.

Цвета — пара синий/оранжевый, проверенная валидатором палитры (различимы при
дальтонизме, контраст ≥ 3:1) отдельно для светлой и тёмной темы. Синий —
IsolationForest и «ниже нормы», оранжевый — классификатор атак и «выше нормы».
"""

from __future__ import annotations

import altair as alt
import pandas as pd

# ink — цвет подписей значений: текст не красится в цвет ряда
PALETTE = {
    "light": {"blue": "#2a78d6", "orange": "#eb6834", "rule": "#8a8a86", "ink": "#3d3d3a"},
    "dark": {"blue": "#3987e5", "orange": "#d95926", "rule": "#6b6b67", "ink": "#c7c7c2"},
}


def _colors(theme: str | None) -> dict:
    return PALETTE["light" if theme == "light" else "dark"]


def timeline_chart(data: pd.DataFrame, models: list[str], theme: str | None = None):
    """Линии тревог по времени с перекрестием и подсказкой при наведении."""
    colors = _colors(theme)
    color = alt.Color(
        "Модель:N",
        scale=alt.Scale(domain=models, range=[colors["blue"], colors["orange"]][: len(models)]),
        legend=alt.Legend(orient="top", title=None),
    )
    hover = alt.selection_point(nearest=True, on="pointerover", fields=["Время"], empty=False)
    base = alt.Chart(data).encode(
        x=alt.X("Время:T", title=None, axis=alt.Axis(grid=False, format="%H:%M")),
        y=alt.Y("Тревог:Q", title="Тревог", axis=alt.Axis(tickMinStep=1)),
        color=color,
    )
    lines = base.mark_line(strokeWidth=2, interpolate="monotone")
    points = base.mark_point(size=70, filled=True).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip("Время:T", format="%d.%m %H:%M"),
            alt.Tooltip("Модель:N"),
            alt.Tooltip("Тревог:Q"),
        ],
    ).add_params(hover)
    rule = alt.Chart(data).mark_rule(color=colors["rule"], strokeWidth=1).encode(
        x="Время:T"
    ).transform_filter(hover)
    return (lines + points + rule).properties(height=260)


def top_chart(data: pd.DataFrame, category: str, theme: str | None = None):
    """Горизонтальные столбцы «категория → число тревог», по убыванию."""
    colors = _colors(theme)
    base = alt.Chart(data).encode(
        y=alt.Y(f"{category}:N", sort="-x", title=None),
        x=alt.X("Тревог:Q", title="Тревог", axis=alt.Axis(tickMinStep=1)),
        tooltip=[alt.Tooltip(f"{category}:N"), alt.Tooltip("Тревог:Q")],
    )
    bars = base.mark_bar(color=colors["blue"], cornerRadiusEnd=4, height={"band": 0.7})
    labels = base.mark_text(align="left", dx=4, color=colors["ink"]).encode(text="Тревог:Q")
    return (bars + labels).properties(height=max(120, 28 * len(data)))


def deviation_chart(deviations: pd.DataFrame, theme: str | None = None):
    """Отклонения признаков от нормы: вправо — выше нормы, влево — ниже."""
    colors = _colors(theme)
    data = deviations.assign(
        Направление=["выше нормы" if z > 0 else "ниже нормы" for z in deviations["z"]],
        Подпись=[
            f"×{r:.1f}" if pd.notna(r) and r >= 1.5 else f"{z:+.1f}σ"
            for r, z in zip(deviations["ratio"], deviations["z"])
        ],
    )
    order = list(data.sort_values("z", key=abs, ascending=False)["feature"])
    base = alt.Chart(data).encode(
        y=alt.Y("feature:N", sort=order, title=None),
        x=alt.X("z:Q", title="Отклонение от нормы, стандартных отклонений"),
        tooltip=[
            alt.Tooltip("feature:N", title="Признак"),
            alt.Tooltip("value:Q", title="Значение", format=",.2f"),
            alt.Tooltip("mean:Q", title="Норма (среднее)", format=",.2f"),
            alt.Tooltip("z:Q", title="Отклонение, σ", format="+.1f"),
            alt.Tooltip("ratio:Q", title="Во сколько раз от нормы", format=".1f"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
        color=alt.Color(
            "Направление:N",
            scale=alt.Scale(domain=["выше нормы", "ниже нормы"],
                            range=[colors["orange"], colors["blue"]]),
            legend=alt.Legend(orient="top", title=None),
        )
    )
    # Подпись у конца столбца: справа от положительных, слева от отрицательных
    text = {"color": colors["ink"]}
    above = base.mark_text(align="left", dx=6, **text).encode(text="Подпись:N").transform_filter("datum.z >= 0")
    below = base.mark_text(align="right", dx=-6, **text).encode(text="Подпись:N").transform_filter("datum.z < 0")
    zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=colors["rule"]).encode(x="z:Q")
    return (bars + above + below + zero).properties(height=max(140, 30 * len(data)))
