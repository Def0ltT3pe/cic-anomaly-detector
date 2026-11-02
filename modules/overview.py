"""Данные для обзорной панели и компактных таблиц приложения.

Только расчёты на pandas — без Streamlit, чтобы их можно было проверить
тестами: тревоги по времени, топ адресов и портов, компактная таблица
потоков и вердикт по решениям моделей.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.alerts import detect_local_ip, has_meta

PROTOCOLS = {6: "TCP", 17: "UDP", 1: "ICMP"}
# Шаг графика по длительности захвата: не больше ~60–120 точек на линию
BUCKETS = (("1min", 2), ("10min", 12), ("1h", 72), ("6h", 24 * 15))

# Уровни вердикта: значок и цвет плашки берёт интерфейс
ATTACK, SUSPICIOUS, ANOMALY, NORMAL = "attack", "suspicious", "anomaly", "normal"


def _peers(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Внешний адрес и порт сервера для каждого потока (сервер — самый частый адрес)."""
    local_ip = detect_local_ip(df)
    outbound = (df["Src IP"].astype(str) == local_ip).to_numpy()
    peer = np.where(outbound, df["Dst IP"].astype(str), df["Src IP"].astype(str))
    port = np.where(outbound, df["Src Port"], df["Dst Port"])
    return peer, pd.to_numeric(pd.Series(port), errors="coerce").to_numpy()


def bucket_for(span: pd.Timedelta) -> str:
    hours = span.total_seconds() / 3600
    for bucket, limit_hours in BUCKETS:
        if hours <= limit_hours:
            return bucket
    return "1D"


def alert_timeline(df: pd.DataFrame, masks: dict[str, np.ndarray]) -> pd.DataFrame | None:
    """Число тревог каждой модели по интервалам времени; None — если времени в данных нет.

    Пустые интервалы заполняются нулями, чтобы линия не перескакивала через них.
    """
    if "Timestamp" not in df.columns:
        return None
    stamps = pd.to_datetime(df["Timestamp"], errors="coerce")
    valid = stamps.notna().to_numpy()
    if not valid.any():
        return None
    bucket = bucket_for(stamps[valid].max() - stamps[valid].min())
    floored = stamps.dt.floor(bucket)
    index = pd.date_range(floored[valid].min(), floored[valid].max(), freq=bucket)
    parts = []
    for name, mask in masks.items():
        counts = floored[valid & np.asarray(mask, dtype=bool)].value_counts()
        parts.append(pd.DataFrame({
            "Время": index,
            "Модель": name,
            "Тревог": counts.reindex(index, fill_value=0).to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True)


def top_peers(df: pd.DataFrame, mask: np.ndarray, n: int = 10) -> pd.DataFrame | None:
    """Внешние адреса с наибольшим числом тревог."""
    if not has_meta(df):
        return None
    peer, _ = _peers(df)
    counts = pd.Series(peer[np.asarray(mask, dtype=bool)]).value_counts().head(n)
    return pd.DataFrame({"Адрес": counts.index, "Тревог": counts.to_numpy()})


def top_ports(df: pd.DataFrame, mask: np.ndarray, n: int = 10) -> pd.DataFrame | None:
    """Порты сервера, на которые приходится больше всего тревог."""
    if not has_meta(df):
        return None
    _, port = _peers(df)
    flagged = port[np.asarray(mask, dtype=bool)]
    counts = pd.Series(flagged[np.isfinite(flagged)].astype(int)).value_counts().head(n)
    return pd.DataFrame({"Порт": counts.index.astype(str), "Тревог": counts.to_numpy()})


def incident_peers(incidents: pd.DataFrame | None) -> int:
    return 0 if incidents is None or not len(incidents) else int(incidents["peer"].nunique())


def flow_table(
    result, rows: np.ndarray, brute_probs: np.ndarray | None = None
) -> pd.DataFrame:
    """Компактная таблица потоков: номер строки, время, кто → куда, вероятность, score IF."""
    df = result.df.iloc[rows]
    table = pd.DataFrame({"#": np.asarray(rows, dtype=int)})
    if has_meta(result.df):
        stamps = pd.to_datetime(df["Timestamp"], errors="coerce")
        table["Время"] = stamps.dt.strftime("%d.%m %H:%M:%S").fillna("—").to_numpy()
        table["Источник"] = (df["Src IP"].astype(str) + ":" + df["Src Port"].astype(str)).to_numpy()
        table["Назначение"] = (df["Dst IP"].astype(str) + ":" + df["Dst Port"].astype(str)).to_numpy()
    if "Protocol" in df.columns:
        protocol = pd.to_numeric(df["Protocol"], errors="coerce")
        table["Протокол"] = [PROTOCOLS.get(int(p), str(int(p))) if np.isfinite(p) else "—"
                             for p in protocol]
    if brute_probs is not None:
        table["Вероятность атаки"] = np.asarray(brute_probs)[rows]
    table["IF score"] = np.round(result.scores[rows], 4)
    table["IF"] = np.where(result.mask[rows], "аномалия", "")
    if "Label" in df.columns:
        table["Метка"] = df["Label"].astype(str).to_numpy()
    return table


def model_verdict(if_flagged: bool, rf_flagged: bool | None, in_incident: bool) -> tuple[str, str]:
    """Вердикт по решениям моделей, без ИИ: (уровень, подпись)."""
    if rf_flagged and in_incident:
        return ATTACK, "Атака: классификатор пометил поток, и он входит в инцидент"
    if rf_flagged:
        return SUSPICIOUS, "Подозрительно: классификатор пометил поток, но инцидента по адресу нет"
    if if_flagged:
        return ANOMALY, "Аномалия IsolationForest: необычный поток, без признаков известной атаки"
    return NORMAL, "Норма: ни одна модель не пометила поток"


def llm_verdict(text: str) -> str | None:
    """Уровень вердикта из ответа ИИ (раздел «Вердикт»), None — если его нет."""
    lowered = text.lower()
    start = lowered.find("вердикт")
    if start < 0:
        return None
    head = lowered[start:start + 200]
    # «ложное» проверяется первым: «вероятно ложное срабатывание атаки» — не атака
    if "ложн" in head:
        return NORMAL
    if "подозр" in head:
        return SUSPICIOUS
    if "атак" in head:
        return ATTACK
    return None
