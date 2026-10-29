"""Агрегация тревог supervised-модели в инциденты по внешнему адресу.

Тревога на каждый поток бесполезна на практике: на сервере 200–350 тыс.
потоков в сутки, и даже FPR 0.07% — это ~150–200 ложных тревог в день.
Атака же почти всегда даёт несколько потоков с одного адреса (бот
перебирает пароли), а ложные срабатывания разбросаны по разным адресам по
одному-два. Поэтому подозрительные потоки группируются по внешнему адресу
соединения и окну времени, и инцидентом считается группа, где:
- есть поток на «немедленный» порт сервера (по умолчанию SSH, 22) — на
  реальном сервере ложных срабатываний на SSH почти нет, а медленные боты
  дают по одной попытке в час с адреса;
- или не меньше min_flows подозрительных потоков за окно.

На двух VPS (проверочные часы, модели их не видели) это убрало ~180–220
ложных тревог в сутки до 0–1 инцидента за несколько часов, сохранив 100%
атакующих адресов.

Нужны колонки Src IP, Dst IP, Src Port, Dst Port и Timestamp — их пишет
pcap_to_flows.py (без --no-meta); в датасетах CIC их нет.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

META_COLUMNS = ("Src IP", "Dst IP", "Src Port", "Dst Port", "Timestamp")
DEFAULT_WINDOW = "1h"
DEFAULT_MIN_FLOWS = 3
DEFAULT_IMMEDIATE_PORTS = (22,)


def has_meta(df: pd.DataFrame) -> bool:
    return all(column in df.columns for column in META_COLUMNS)


def detect_local_ip(df: pd.DataFrame) -> str:
    """Самый частый адрес в Src/Dst — сам сервер (или основной хост захвата)."""
    addresses = pd.concat([df["Src IP"], df["Dst IP"]]).astype(str)
    return str(addresses.value_counts().index[0])


def aggregate_alerts(
    df: pd.DataFrame,
    flagged: np.ndarray,
    proba: np.ndarray | None = None,
    window: str = DEFAULT_WINDOW,
    min_flows: int = DEFAULT_MIN_FLOWS,
    immediate_ports=DEFAULT_IMMEDIATE_PORTS,
    local_ip: str | None = None,
) -> pd.DataFrame | None:
    """Инциденты по внешнему адресу и окну. None, если в данных нет адресов.

    Колонки результата: peer, window_start, flows, services, max_proba,
    first_seen, last_seen, reason, rows (позиционные индексы потоков).
    """
    if not has_meta(df):
        return None
    flagged = np.asarray(flagged, dtype=bool)
    columns = ["peer", "window_start", "flows", "services", "max_proba",
               "first_seen", "last_seen", "reason", "rows"]
    if not flagged.any():
        return pd.DataFrame(columns=columns)

    local_ip = local_ip or detect_local_ip(df)
    src = df["Src IP"].astype(str).to_numpy()
    outbound = src == local_ip
    peer = np.where(outbound, df["Dst IP"].astype(str), src)
    # Порт сервиса на стороне сервера: для входящих — Dst Port, для исходящих — Src Port
    local_port = np.where(outbound, df["Src Port"], df["Dst Port"]).astype(int)
    stamps = pd.to_datetime(df["Timestamp"])
    if proba is None:
        proba = np.ones(len(df), dtype=float)

    frame = pd.DataFrame(
        {
            "peer": peer,
            "window_start": stamps.dt.floor(window),
            "stamp": stamps,
            "port": local_port,
            "proba": np.asarray(proba, dtype=float),
            "row": np.arange(len(df)),
        }
    )[flagged]

    immediate = set(int(port) for port in (immediate_ports or ()))
    incidents = []
    for (address, start), group in frame.groupby(["peer", "window_start"], sort=False):
        ports = sorted(set(int(p) for p in group["port"]))
        hit = sorted(immediate & set(ports))
        if hit:
            reason = "порт " + ", ".join(map(str, hit))
        elif len(group) >= min_flows:
            reason = f"≥{min_flows} потоков"
        else:
            continue
        incidents.append(
            {
                "peer": address,
                "window_start": start,
                "flows": int(len(group)),
                "services": ", ".join(map(str, ports[:5])) + (" …" if len(ports) > 5 else ""),
                "max_proba": float(group["proba"].max()),
                "first_seen": group["stamp"].min(),
                "last_seen": group["stamp"].max(),
                "reason": reason,
                "rows": group["row"].tolist(),
            }
        )
    if not incidents:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(incidents, columns=columns)
        .sort_values(["window_start", "flows"], ascending=[True, False], kind="stable")
        .reset_index(drop=True)
    )


def summarize(incidents: pd.DataFrame | None, flagged: np.ndarray) -> str:
    """Одна строка для отчёта: сколько потоков свелось к скольким инцидентам."""
    flagged_count = int(np.asarray(flagged, dtype=bool).sum())
    if incidents is None:
        return f"подозрительных потоков {flagged_count}; инциденты не собраны — в данных нет адресов"
    covered = sum(len(rows) for rows in incidents["rows"]) if len(incidents) else 0
    return (
        f"подозрительных потоков {flagged_count} -> инцидентов {len(incidents)} "
        f"(в них {covered} потоков, от {incidents['peer'].nunique() if len(incidents) else 0} адресов)"
    )
