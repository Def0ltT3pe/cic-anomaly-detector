"""Подробный отчёт по одному потоку: факты для человека и для LLM.

Отчёт собирается из того, что уже посчитали модели: мета потока, решения
IsolationForest и supervised-модели, признаки, сильнее всего отличающиеся от
нормы, и контекст адреса (сколько ещё тревог и инцидентов от него в файле).
Для LLM адреса маскируются: наружу уходят первые два октета внешнего адреса
и роль хоста, реальные IP видны только в приложении.
"""

from __future__ import annotations

import ipaddress

import numpy as np
import pandas as pd

from modules.alerts import detect_local_ip, has_meta
from modules.explainer import explain_anomaly

TOP_FEATURES = 10
PROTOCOLS = {6: "TCP", 17: "UDP", 1: "ICMP"}

SYSTEM_PROMPT = """Ты — аналитик SOC. Тебе дают факты об одном сетевом потоке \
(признаки CICFlowMeter) и решения двух моделей детектора: IsolationForest \
(ищет любую необычность, обучен без меток) и supervised-классификатора атак \
(обучен на размеченных потоках). Адреса замаскированы: «сервер» — наблюдаемый \
хост, «внешний a.b.x.x» — первые два октета внешнего адреса.

Отвечай по-русски, опираясь только на переданные факты; если чего-то в данных \
нет — так и скажи, не выдумывай. Не пересказывай таблицу целиком: объясняй, \
что значат числа (во сколько раз больше нормы, что это говорит о соединении).

На первый вопрос о потоке отвечай отчётом с разделами:
1. **Вердикт** — атака / подозрительно / вероятно ложное срабатывание, с уверенностью.
2. **Почему сработали модели** — отдельно для каждой модели, которая пометила поток (и почему другая — нет).
3. **Ключевые признаки** — 3–5 самых важных, понятным языком.
4. **На что похоже** — тип активности (брутфорс SSH, сканирование, DoS, обычный трафик…).
5. **Что сделать** — конкретные шаги проверки и реагирования.

На уточняющие вопросы отвечай прямо и коротко, без повторения отчёта."""

DEFAULT_QUESTION = "Почему модель отреагировала на этот поток и насколько это опасно?"


def mask_ip(address, local_ip: str | None = None) -> str:
    """Роль хоста и сеть без последних октетов: «сервер», «внешний 185.12.x.x»."""
    text = str(address)
    if local_ip is not None and text == str(local_ip):
        return "сервер"
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return "адрес скрыт"
    kind = "внутренний" if (ip.is_private or ip.is_loopback or ip.is_link_local) else "внешний"
    if ip.version == 4:
        first, second = text.split(".")[:2]
        return f"{kind} {first}.{second}.x.x"
    return f"{kind} {':'.join(ip.exploded.split(':')[:2])}::x"


def _number(row, column):
    value = pd.to_numeric(row.get(column), errors="coerce") if column in row.index else np.nan
    return float(value) if np.isfinite(value) else None


def _flow_section(row, show, local_ip) -> list[str]:
    lines = []
    if local_ip is not None:
        protocol = _number(row, "Protocol")
        name = PROTOCOLS.get(int(protocol), str(int(protocol))) if protocol is not None else "?"
        src = f"{show(row['Src IP'])}:{int(row['Src Port'])}"
        dst = f"{show(row['Dst IP'])}:{int(row['Dst Port'])}"
        direction = (" (исходящий с сервера)" if str(row["Src IP"]) == local_ip
                     else " (входящий на сервер)" if str(row["Dst IP"]) == local_ip else "")
        lines.append(f"- Соединение: {src} → {dst}, {name}{direction}")
        lines.append(f"- Начало: {row['Timestamp']}")
    duration = _number(row, "Flow Duration")
    if duration is not None:
        lines.append(f"- Длительность: {duration / 1e6:.3f} с")
    fwd, bwd = _number(row, "Total Fwd Packets"), _number(row, "Total Backward Packets")
    if fwd is not None and bwd is not None:
        lines.append(f"- Пакеты: {int(fwd)} от инициатора, {int(bwd)} в ответ")
    fwd_bytes = _number(row, "Fwd Packets Length Total")
    bwd_bytes = _number(row, "Bwd Packets Length Total")
    if fwd_bytes is not None and bwd_bytes is not None:
        lines.append(f"- Полезная нагрузка: {int(fwd_bytes)} Б от инициатора, {int(bwd_bytes)} Б в ответ")
    if "Label" in row.index:
        lines.append(f"- Метка в файле: {row['Label']}")
    return lines


def _context_section(df, index, show, local_ip, if_mask, brute_mask, incidents) -> list[str]:
    row = df.iloc[index]
    peer = str(row["Dst IP"]) if str(row["Src IP"]) == local_ip else str(row["Src IP"])
    same = ((df["Src IP"].astype(str) == peer) | (df["Dst IP"].astype(str) == peer)).to_numpy()
    lines = [f"- Адрес собеседника: {show(peer)}, потоков с ним в файле: {int(same.sum())}"]
    if if_mask is not None:
        lines.append(f"- Из них аномалий IsolationForest: {int((same & if_mask).sum())}")
    if brute_mask is not None:
        lines.append(f"- Из них помечено supervised-моделью: {int((same & brute_mask).sum())}")
        ports = df.loc[same & brute_mask, "Dst Port"].astype(int).value_counts().head(5)
        if len(ports):
            lines.append("- Порты назначения помеченных потоков: "
                         + ", ".join(f"{port} ({count})" for port, count in ports.items()))
    if incidents is not None and len(incidents):
        own = incidents[incidents["rows"].apply(lambda rows: index in rows)]
        for incident in own.itertuples():
            lines.append(
                f"- Входит в инцидент: {incident.window_start:%Y-%m-%d %H:%M}, "
                f"{incident.flows} потоков, порты {incident.services}, причина: {incident.reason}"
            )
        if not len(own):
            lines.append("- В инцидент не входит")
    return lines


def build_flow_report(
    result,
    index: int,
    means,
    stds,
    brute=None,
    brute_probs=None,
    brute_mask=None,
    incidents=None,
    mask_ips: bool = True,
) -> str:
    """Markdown-отчёт по потоку result.df.iloc[index]; mask_ips — для отправки в LLM."""
    df = result.df
    row = df.iloc[index]
    local_ip = detect_local_ip(df) if has_meta(df) else None

    def show(address):
        return mask_ip(address, local_ip) if mask_ips else str(address)

    lines = [f"### Поток #{index}", *_flow_section(row, show, local_ip), "", "### Решения моделей"]
    flagged = bool(result.mask[index])
    detail = f"score {result.scores[index]:.4f}"
    if result.threshold is not None:
        detail += f", порог ≤ {result.threshold:.4f}"
    if result.percentiles is not None:
        detail += f", аномальнее {result.percentiles[index]:.1f}% нормального трафика"
    lines.append(f"- IsolationForest: {'аномалия' if flagged else 'норма'} ({detail})")
    if brute is not None and brute_probs is not None:
        hit = bool(brute_probs[index] >= brute.threshold)
        lines.append(
            f"- {brute.display_name} (supervised): {'атака' if hit else 'не атака'} "
            f"(вероятность {brute_probs[index]:.3f}, порог {brute.threshold:.3f})"
        )
    lines += ["", f"### Признаки, сильнее всего отличающиеся от нормы (топ-{TOP_FEATURES})",
              explain_anomaly(result.X.iloc[index], means, stds, top_n=TOP_FEATURES)]
    if local_ip is not None:
        lines += ["", "### Контекст адреса",
                  *_context_section(df, index, show, local_ip, result.mask, brute_mask, incidents)]
    return "\n".join(lines)


def flagged_candidates(result, brute_probs=None, brute_mask=None, limit: int = 500) -> list[int]:
    """Потоки для разбора: сначала помеченные supervised (по вероятности), затем IF (по score)."""
    order: list[int] = []
    if brute_mask is not None and brute_probs is not None:
        flagged = np.flatnonzero(brute_mask)
        order += flagged[np.argsort(-brute_probs[flagged], kind="stable")].tolist()
    if result.n_anomalies:
        anomalies = result.indices
        order += anomalies[np.argsort(result.scores[anomalies], kind="stable")].tolist()
    seen: set[int] = set()
    unique = [int(i) for i in order if not (int(i) in seen or seen.add(int(i)))]
    return unique[:limit]


def first_message(report: str, question: str) -> dict:
    return {"role": "user", "content": f"Факты о потоке:\n\n{report}\n\nВопрос: {question}"}
