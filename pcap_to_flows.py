"""pcap/pcapng -> признаки потоков в формате CICFlowMeter (как в датасетах CIC).

Пакеты читает tshark (идёт вместе с Wireshark), потоки и признаки считаются
здесь — Java-версия CICFlowMeter и pcap-библиотеки Python не нужны.

Правила потока как в CICFlowMeter:
- двунаправленный 5-tuple (TCP/UDP), направление задаёт первый пакет;
- поток закрывается через 120 с от начала (новый пакет начинает новый
  поток) и после пакета с FIN или RST;
- Active/Idle — периоды активности, разделённые паузами > 5 с;
- длины пакетов — полезная нагрузка (без заголовков), время — мкс;
- потоки из одного пакета и с нулевой длительностью отбрасываются (их нет
  в датасетах CIC; --keep-short оставляет).

Колонки совпадают с feature_cols модели (названия датасетов dhoogla), плюс
служебные Src/Dst IP/Port и Timestamp для просмотра (модель их игнорирует).

Где намеренно не копируются особенности CIC-IDS2017 (старый CICFlowMeter):
- длина заголовка UDP — 8 байт (в CIC-IDS2017 у UDP-потоков 20/32 на пакет);
- счётчики флагов — 0/1 (есть ли флаг в потоке), как диапазон в датасете,
  но не его баги (в CIC-IDS2017 SYN Flag Count совпадает с Fwd PSH Flags);
- Fwd Seg Size Min без отрицательных переполнений.
Поэтому базовая модель на таких признаках ведёт себя как на «чужой сети»;
для adapt_model.py важно другое — один и тот же конвертер и при обучении,
и при применении на этой сети.

Пример:
    python pcap_to_flows.py capture.pcapng --output flows.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FLOW_TIMEOUT_US = 120_000_000
ACTIVITY_TIMEOUT_US = 5_000_000
UDP_HEADER = 8
SWEEP_EVERY = 100_000  # пакетов между проверками истёкших потоков

TSHARK_FIELDS = [
    "frame.time_epoch",
    "ip.src",
    "ip.dst",
    "ipv6.src",
    "ipv6.dst",
    "ip.proto",
    "ipv6.nxt",
    "tcp.srcport",
    "tcp.dstport",
    "udp.srcport",
    "udp.dstport",
    "tcp.len",
    "tcp.hdr_len",
    "tcp.flags",
    "tcp.window_size_value",
    "udp.length",
]

FIN, SYN, RST, PSH, ACK, URG, ECE = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40
FLAG_FEATURES = {
    "FIN Flag Count": FIN,
    "SYN Flag Count": SYN,
    "RST Flag Count": RST,
    "PSH Flag Count": PSH,
    "ACK Flag Count": ACK,
    "URG Flag Count": URG,
    "ECE Flag Count": ECE,
}

META_COLUMNS = ["Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp"]


class ConversionError(RuntimeError):
    """tshark не найден или pcap не читается."""


def find_tshark(explicit: str | None = None) -> str:
    candidates = [explicit] if explicit else []
    candidates += [
        shutil.which("tshark"),
        r"C:\Program Files\Wireshark\tshark.exe",
        r"C:\Program Files (x86)\Wireshark\tshark.exe",
        "/usr/bin/tshark",
        "/Applications/Wireshark.app/Contents/MacOS/tshark",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise ConversionError(
        "tshark не найден. Установите Wireshark (tshark входит в него) или "
        "передайте путь через --tshark."
    )


class PacketReader:
    """Пакеты TCP/UDP из pcap через tshark — построчно, без загрузки в память.

    Захват на сервере даёт гигабайты в час (2.2 млн пакетов за 24 минуты),
    поэтому весь вывод tshark в DataFrame не помещается в память. Итератор
    отдаёт словари с полями TSHARK_FIELDS (точки заменены на «_»).

    Обрезанный файл (скопирован во время записи) — не ошибка: прочитанные
    пакеты обрабатываются, текст tshark остаётся в .warning.
    """

    def __init__(self, pcap: str, tshark: str):
        self.pcap = pcap
        self.tshark = tshark
        self.count = 0
        self.warning = ""

    def __iter__(self):
        command = [self.tshark, "-r", self.pcap, "-n", "-Y", "tcp or udp", "-T", "fields",
                   "-E", "header=y", "-E", "separator=,", "-E", "quote=d",
                   "-E", "occurrence=f"]
        for name in TSHARK_FIELDS:
            command += ["-e", name]
        # stderr во временный файл: из PIPE его пришлось бы читать параллельно
        with tempfile.TemporaryFile() as errors:
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
            except OSError as exc:
                raise ConversionError(f"Не удалось запустить tshark: {exc}") from exc
            # csv требует поток с newline="" — у Popen такого параметра нет
            stdout = io.TextIOWrapper(process.stdout, encoding="utf-8", errors="replace", newline="")
            try:
                rows = csv.reader(stdout)
                header = [name.replace(".", "_") for name in next(rows, [])]
                for values in rows:
                    self.count += 1
                    yield dict(zip(header, values))
            finally:
                stdout.close()
                returncode = process.wait()
            errors.seek(0)
            message = errors.read().decode("utf-8", "replace").strip()
        if returncode != 0:
            if self.count == 0:
                raise ConversionError(f"tshark завершился с ошибкой: {message[:500]}")
            self.warning = message[:500]


def _int(value: str, default: int = 0) -> int:
    if not value:
        return default
    try:
        return int(value, 0) if value.startswith("0x") else int(float(value))
    except ValueError:
        return default


@dataclass
class Flow:
    src: str
    dst: str
    sport: int
    dport: int
    proto: int
    start: int
    times: list = field(default_factory=list)
    forward: list = field(default_factory=list)
    payloads: list = field(default_factory=list)
    headers: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    init_fwd_win: int = -1
    init_bwd_win: int = -1

    def add(self, time_us: int, is_forward: bool, payload: int, header: int, flags: int, window: int):
        if is_forward and self.init_fwd_win == -1 and self.proto == 6:
            self.init_fwd_win = window
        if not is_forward and self.init_bwd_win == -1 and self.proto == 6:
            self.init_bwd_win = window
        self.times.append(time_us)
        self.forward.append(is_forward)
        self.payloads.append(payload)
        self.headers.append(header)
        self.flags.append(flags)


def _stats(values: np.ndarray) -> tuple[float, float, float, float]:
    """mean, std (выборочная, как SummaryStatistics в CICFlowMeter), max, min."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0, 0.0
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return float(values.mean()), std, float(values.max()), float(values.min())


def _active_idle(times: np.ndarray) -> tuple[list, list]:
    active, idle = [], []
    start = end = times[0]
    for t in times[1:]:
        if t - end > ACTIVITY_TIMEOUT_US:
            if end - start > 0:
                active.append(end - start)
            idle.append(t - end)
            start = end = t
        else:
            end = t
    if end - start > 0:
        active.append(end - start)
    return active, idle


def flow_features(flow: Flow) -> dict:
    times = np.asarray(flow.times, dtype=np.int64)
    fwd = np.asarray(flow.forward, dtype=bool)
    payload = np.asarray(flow.payloads, dtype=float)
    header = np.asarray(flow.headers, dtype=float)
    flags = np.asarray(flow.flags, dtype=np.int64)

    duration = int(times[-1] - times[0])
    seconds = duration / 1e6
    n_fwd, n_bwd = int(fwd.sum()), int((~fwd).sum())
    fwd_bytes, bwd_bytes = float(payload[fwd].sum()), float(payload[~fwd].sum())

    fwd_mean, fwd_std, fwd_max, fwd_min = _stats(payload[fwd])
    bwd_mean, bwd_std, bwd_max, bwd_min = _stats(payload[~fwd])
    pkt_mean, pkt_std, pkt_max, pkt_min = _stats(payload)
    flow_iat = _stats(np.diff(times).astype(float))
    fwd_iat_values = np.diff(times[fwd]).astype(float)
    bwd_iat_values = np.diff(times[~fwd]).astype(float)
    fwd_iat = _stats(fwd_iat_values)
    bwd_iat = _stats(bwd_iat_values)
    active, idle = _active_idle(times)
    active_stats = _stats(np.asarray(active, dtype=float))
    idle_stats = _stats(np.asarray(idle, dtype=float))

    def rate(count: float) -> float:
        # Как в CICFlowMeter: при нулевой длительности — inf; pipeline
        # заменяет inf средними нормы
        return count / seconds if seconds > 0 else float("inf")

    total_packets = n_fwd + n_bwd
    row = {
        "Src IP": flow.src,
        "Src Port": flow.sport,
        "Dst IP": flow.dst,
        "Dst Port": flow.dport,
        "Timestamp": pd.Timestamp(int(times[0]), unit="us"),
        "Protocol": flow.proto,
        "Flow Duration": duration,
        "Total Fwd Packets": n_fwd,
        "Total Backward Packets": n_bwd,
        "Fwd Packets Length Total": fwd_bytes,
        "Bwd Packets Length Total": bwd_bytes,
        "Fwd Packet Length Max": fwd_max,
        "Fwd Packet Length Min": fwd_min,
        "Fwd Packet Length Mean": fwd_mean,
        "Fwd Packet Length Std": fwd_std,
        "Bwd Packet Length Max": bwd_max,
        "Bwd Packet Length Min": bwd_min,
        "Bwd Packet Length Mean": bwd_mean,
        "Bwd Packet Length Std": bwd_std,
        "Flow Bytes/s": rate(fwd_bytes + bwd_bytes),
        "Flow Packets/s": rate(total_packets),
        "Flow IAT Mean": flow_iat[0],
        "Flow IAT Std": flow_iat[1],
        "Flow IAT Max": flow_iat[2],
        "Flow IAT Min": flow_iat[3],
        "Fwd IAT Total": float(fwd_iat_values.sum()),
        "Fwd IAT Mean": fwd_iat[0],
        "Fwd IAT Std": fwd_iat[1],
        "Fwd IAT Max": fwd_iat[2],
        "Fwd IAT Min": fwd_iat[3],
        "Bwd IAT Total": float(bwd_iat_values.sum()),
        "Bwd IAT Mean": bwd_iat[0],
        "Bwd IAT Std": bwd_iat[1],
        "Bwd IAT Max": bwd_iat[2],
        "Bwd IAT Min": bwd_iat[3],
        "Fwd PSH Flags": int(bool((flags[fwd] & PSH).any())),
        "Fwd Header Length": float(header[fwd].sum()),
        "Bwd Header Length": float(header[~fwd].sum()),
        "Fwd Packets/s": rate(n_fwd),
        "Bwd Packets/s": rate(n_bwd),
        "Packet Length Min": pkt_min,
        "Packet Length Max": pkt_max,
        "Packet Length Mean": pkt_mean,
        "Packet Length Std": pkt_std,
        "Packet Length Variance": pkt_std ** 2,
        "Down/Up Ratio": n_bwd // n_fwd if n_fwd else 0,
        "Avg Packet Size": (fwd_bytes + bwd_bytes) / total_packets,
        "Avg Fwd Segment Size": fwd_mean,
        "Avg Bwd Segment Size": bwd_mean,
        # В датасетах CIC subflow всегда равен итогам по потоку
        "Subflow Fwd Packets": n_fwd,
        "Subflow Fwd Bytes": fwd_bytes,
        "Subflow Bwd Packets": n_bwd,
        "Subflow Bwd Bytes": bwd_bytes,
        "Init Fwd Win Bytes": flow.init_fwd_win,
        "Init Bwd Win Bytes": flow.init_bwd_win,
        "Fwd Act Data Packets": int((payload[fwd] > 0).sum()),
        "Fwd Seg Size Min": float(header[fwd].min()) if n_fwd else 0.0,
        "Active Mean": active_stats[0],
        "Active Std": active_stats[1],
        "Active Max": active_stats[2],
        "Active Min": active_stats[3],
        "Idle Mean": idle_stats[0],
        "Idle Std": idle_stats[1],
        "Idle Max": idle_stats[2],
        "Idle Min": idle_stats[3],
    }
    for name, bit in FLAG_FEATURES.items():
        row[name] = int(bool((flags & bit).any()))
    return row


def build_flows(packets) -> pd.DataFrame:
    """Собирает пакеты в потоки по правилам CICFlowMeter.

    packets — DataFrame или итератор словарей (PacketReader) с полями tshark.
    """
    active: dict[tuple, Flow] = {}
    rows: list[dict] = []
    if isinstance(packets, pd.DataFrame):
        packets = packets.to_dict("records")

    def finish(flow: Flow) -> None:
        # Признаки сразу, а пакеты потока — в сборщик мусора
        rows.append(flow_features(flow))

    for number, record in enumerate(packets, start=1):
        src = record["ip_src"] or record["ipv6_src"]
        dst = record["ip_dst"] or record["ipv6_dst"]
        proto = _int(record["ip_proto"] or record["ipv6_nxt"])
        if not src or not dst or proto not in (6, 17):
            continue
        if proto == 6:
            sport, dport = _int(record["tcp_srcport"]), _int(record["tcp_dstport"])
            payload = _int(record["tcp_len"])
            header = _int(record["tcp_hdr_len"], 20)
            flags = _int(record["tcp_flags"])
            window = _int(record["tcp_window_size_value"], -1)
        else:
            sport, dport = _int(record["udp_srcport"]), _int(record["udp_dstport"])
            payload = max(_int(record["udp_length"]) - UDP_HEADER, 0)
            header, flags, window = UDP_HEADER, 0, -1
        time_us = int(round(float(record["frame_time_epoch"]) * 1e6))

        key = (proto,) + tuple(sorted([(src, sport), (dst, dport)]))
        flow = active.get(key)
        if flow is not None and time_us - flow.start > FLOW_TIMEOUT_US:
            finish(active.pop(key))
            flow = None
        if flow is None:
            flow = Flow(src, dst, sport, dport, proto, start=time_us)
            active[key] = flow
        is_forward = (src, sport) == (flow.src, flow.sport)
        flow.add(time_us, is_forward, payload, header, flags, window)
        if flags & (FIN | RST):
            finish(active.pop(key))

        # Как CICFlowMeter: периодически закрываем истёкшие потоки, иначе на
        # сервере с тысячами соединений они копятся в памяти до конца файла
        if number % SWEEP_EVERY == 0:
            expired = [k for k, f in active.items() if time_us - f.start > FLOW_TIMEOUT_US]
            for expired_key in expired:
                finish(active.pop(expired_key))

    for flow in active.values():
        finish(flow)
    if not rows:
        raise ConversionError("В захвате нет TCP/UDP-потоков.")
    frame = pd.DataFrame(rows).sort_values("Timestamp", kind="stable").reset_index(drop=True)
    return frame


def drop_short_flows(flows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Убирает потоки из одного пакета и с нулевой длительностью.

    В датасетах CIC (CIC-IDS2017, CSE-CIC-IDS2018 в версии dhoogla) таких
    потоков нет вовсе, поэтому модели их не видели. У них бесконечные
    скорости, и pipeline подставляет среднее нормы — разное у разных моделей,
    из-за чего одни и те же потоки то помечаются, то нет. На реальном
    захвате это были ~34% потоков: «хвосты» TCP после FIN/RST и одиночные UDP.
    """
    packets = flows["Total Fwd Packets"] + flows["Total Backward Packets"]
    keep = (packets >= 2) & (flows["Flow Duration"] > 0)
    return flows[keep].reset_index(drop=True), int((~keep).sum())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="pcap/pcapng -> признаки потоков CICFlowMeter")
    parser.add_argument("pcap", nargs="+", help="Файлы .pcap/.pcapng")
    parser.add_argument("--output", required=True, help="CSV или Parquet с потоками")
    parser.add_argument("--tshark", default=None, help="Путь к tshark")
    parser.add_argument(
        "--keep-short",
        action="store_true",
        help="Оставить потоки из одного пакета и с нулевой длительностью "
        "(в датасетах CIC их нет, модели на них не обучались)",
    )
    parser.add_argument(
        "--no-meta",
        action="store_true",
        help="Не сохранять Src/Dst IP/Port и Timestamp (если файл уйдёт третьим лицам)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        tshark = find_tshark(args.tshark)
        frames = []
        for path in args.pcap:
            if not os.path.isfile(path):
                raise ConversionError(f"Файл не найден: {path}")
            packets = PacketReader(path, tshark)
            flows = build_flows(packets)
            message = f"{os.path.basename(path)}: пакетов TCP/UDP {packets.count:,}, потоков {len(flows):,}"
            if packets.warning:
                print(f"ВНИМАНИЕ ({os.path.basename(path)}): {packets.warning}")
            if not args.keep_short:
                flows, dropped = drop_short_flows(flows)
                message += f", из них коротких (1 пакет / 0 мкс) отброшено {dropped:,}"
            print(message)
            frames.append(flows)
    except ConversionError as exc:
        print(f"Ошибка: {exc}")
        return 2

    flows = pd.concat(frames, ignore_index=True)
    if args.no_meta:
        flows = flows.drop(columns=META_COLUMNS)
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    if output.lower().endswith(".parquet"):
        flows.to_parquet(output, index=False)
    else:
        flows.to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Сохранено: {output} ({len(flows):,} потоков)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
