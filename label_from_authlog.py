"""Разметка SSH-потоков по журналу sshd.

На публичном сервере ботов, перебирающих пароли, обычно десятки в час — это
готовые размеченные атаки именно этой сети. В журнале sshd у каждой сессии
есть адрес и порт клиента, поэтому поток сопоставляется с сессией точно:
по (Src IP, Src Port) входящего соединения на SSH-порт и по времени.

Сессия (один PID sshd) получает тип:
- accepted — в ней был успешный вход (Accepted password/publickey);
- failed   — неудачный пароль, неизвестный пользователь, отказ PAM;
- probe    — подключились и ушли без попытки входа (сканеры, kex-ошибки).

Метки потоков: failed -> SSH-Bruteforce, probe -> SSH-Probe, accepted и весь
не-SSH трафик -> Benign. Входящий SSH без записи в журнале по умолчанию
выбрасывается (--unlogged drop): по трафику нельзя понять, атака ли это.

Понимает формат journalctl по умолчанию / syslog ("Oct 28 19:01:56 host
sshd[123]: ...", год и часовой пояс задаются --year/--tz) и ISO
(journalctl -o short-iso: "2026-10-28T19:01:56+0000 host sshd[123]: ...").

Пример:
    journalctl -u ssh -o short-iso --since ... --until ... > ssh_auth.log
    python label_from_authlog.py --flows flows.parquet --log ssh_auth.log \
        --output flows_labeled.parquet
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field

import pandas as pd

LABELS = {"failed": "SSH-Bruteforce", "probe": "SSH-Probe", "accepted": "Benign"}
UNLOGGED_LABEL = "SSH-Unlogged"

_SYSLOG = re.compile(
    r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+\S+\s+sshd(?:-session)?\[(?P<pid>\d+)\]:\s+(?P<msg>.*)$"
)
_ISO = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:?\d{2}|Z)?)\s+\S+\s+sshd(?:-session)?\[(?P<pid>\d+)\]:\s+(?P<msg>.*)$"
)
# Адрес — IPv4/IPv6 прямо перед «port N»: «from 1.2.3.4 port 5», «Connection
# closed by invalid user admin 1.2.3.4 port 5», «Disconnected from user root ...»
_ADDRESS = re.compile(
    r"(?<![\w.:])(?P<ip>(?:\d{1,3}\.){3}\d{1,3}|[0-9a-fA-F]{0,4}(?::[0-9a-fA-F]{0,4}){2,7})"
    r"\s+port\s+(?P<port>\d+)"
)
_FAILED = ("Failed password", "Invalid user", "authentication failure", "Failed keyboard-interactive",
           "maximum authentication attempts", "more authentication failures")
_ACCEPTED = ("Accepted ",)


class AuthLogError(RuntimeError):
    """Журнал не распознан или потоки без адресов."""


@dataclass
class Session:
    pid: str
    start: pd.Timestamp
    end: pd.Timestamp
    ip: str | None = None
    port: int | None = None
    failed: bool = False
    accepted: bool = False
    lines: int = 0

    @property
    def kind(self) -> str:
        if self.accepted:
            return "accepted"
        return "failed" if self.failed else "probe"


@dataclass
class ParsedLog:
    sessions: list[Session] = field(default_factory=list)
    lines: int = 0
    skipped: int = 0


def _timestamp(text: str, iso: bool, year: int, tz: str) -> pd.Timestamp:
    if iso:
        stamp = pd.Timestamp(text)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(tz)
    else:
        stamp = pd.to_datetime(f"{year} {text}", format="%Y %b %d %H:%M:%S").tz_localize(tz)
    # Timestamp потоков в pcap_to_flows — UTC без часового пояса
    return stamp.tz_convert("UTC").tz_localize(None)


def parse_authlog(lines, year: int, tz: str = "UTC") -> ParsedLog:
    """Группирует строки sshd по PID в сессии с адресом, портом и исходом."""
    parsed = ParsedLog()
    sessions: dict[str, Session] = {}
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        parsed.lines += 1
        match = _ISO.match(line)
        iso = match is not None
        if match is None:
            match = _SYSLOG.match(line)
        if match is None:
            parsed.skipped += 1
            continue
        stamp = _timestamp(match["ts"], iso, year, tz)
        pid, message = match["pid"], match["msg"]
        session = sessions.get(pid)
        if session is None:
            session = Session(pid=pid, start=stamp, end=stamp)
            sessions[pid] = session
        session.end = max(session.end, stamp)
        session.lines += 1
        address = _ADDRESS.search(message)
        if address and session.port is None:
            session.ip, session.port = address["ip"], int(address["port"])
        if any(token in message for token in _ACCEPTED):
            session.accepted = True
        elif any(token in message for token in _FAILED):
            session.failed = True
    parsed.sessions = [s for s in sessions.values() if s.ip and s.port]
    return parsed


def label_flows(
    flows: pd.DataFrame,
    sessions: list[Session],
    ssh_port: int = 22,
    tolerance_s: float = 300.0,
    unlogged: str = "drop",
) -> tuple[pd.DataFrame, dict]:
    """Ставит Label по сессиям журнала. Возвращает (потоки, сводку)."""
    required = {"Src IP", "Src Port", "Dst Port", "Timestamp"}
    if not required <= set(flows.columns):
        raise AuthLogError(
            "В потоках нет Src IP/Src Port/Dst Port/Timestamp — сконвертируйте pcap "
            "без --no-meta."
        )
    by_key: dict[tuple[str, int], list[Session]] = {}
    for session in sessions:
        by_key.setdefault((session.ip, session.port), []).append(session)

    tolerance = pd.Timedelta(seconds=tolerance_s)
    timestamps = pd.to_datetime(flows["Timestamp"])
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_convert("UTC").dt.tz_localize(None)
    inbound = (flows["Dst Port"] == ssh_port).to_numpy()

    labels = ["Benign"] * len(flows)
    matched_sessions: set[str] = set()
    for row, (ip, port, stamp, is_inbound) in enumerate(
        zip(flows["Src IP"], flows["Src Port"], timestamps, inbound)
    ):
        if not is_inbound:
            continue
        candidates = [
            s for s in by_key.get((str(ip), int(port)), [])
            if s.start - tolerance <= stamp <= s.end + tolerance
        ]
        if not candidates:
            labels[row] = UNLOGGED_LABEL
            continue
        session = min(candidates, key=lambda s: abs((s.start - stamp).total_seconds()))
        matched_sessions.add(session.pid)
        labels[row] = LABELS[session.kind]

    result = flows.copy()
    result["Label"] = labels
    unlogged_rows = int((result["Label"] == UNLOGGED_LABEL).sum())
    if unlogged == "drop":
        result = result[result["Label"] != UNLOGGED_LABEL].reset_index(drop=True)
    elif unlogged == "benign":
        result.loc[result["Label"] == UNLOGGED_LABEL, "Label"] = "Benign"

    in_window = [
        s for s in sessions
        if timestamps.min() - tolerance <= s.start <= timestamps.max() + tolerance
    ]
    summary = {
        "sessions": len(sessions),
        "sessions_in_capture_window": len(in_window),
        "sessions_matched": len(matched_sessions),
        "inbound_ssh_flows": int(inbound.sum()),
        "unlogged_flows": unlogged_rows,
        "labels": result["Label"].value_counts().to_dict(),
    }
    return result, summary


def _read(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Разметка SSH-потоков по журналу sshd")
    parser.add_argument("--flows", nargs="+", required=True, help="Потоки из pcap_to_flows.py")
    parser.add_argument("--log", nargs="+", required=True, help="Журнал sshd (journalctl / auth.log)")
    parser.add_argument("--output", required=True, help="CSV/Parquet; при нескольких --flows — каталог")
    parser.add_argument("--year", type=int, default=None,
                        help="Год для строк без года (syslog); по умолчанию — год потоков")
    parser.add_argument("--tz", default="UTC", help="Часовой пояс журнала без зоны (по умолчанию UTC)")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--tolerance", type=float, default=300.0,
                        help="Допуск по времени между сессией и потоком, с")
    parser.add_argument("--unlogged", choices=("drop", "benign", "keep"), default="drop",
                        help="Входящий SSH без записи в журнале: выбросить, считать нормой "
                        "или оставить с меткой SSH-Unlogged")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        frames = {path: _read(path) for path in args.flows}
        year = args.year
        if year is None:
            first = pd.to_datetime(next(iter(frames.values()))["Timestamp"]).min()
            year = int(first.year)
        lines = []
        for path in args.log:
            with open(path, encoding="utf-8", errors="replace") as handle:
                lines.extend(handle)
        parsed = parse_authlog(lines, year=year, tz=args.tz)
        if not parsed.sessions:
            raise AuthLogError(
                f"В журнале не найдено сессий sshd с адресом и портом "
                f"(строк {parsed.lines}, нераспознано {parsed.skipped})."
            )
        kinds = pd.Series([s.kind for s in parsed.sessions]).value_counts().to_dict()
        print(f"Журнал: строк {parsed.lines:,}, сессий {len(parsed.sessions):,} {kinds}")

        many = len(frames) > 1
        if many:
            os.makedirs(args.output, exist_ok=True)
        for path, flows in frames.items():
            labeled, summary = label_flows(
                flows, parsed.sessions, args.ssh_port, args.tolerance, args.unlogged
            )
            output = (
                os.path.join(args.output, os.path.splitext(os.path.basename(path))[0] + "_labeled.parquet")
                if many else args.output
            )
            if output.lower().endswith(".csv"):
                labeled.to_csv(output, index=False)
            else:
                labeled.to_parquet(output, index=False)
            print(
                f"{os.path.basename(path)}: входящих SSH {summary['inbound_ssh_flows']}, "
                f"без записи в журнале {summary['unlogged_flows']} ({args.unlogged}) -> "
                f"{summary['labels']}"
            )
            if summary["sessions_in_capture_window"] and not summary["sessions_matched"]:
                print(
                    "  ВНИМАНИЕ: ни одна сессия журнала не совпала с потоками. Проверьте, "
                    "что журнал с того же сервера и --tz (время журнала vs UTC потоков)."
                )
            print(f"  -> {os.path.abspath(output)}")
    except (AuthLogError, OSError, ValueError, KeyError) as exc:
        print(f"Ошибка: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
