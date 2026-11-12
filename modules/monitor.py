"""Мониторинг серверов: новые захваты -> тревоги -> инциденты -> уведомления.

Работает по папкам, куда pull_captures.ps1 складывает файлы collect_capture.sh
(`<local>/<сессия>/capture_*.pcapng`). За один прогон:

1. находит захваты, которых ещё не было (или у которых изменился размер);
2. считает потоки и тревоги классификатора атак (analyze);
3. копит тревожные потоки сервера в `<state>/<сервер>/flagged.parquet` за
   последние RETAIN_DAYS дней — инцидент «≥3 потоков за час» может
   складываться из двух соседних файлов;
4. собирает инциденты по всей накопленной истории и пишет их в
   `<state>/incidents.parquet` (его показывает приложение);
5. сообщает о каждом новом инциденте один раз: ключ «сервер, адрес, час»
   попадает в state.json только после успешной отправки, поэтому при сбое
   сети сообщение уйдёт в следующий прогон.

Модели, pcap и Telegram сюда передаются функциями (analyze, notify): так
логику можно проверить без tshark и сети.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from modules.alerts import META_COLUMNS, aggregate_alerts, parse_trusted
from modules.fsutil import replace_file
from modules.servers import NAME_RE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STATE_DIR = os.path.join(BASE_DIR, "monitor")
CAPTURE_PATTERN = "capture_*.pcapng"
STATE_FILE = "state.json"
INCIDENTS_FILE = "incidents.parquet"
FLAGGED_FILE = "flagged.parquet"
HOURLY_FILE = "hourly.parquet"
HOURLY_COLUMNS = ["file", "hour", "flows"]
RETAIN_DAYS = 14
# Больше новых инцидентов за прогон — одно сводное сообщение вместо потока
MAX_SEPARATE_MESSAGES = 10
FLAGGED_COLUMNS = [*META_COLUMNS, "proba"]
INCIDENT_COLUMNS = ["server", "peer", "window_start", "flows", "services", "max_proba",
                    "first_seen", "last_seen", "reason", "notified"]


def state_dir() -> str:
    path = os.environ.get("MONITOR_DIR") or DEFAULT_STATE_DIR
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    return os.path.abspath(path)


@dataclass(frozen=True)
class Server:
    name: str
    local: str
    model_dir: str | None = None
    trusted: tuple = ()


def load_servers(config_path: str, trusted_env: str = "") -> list[Server]:
    """Серверы из pull_captures.json; model и trusted_ips — необязательные поля."""
    with open(config_path, encoding="utf-8-sig") as handle:
        entries = json.load(handle)
    if isinstance(entries, dict):
        entries = [entries]
    common = parse_trusted(trusted_env)
    servers = []
    for entry in entries:
        name = str(entry.get("name") or "").strip()
        local = str(entry.get("local") or "").strip()
        if not name or not local:
            raise ValueError(f"{config_path}: у сервера нужны name и local")
        # Имя становится папкой истории (monitor/<имя>/): «..\x» вывело бы её наружу
        if not NAME_RE.match(name):
            raise ValueError(f"{config_path}: имя сервера {name!r} — только латиница, цифры, «-» и «_»")
        servers.append(Server(
            name=name,
            local=local,
            model_dir=(str(entry["model"]).strip() or None) if entry.get("model") else None,
            trusted=tuple(common + parse_trusted(entry.get("trusted_ips", ""))),
        ))
    return servers


def load_state(directory: str) -> dict:
    path = os.path.join(directory, STATE_FILE)
    try:
        with open(path, encoding="utf-8") as handle:
            state = json.load(handle)
    except FileNotFoundError:
        state = {}
    state.setdefault("processed", {})
    state.setdefault("local_ip", {})
    state.setdefault("notified", [])
    return state


def save_state(directory: str, state: dict) -> None:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, STATE_FILE)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
    replace_file(temporary, path)


def _capture_key(server: Server, path: str) -> str:
    return f"{server.name}/" + os.path.relpath(path, server.local).replace(os.sep, "/")


def pending_captures(server: Server, state: dict) -> list[str]:
    """Захваты, которых нет в state или у которых изменился размер; старые — первыми."""
    found = glob.glob(os.path.join(glob.escape(server.local), "**", CAPTURE_PATTERN), recursive=True)
    pending = []
    for path in found:
        size = os.path.getsize(path)
        if state["processed"].get(_capture_key(server, path)) != size:
            pending.append(path)
    return sorted(pending, key=os.path.getmtime)


def mark_backlog(server: Server, state: dict, days: float, now: float) -> int:
    """Первый запуск: файлы старше days считаются проверенными, чтобы не
    разбирать архив за недели и не слать по нему сотни сообщений."""
    marked = 0
    for path in pending_captures(server, state):
        if now - os.path.getmtime(path) > days * 86400:
            state["processed"][_capture_key(server, path)] = os.path.getsize(path)
            marked += 1
    return marked


def _flagged_path(directory: str, server: Server) -> str:
    return os.path.join(directory, server.name, FLAGGED_FILE)


def load_flagged(directory: str, server: Server) -> pd.DataFrame:
    path = _flagged_path(directory, server)
    if not os.path.isfile(path):
        return pd.DataFrame(columns=FLAGGED_COLUMNS)
    return pd.read_parquet(path)


def merge_flagged(history: pd.DataFrame, new: pd.DataFrame, retain_days: float = RETAIN_DAYS) -> pd.DataFrame:
    """История + новые тревожные потоки: без повторов (файл мог скачаться
    заново) и без потоков старше retain_days от самого свежего."""
    frames = [frame[FLAGGED_COLUMNS] for frame in (history, new) if len(frame)]
    if not frames:
        return pd.DataFrame(columns=FLAGGED_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    merged["Timestamp"] = pd.to_datetime(merged["Timestamp"])
    merged = (
        merged.sort_values("proba", ascending=False, kind="stable")
        .drop_duplicates(subset=list(META_COLUMNS))
        .sort_values("Timestamp", kind="stable")
        .reset_index(drop=True)
    )
    cutoff = merged["Timestamp"].max() - pd.Timedelta(days=retain_days)
    return merged[merged["Timestamp"] >= cutoff].reset_index(drop=True)


def hourly_counts(flows: pd.DataFrame) -> pd.DataFrame:
    """Все потоки файла по часам (час, потоков) — знаменатель для доли тревог."""
    stamps = pd.to_datetime(flows["Timestamp"], errors="coerce").dropna()
    counts = stamps.dt.floor("h").value_counts().sort_index()
    return pd.DataFrame({"hour": counts.index, "flows": counts.to_numpy(dtype="int64")})


def load_hourly(directory: str, name: str) -> pd.DataFrame:
    path = os.path.join(directory, name, HOURLY_FILE)
    if not os.path.isfile(path):
        return pd.DataFrame(columns=HOURLY_COLUMNS)
    return pd.read_parquet(path)


def merge_hourly(history: pd.DataFrame, new: dict, retain_days: float = RETAIN_DAYS) -> pd.DataFrame:
    """Счётчики по файлам: файл, скачанный заново, заменяет свои строки, а не
    добавляется второй раз."""
    frames = [history[~history["file"].isin(new)]] if len(history) else []
    frames += [counts.assign(file=key)[HOURLY_COLUMNS] for key, counts in new.items() if len(counts)]
    frames = [frame for frame in frames if len(frame)]
    if not frames:
        return pd.DataFrame(columns=HOURLY_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    merged["hour"] = pd.to_datetime(merged["hour"])
    cutoff = merged["hour"].max() - pd.Timedelta(days=retain_days)
    return merged[merged["hour"] >= cutoff].sort_values("hour", kind="stable").reset_index(drop=True)


def server_incidents(server: Server, flagged: pd.DataFrame, local_ip: str | None) -> pd.DataFrame:
    if not len(flagged) or not local_ip:
        return pd.DataFrame(columns=INCIDENT_COLUMNS)
    incidents = aggregate_alerts(
        flagged, [True] * len(flagged), flagged["proba"].to_numpy(),
        local_ip=local_ip, trusted=list(server.trusted),
    )
    incidents = incidents.drop(columns=["rows"])
    incidents.insert(0, "server", server.name)
    return incidents


def incident_key(server: str, peer: str, window_start) -> str:
    return f"{server}|{peer}|{pd.Timestamp(window_start):%Y-%m-%dT%H:%M}"


def format_incident(row) -> str:
    return (
        f"Инцидент на {row.server}: {row.peer}\n"
        f"Час: {pd.Timestamp(row.window_start):%Y-%m-%d %H:%M} UTC\n"
        f"Потоков: {row.flows}, порты сервера: {row.services}\n"
        f"Почему: {row.reason}, макс. вероятность атаки {row.max_proba:.2f}"
    )


def format_summary(rows: pd.DataFrame) -> str:
    lines = [f"Новых инцидентов: {len(rows)}"]
    for server, group in rows.groupby("server", sort=True):
        top = group.groupby("peer")["flows"].sum().sort_values(ascending=False).head(5)
        lines.append(f"\n{server}: {len(group)} инцидентов, {group['peer'].nunique()} адресов")
        lines += [f"  {peer}: {flows} потоков" for peer, flows in top.items()]
    lines.append("\nПодробности — во вкладке «Мониторинг» приложения.")
    return "\n".join(lines)


@dataclass
class RunSummary:
    files: int = 0
    flagged: int = 0
    incidents: int = 0
    new_incidents: int = 0
    notified: int = 0
    errors: list = field(default_factory=list)


def run(
    servers: list[Server],
    analyze,
    notify=None,
    directory: str | None = None,
    backfill_days: float = 1.0,
    now: float | None = None,
    log=print,
) -> RunSummary:
    """Один прогон по всем серверам.

    analyze(path, server) -> (тревожные потоки с колонками FLAGGED_COLUMNS,
    адрес сервера[, все потоки по часам — hourly_counts]); notify(text) отправляет сообщение или бросает исключение;
    None — не отправлять (новые инциденты останутся неотправленными).
    """
    directory = directory or state_dir()
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    state = load_state(directory)
    summary = RunSummary()
    all_incidents = []

    for server in servers:
        if not os.path.isdir(server.local):
            summary.errors.append(f"{server.name}: нет папки {server.local}")
            log(f"{server.name}: нет папки {server.local} — пропуск")
            continue
        # Первый запуск для сервера (в том числе добавленного в конфиг позже)
        first_run = not any(key.startswith(server.name + "/") for key in state["processed"])
        if first_run and backfill_days >= 0:
            skipped = mark_backlog(server, state, backfill_days, now)
            if skipped:
                log(f"{server.name}: первый запуск, архив старше {backfill_days:g} сут. "
                    f"не проверяется ({skipped} файлов)")
        flagged = load_flagged(directory, server)
        new_frames = []
        new_hourly = {}
        for path in pending_captures(server, state):
            key = _capture_key(server, path)
            try:
                rows, local_ip, *rest = analyze(path, server)
            except Exception as exc:  # битый файл не должен останавливать остальные
                summary.errors.append(f"{key}: {exc}")
                log(f"{key}: ошибка — {exc}")
                continue
            if local_ip:
                state["local_ip"][server.name] = local_ip
            new_frames.append(rows)
            if rest and rest[0] is not None:
                new_hourly[key] = rest[0]
            state["processed"][key] = os.path.getsize(path)
            summary.files += 1
            summary.flagged += len(rows)
            log(f"{key}: тревожных потоков {len(rows)}")
        if new_frames:
            flagged = merge_flagged(flagged, pd.concat(new_frames, ignore_index=True))
            os.makedirs(os.path.join(directory, server.name), exist_ok=True)
            flagged.to_parquet(_flagged_path(directory, server), index=False)
        if new_hourly:
            hourly = merge_hourly(load_hourly(directory, server.name), new_hourly)
            hourly.to_parquet(os.path.join(directory, server.name, HOURLY_FILE), index=False)
        all_incidents.append(server_incidents(server, flagged, state["local_ip"].get(server.name)))

    frames = [frame for frame in all_incidents if len(frame)]
    incidents = (pd.concat(frames, ignore_index=True) if frames
                 else pd.DataFrame(columns=INCIDENT_COLUMNS[:-1]))
    keys = [incident_key(r.server, r.peer, r.window_start) for r in incidents.itertuples()]
    notified = set(state["notified"])
    fresh = incidents[[key not in notified for key in keys]]
    summary.incidents = len(incidents)
    summary.new_incidents = len(fresh)

    if len(fresh) and notify is not None:
        fresh_keys = [incident_key(r.server, r.peer, r.window_start) for r in fresh.itertuples()]
        try:
            if len(fresh) > MAX_SEPARATE_MESSAGES:
                notify(format_summary(fresh))
                notified.update(fresh_keys)
                summary.notified = len(fresh)
            else:
                for key, row in zip(fresh_keys, fresh.itertuples()):
                    notify(format_incident(row))
                    notified.add(key)
                    summary.notified += 1
        except Exception as exc:
            summary.errors.append(f"уведомление: {exc}")
            log(f"Уведомление не отправлено: {exc}")

    # Ключи старше истории тревог уже не появятся снова
    cutoff = datetime.fromtimestamp(now, timezone.utc).replace(tzinfo=None) - timedelta(days=RETAIN_DAYS + 1)
    state["notified"] = sorted(k for k in notified if pd.Timestamp(k.rsplit("|", 1)[1]) >= cutoff)
    incidents = incidents.copy()
    incidents["notified"] = [key in notified for key in keys]
    os.makedirs(directory, exist_ok=True)
    incidents.to_parquet(os.path.join(directory, INCIDENTS_FILE), index=False)
    save_state(directory, state)
    return summary


def load_incidents(directory: str | None = None) -> tuple[pd.DataFrame, datetime] | None:
    """Инциденты последнего прогона и его время (UTC) для приложения;
    None — мониторинг не запускался."""
    path = os.path.join(directory or state_dir(), INCIDENTS_FILE)
    if not os.path.isfile(path):
        return None
    updated = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
    return pd.read_parquet(path), updated


def daily_counts(incidents: pd.DataFrame, days: int = RETAIN_DAYS, today=None) -> pd.DataFrame:
    """Инцидентов по дням за последние days дней, пустые дни — нулями."""
    today = pd.Timestamp(today or datetime.now(timezone.utc).replace(tzinfo=None)).normalize()
    index = pd.date_range(today - pd.Timedelta(days=days - 1), today, freq="D")
    counts = pd.to_datetime(incidents["window_start"]).dt.normalize().value_counts()
    return pd.DataFrame({"День": index, "Инцидентов": counts.reindex(index, fill_value=0).to_numpy()})


# --- статистика сервера для приложения -----------------------------------------

def server_traffic(directory: str, name: str) -> pd.DataFrame:
    """По часам: всего потоков и тревожных (из истории monitor.py)."""
    hourly = load_hourly(directory, name)
    # Пустые ряды — тоже по времени: у сервера, который monitor.py ещё не проверял,
    # иначе колонка «Час» выходит целочисленной и сравнение со временем падает
    empty = pd.Series(dtype="int64", index=pd.DatetimeIndex([]))
    totals = (hourly.groupby(pd.to_datetime(hourly["hour"]))["flows"].sum()
              if len(hourly) else empty)
    path = os.path.join(directory, name, FLAGGED_FILE)
    # Нужно только время — остальные колонки истории не читаются
    flagged = (pd.read_parquet(path, columns=["Timestamp"]) if os.path.isfile(path)
               else pd.DataFrame(columns=FLAGGED_COLUMNS))
    alerts = (pd.to_datetime(flagged["Timestamp"]).dt.floor("h").value_counts()
              if len(flagged) else empty)
    index = totals.index.union(alerts.index).sort_values()
    return pd.DataFrame({
        "Час": index,
        "Потоков": totals.reindex(index, fill_value=0).to_numpy(dtype="int64"),
        "Тревог": alerts.reindex(index, fill_value=0).to_numpy(dtype="int64"),
    })


def traffic_summary(traffic: pd.DataFrame, since) -> dict:
    """Потоков, тревог и доля тревог с момента since."""
    recent = traffic[traffic["Час"] >= pd.Timestamp(since)]
    flows, alerts = int(recent["Потоков"].sum()), int(recent["Тревог"].sum())
    return {"flows": flows, "alerts": alerts, "share": alerts / flows if flows else None}


def top_attackers(incidents: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Адреса с инцидентами: сколько инцидентов и потоков, порты, когда в последний раз."""
    if not len(incidents):
        return pd.DataFrame(columns=["Адрес", "Инцидентов", "Потоков", "Порты", "Последний раз"])
    grouped = incidents.groupby("peer").agg(
        Инцидентов=("window_start", "count"),
        Потоков=("flows", "sum"),
        Порты=("services", lambda values: ", ".join(sorted(set(", ".join(values).split(", "))))),
        Последний=("last_seen", "max"),
    )
    grouped = grouped.sort_values(["Инцидентов", "Потоков"], ascending=False).head(n)
    return grouped.reset_index().rename(columns={"peer": "Адрес", "Последний": "Последний раз"})


def attacked_ports(directory: str, name: str, local_ip: str | None, n: int = 10) -> pd.DataFrame:
    """Порты сервера, на которые приходится больше всего тревог."""
    path = os.path.join(directory, name, FLAGGED_FILE)
    if not local_ip or not os.path.isfile(path):
        return pd.DataFrame(columns=["Порт", "Тревог"])
    flagged = pd.read_parquet(path, columns=["Src IP", "Src Port", "Dst Port"])
    outbound = flagged["Src IP"].astype(str) == local_ip
    ports = flagged["Src Port"].where(outbound, flagged["Dst Port"]).astype(int)
    counts = ports.value_counts().head(n)
    return pd.DataFrame({"Порт": counts.index.astype(str), "Тревог": counts.to_numpy()})


def last_download(local: str) -> datetime | None:
    """Время самого свежего скачанного захвата (UTC) или None."""
    found = glob.glob(os.path.join(glob.escape(local), "**", CAPTURE_PATTERN), recursive=True)
    if not found:
        return None
    return datetime.fromtimestamp(max(os.path.getmtime(path) for path in found), timezone.utc)
