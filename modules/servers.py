"""Серверы сбора: список в pull_captures.json и управление ими по SSH.

Добавление сервера из приложения:
1. первый вход по паролю (paramiko): ключ хоста записывается в known_hosts,
   ваш публичный ключ — в authorized_keys сервера; пароль нигде не хранится;
2. дальше всё — по ключу через системный ssh (как у pull_captures.ps1):
   копирование collect_capture.sh, расчёт места (plan), запуск, статус,
   остановка;
3. сервер попадает в pull_captures.json — его начнёт забирать задача
   Планировщика, а monitor.py — проверять.

Все значения из формы проверяются по белым спискам символов и попадают в
удалённые команды только через shlex.quote: имя, пользователь и хост не
могут стать опцией ssh или частью команды.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(BASE_DIR, "pull_captures.json")
COLLECTOR_SCRIPT = os.path.join(BASE_DIR, "collect_capture.sh")
REMOTE_SCRIPT = "/usr/local/sbin/collect_capture.sh"
DEFAULT_REMOTE_DIR = "/var/lib/cic-capture"
SSH_TIMEOUT = 15
COMMAND_TIMEOUT = 120
INSTALL_TIMEOUT = 600

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
USER_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")
HOST_RE = re.compile(r"^(?=.{1,253}$)[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$")
REMOTE_DIR_RE = re.compile(r"^/[A-Za-z0-9._/-]{1,200}$")
DURATION_RE = re.compile(r"^(0|[1-9][0-9]{0,4}[smhd])$")


class ServerError(RuntimeError):
    """Понятная пользователю ошибка: неверные данные, нет связи, команда не прошла."""


@dataclass(frozen=True)
class ServerEntry:
    name: str
    user: str
    host: str
    remote: str = DEFAULT_REMOTE_DIR
    local: str = ""

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"

    def as_config(self) -> dict:
        return {"name": self.name, "target": self.target, "remote": self.remote, "local": self.local}


def validate(name: str, user: str, host: str, remote: str, local: str) -> ServerEntry:
    """Значения формы -> ServerEntry; ServerError с понятной причиной."""
    name, user, host, remote, local = (str(v or "").strip() for v in (name, user, host, remote, local))
    if not NAME_RE.match(name):
        raise ServerError("Имя: латиница, цифры, «-» и «_», до 40 символов, с буквы или цифры.")
    if not USER_RE.match(user):
        raise ServerError("Пользователь: строчная латиница, цифры, «-» и «_» (например, root).")
    if not HOST_RE.match(host):
        raise ServerError("Адрес: IPv4 или имя хоста (латиница, цифры, «-», «.»), без порта.")
    if not REMOTE_DIR_RE.match(remote) or "/../" in remote + "/":
        raise ServerError("Папка на сервере: абсолютный путь без «..» и пробелов.")
    if not local or not os.path.isabs(local):
        raise ServerError("Папка на этом компьютере: полный путь, например D:\\Захваты\\serverC.")
    return ServerEntry(name, user, host, remote, os.path.normpath(local))


def config_path() -> str:
    return os.environ.get("PULL_CONFIG") or DEFAULT_CONFIG


def load_config(path: str | None = None) -> list[dict]:
    path = path or config_path()
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8-sig") as handle:
        entries = json.load(handle)
    return entries if isinstance(entries, list) else [entries]


def save_config(entries: list[dict], path: str | None = None) -> None:
    """Атомарно; UTF-8 без BOM — его читают и pull_captures.ps1, и monitor.py."""
    path = path or config_path()
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(entries, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def add_to_config(entry: ServerEntry, path: str | None = None) -> None:
    entries = load_config(path)
    if any(str(e.get("name", "")).lower() == entry.name.lower() for e in entries):
        raise ServerError(f"Сервер «{entry.name}» уже есть в списке.")
    if any(str(e.get("target", "")).split("@")[-1] == entry.host for e in entries):
        raise ServerError(f"Адрес {entry.host} уже есть в списке.")
    save_config(entries + [entry.as_config()], path)


def remove_from_config(name: str, path: str | None = None) -> None:
    entries = load_config(path)
    kept = [e for e in entries if e.get("name") != name]
    if len(kept) == len(entries):
        raise ServerError(f"Сервера «{name}» нет в списке.")
    save_config(kept, path)


def entry_from_config(item: dict) -> ServerEntry:
    user, _, host = str(item.get("target", "")).rpartition("@")
    return ServerEntry(str(item.get("name", "")), user, host,
                       str(item.get("remote") or DEFAULT_REMOTE_DIR), str(item.get("local", "")))


# --- ключ этого компьютера -------------------------------------------------

def ssh_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".ssh")


def public_key(create: bool = False) -> str | None:
    """Публичный ключ для authorized_keys; create — создать ed25519 без пароля
    (задача Планировщика ходит по ключу без участия человека)."""
    for name in ("id_ed25519.pub", "id_ecdsa.pub", "id_rsa.pub"):
        path = os.path.join(ssh_dir(), name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                return handle.read().strip()
    if not create:
        return None
    keygen = shutil.which("ssh-keygen")
    if keygen is None:
        raise ServerError("Нет ssh-keygen: установите клиент OpenSSH.")
    os.makedirs(ssh_dir(), exist_ok=True)
    private = os.path.join(ssh_dir(), "id_ed25519")
    subprocess.run([keygen, "-q", "-t", "ed25519", "-N", "", "-f", private],
                   check=True, capture_output=True, timeout=30)
    return public_key()


# --- первый вход по паролю -------------------------------------------------

def _remember_host_key(host: str, key) -> None:
    """Дописать ключ хоста в known_hosts (без перезаписи файла: в нём могут
    быть хэшированные записи, которые paramiko сохранил бы иначе)."""
    os.makedirs(ssh_dir(), exist_ok=True)
    with open(os.path.join(ssh_dir(), "known_hosts"), "a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{host} {key.get_name()} {key.get_base64()}\n")


def authorized_keys_command(key: str) -> str:
    quoted = shlex.quote(key)
    return (
        "umask 077; mkdir -p ~/.ssh && touch ~/.ssh/authorized_keys && "
        f"(grep -qxF {quoted} ~/.ssh/authorized_keys || printf '%s\\n' {quoted} >> ~/.ssh/authorized_keys)"
    )


def install_key_with_password(entry: ServerEntry, password: str, key: str) -> str:
    """Входит по паролю, кладёт key в authorized_keys; возвращает отпечаток ключа хоста.

    Ключ хоста при первом входе принимается (как accept-new у OpenSSH) и
    записывается в known_hosts; если хост уже известен и ключ другой —
    вход отклоняется.
    """
    try:
        import paramiko
    except ImportError as exc:
        raise ServerError("Нет пакета paramiko: pip install -r requirements.txt") from exc

    known_hosts = os.path.join(ssh_dir(), "known_hosts")
    client = paramiko.SSHClient()
    if os.path.isfile(known_hosts):
        client.load_host_keys(known_hosts)
    new_keys = []

    class AcceptNew(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self, client_, hostname, key_):
            new_keys.append(key_)

    client.set_missing_host_key_policy(AcceptNew())
    try:
        client.connect(entry.host, username=entry.user, password=password, timeout=SSH_TIMEOUT,
                       allow_agent=False, look_for_keys=False)
        _, stdout, stderr = client.exec_command(authorized_keys_command(key), timeout=SSH_TIMEOUT)
        status = stdout.channel.recv_exit_status()
        if status != 0:
            raise ServerError(f"Ключ не записан: {stderr.read().decode('utf-8', 'replace')[:300]}")
        host_key = client.get_transport().get_remote_server_key()
    except paramiko.AuthenticationException:
        raise ServerError("Неверный пользователь или пароль (или вход по паролю на сервере выключен).") from None
    except paramiko.BadHostKeyException:
        raise ServerError(
            f"Ключ хоста {entry.host} не совпадает с known_hosts — возможна подмена сервера. "
            "Если сервер переустановлен, удалите его строку: ssh-keygen -R " + entry.host
        ) from None
    except (paramiko.SSHException, OSError) as exc:
        raise ServerError(f"Не удалось подключиться к {entry.host}: {exc}") from None
    finally:
        client.close()
    if new_keys:
        _remember_host_key(entry.host, host_key)
    return host_key.get_name() + " " + fingerprint(host_key.asbytes())


def fingerprint(blob: bytes) -> str:
    import base64
    import hashlib

    return "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")


# --- дальше по ключу через системный ssh ------------------------------------

def _ssh_binary() -> str:
    ssh = shutil.which("ssh")
    if ssh is None:
        raise ServerError("Нет ssh: установите клиент OpenSSH (Параметры → Приложения → Дополнительные компоненты).")
    return ssh


def ssh_run(entry: ServerEntry, command: str, stdin: bytes | None = None,
            timeout: int = COMMAND_TIMEOUT, runner=subprocess.run) -> str:
    """Команда на сервере по ключу; вывод (stdout+stderr) или ServerError."""
    args = [_ssh_binary(), "-o", "BatchMode=yes", "-o", f"ConnectTimeout={SSH_TIMEOUT}",
            "-o", "StrictHostKeyChecking=accept-new", "--", entry.target, command]
    try:
        result = runner(args, input=stdin, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ServerError(f"{entry.host}: команда не уложилась в {timeout} с.") from None
    output = (result.stdout or b"").decode("utf-8", "replace") + (result.stderr or b"").decode("utf-8", "replace")
    if result.returncode != 0:
        if result.returncode == 255:  # ошибка самого ssh: сеть, ключ хоста, вход по ключу
            raise ServerError(f"Не удалось войти на {entry.target} по SSH: {output.strip()[:300]}")
        raise ServerError(output.strip()[:2000] or f"код {result.returncode}")
    return output


def _sudo(entry: ServerEntry) -> str:
    return "" if entry.user == "root" else "sudo -n "


def collector_command(entry: ServerEntry, action: str, *options: str) -> str:
    """Команда collect_capture.sh на сервере; options уже проверены."""
    parts = [REMOTE_SCRIPT, action, "--dir", entry.remote, *options]
    return _sudo(entry) + " ".join(shlex.quote(part) for part in parts)


def start_options(duration: str, reserve_gb: int, file_mb: int) -> list[str]:
    if not DURATION_RE.match(duration.strip()):
        raise ServerError("Длительность: число и единица — 30m, 12h, 7d; 0 — до остановки.")
    if not 0 <= int(reserve_gb) <= 1000 or not 10 <= int(file_mb) <= 2000:
        raise ServerError("Запас места 0–1000 ГБ, файл кольца 10–2000 МБ.")
    return ["--duration", duration.strip(), "--reserve-gb", str(int(reserve_gb)),
            "--file-mb", str(int(file_mb))]


def check_key_login(entry: ServerEntry, runner=subprocess.run) -> str:
    """Вход по ключу и права: 'root', 'sudo' или ServerError."""
    output = ssh_run(entry, 'id -u; command -v sudo >/dev/null && sudo -n true 2>/dev/null && echo sudo-ok; true',
                     timeout=SSH_TIMEOUT * 2, runner=runner)
    lines = output.split()
    if lines and lines[0] == "0":
        return "root"
    if "sudo-ok" in lines:
        return "sudo"
    raise ServerError(f"{entry.user} — не root и без sudo без пароля: сбору нужны права root.")


def install_collector(entry: ServerEntry, runner=subprocess.run) -> None:
    with open(COLLECTOR_SCRIPT, "rb") as handle:
        script = handle.read().replace(b"\r\n", b"\n")
    path = shlex.quote(REMOTE_SCRIPT)
    ssh_run(entry, f"{_sudo(entry)}sh -c {shlex.quote(f'cat > {path} && chmod 755 {path}')}",
            stdin=script, runner=runner)


def has_dumpcap(entry: ServerEntry, runner=subprocess.run) -> bool:
    return "yes" in ssh_run(entry, "command -v dumpcap >/dev/null && echo yes || echo no", runner=runner)


def install_dumpcap(entry: ServerEntry, runner=subprocess.run) -> str:
    """dumpcap из пакета tshark (Debian/Ubuntu); на других системах — ServerError."""
    command = (
        "command -v apt-get >/dev/null || { echo 'нет apt-get: установите tshark вручную'; exit 3; }; "
        f"{_sudo(entry)}env DEBIAN_FRONTEND=noninteractive apt-get install -y -q tshark"
    )
    return ssh_run(entry, command, timeout=INSTALL_TIMEOUT, runner=runner)


def collector(entry: ServerEntry, action: str, *options: str, runner=subprocess.run) -> str:
    if action not in ("plan", "start", "status", "stop"):
        raise ServerError(f"Неизвестное действие: {action}")
    return ssh_run(entry, collector_command(entry, action, *options), runner=runner)


# --- состояние сбора для списка серверов -------------------------------------

RUNNING, STOPPED, EMPTY, UNREACHABLE = "running", "stopped", "empty", "unreachable"


def collector_state(output: str) -> str:
    """Вывод `collect_capture.sh status` -> running / stopped / empty."""
    if "идёт запись" in output:
        return RUNNING
    if "Сессий в" in output and "нет" in output:
        return EMPTY
    return STOPPED


def check_all(entries: list[ServerEntry], runner=subprocess.run, workers: int = 8) -> dict:
    """Статус сбора на всех серверах сразу: {имя: (состояние, вывод)}.

    Параллельно: недоступный сервер ждёт ConnectTimeout и не задерживает
    остальные.
    """
    from concurrent.futures import ThreadPoolExecutor

    def one(entry):
        try:
            output = collector(entry, "status", runner=runner)
        except ServerError as exc:
            return entry.name, (UNREACHABLE, str(exc))
        return entry.name, (collector_state(output), output)

    if not entries:
        return {}
    with ThreadPoolExecutor(max_workers=min(workers, len(entries))) as pool:
        return dict(pool.map(one, entries))


# --- состояние сбора подробно: место, объём, сколько живёт файл в кольце ------

SESSION_JSON_MARK = "--- session.json ---"
SPAN_MARK = "--- capture mtimes ---"
_SIZE_UNITS = {"K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024}


def _size_mb(text: str) -> float | None:
    """Размер из `du -h` («1.2G», «640M») в МБ."""
    match = re.match(r"^\s*([\d.,]+)\s*([KMGT])?", text or "")
    if not match:
        return None
    return float(match.group(1).replace(",", ".")) * _SIZE_UNITS.get(match.group(2) or "M", 1.0)


def collection_command(entry: ServerEntry) -> str:
    """status и session.json последней сессии одним входом по SSH."""
    show_session = (
        f"d=$(ls -1d {shlex.quote(entry.remote)}/*/ 2>/dev/null | sort | tail -1); "
        f'[ -n "$d" ] && echo {shlex.quote(SESSION_JSON_MARK)} && cat "$d/session.json" && '
        f'echo {shlex.quote(SPAN_MARK)} && stat -c %Y "$d"/capture_*.pcapng 2>/dev/null | sort -n | sed -n "1p;\\$p"; true'
    )
    return (collector_command(entry, "status") + "; "
            + _sudo(entry) + "sh -c " + shlex.quote(show_session))


def parse_collection(output: str, now: datetime | None = None) -> dict:
    """Вывод collection_command -> числа для статистики (None, где неизвестно).

    ring_hours — сколько часов файл живёт в кольце до перезаписи при
    нынешнем темпе: если это меньше интервала скачивания, часть захвата
    пропадает, не дойдя до этого компьютера.
    """
    status, _, rest = output.partition(SESSION_JSON_MARK)
    session_text, _, span_text = rest.partition(SPAN_MARK)
    stats = {"state": collector_state(status), "files": None, "ssh_logs": None,
             "used_mb": None, "free_mb": None, "started": None, "ring_mb": None,
             "rate_mb_h": None, "ring_hours": None}
    if match := re.search(r"Файлов захвата:\s*(\d+), журналов SSH:\s*(\d+)", status):
        stats["files"], stats["ssh_logs"] = int(match.group(1)), int(match.group(2))
    if match := re.search(r"Занято:\s*(\S+), свободно на диске:\s*(\d+)", status):
        stats["used_mb"], stats["free_mb"] = _size_mb(match.group(1)), int(match.group(2))
    try:
        session = json.loads(session_text) if session_text.strip() else {}
    except ValueError:
        session = {}
    if session.get("started_utc"):
        try:
            stats["started"] = datetime.strptime(session["started_utc"], "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            pass
    if session.get("ring_files") and session.get("file_mb"):
        stats["ring_mb"] = float(session["ring_files"]) * float(session["file_mb"])
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    ring_files = int(session.get("ring_files") or 0)
    stamps = [int(value) for value in span_text.split() if value.isdigit()]
    if ring_files and stats["files"] is not None and stats["files"] >= ring_files and len(stamps) == 2:
        # Кольцо заполнено: файл живёт столько, сколько между старейшим и новейшим
        stats["ring_hours"] = (stamps[1] - stamps[0]) / 3600 or None
        if stats["ring_hours"] and stats["used_mb"]:
            stats["rate_mb_h"] = stats["used_mb"] / stats["ring_hours"]
    elif stats["started"] and stats["used_mb"]:
        hours = (now - stats["started"]).total_seconds() / 3600
        if hours > 0.1:  # пока кольцо не заполнилось, место растёт с темпом записи
            stats["rate_mb_h"] = stats["used_mb"] / hours
            if stats["ring_mb"]:
                stats["ring_hours"] = stats["ring_mb"] / stats["rate_mb_h"]
    return stats


def collection_stats(entry: ServerEntry, runner=subprocess.run) -> dict:
    return parse_collection(ssh_run(entry, collection_command(entry), runner=runner))
