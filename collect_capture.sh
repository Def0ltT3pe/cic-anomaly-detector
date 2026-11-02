#!/usr/bin/env bash
# Сбор трафика сервера и журнала SSH для adapt_model.py.
#
# Пишет только заголовки пакетов (dumpcap -s 128) кольцом файлов, объём
# которого рассчитывается от свободного места с запасом; каждую минуту
# проверяет место и останавливает запись, если запас кончается (раньше
# захват без ограничения заполнил диск сервера). Раз в час выгружает журнал
# sshd за прошедший час в формате short-iso (с годом и часовым поясом) —
# по нему label_from_authlog.py размечает SSH-брутфорс.
#
# Использование (от root или пользователя из группы wireshark):
#   ./collect_capture.sh start  [опции]   # в фоне, переживает выход из SSH
#   ./collect_capture.sh status [--dir D]
#   ./collect_capture.sh stop   [--dir D]
#   ./collect_capture.sh run    [опции]   # на переднем плане (для отладки/tmux)
#   ./collect_capture.sh plan   [опции]   # показать расчёт и команду, ничего не запуская
#
# Опции:
#   --dir DIR          каталог сессий (по умолчанию /var/lib/cic-capture)
#   --iface IFACE      интерфейс (по умолчанию — интерфейс маршрута по умолчанию)
#   --duration DUR     сколько писать: 30m, 12h, 7d; 0 — до stop (по умолчанию 7d)
#   --reserve-gb N     сколько места оставить свободным всегда (по умолчанию 5)
#   --max-gb N         не больше N ГБ на захват, даже если места больше (по умолчанию 20)
#   --file-mb N        размер одного файла кольца, МБ (по умолчанию 500)
#   --snaplen N        байт на пакет; 128 хватает для заголовков (по умолчанию 128)
#   --check-every S    проверка места, с (по умолчанию 60)
#   --log-every S      выгрузка журнала SSH, с (по умолчанию 3600)
#   --filter BPF       фильтр захвата, например "not port 5353"
#
# Результат: DIR/<host>_<UTC-время>/ — capture_*.pcapng, ssh_*.log, session.json.
# Скопировать на компьютер:  scp -r root@SERVER:/var/lib/cic-capture/<сессия> .
# Дальше: pcap_to_flows.py -> label_from_authlog.py -> adapt_model.py (см. README).
#
# В захвате и журнале — IP-адреса и логины: храните их как приватные данные.

set -euo pipefail

DIR="/var/lib/cic-capture"
IFACE=""
DURATION="7d"
RESERVE_GB=5
MAX_GB=20
FILE_MB=500
SNAPLEN=128
CHECK_EVERY=60
LOG_EVERY=3600
FILTER=""

log() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }
die() { printf 'Ошибка: %s\n' "$*" >&2; exit 2; }

usage() { sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'; }

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dir) DIR="$2"; shift 2 ;;
            --iface) IFACE="$2"; shift 2 ;;
            --duration) DURATION="$2"; shift 2 ;;
            --reserve-gb) RESERVE_GB="$2"; shift 2 ;;
            --max-gb) MAX_GB="$2"; shift 2 ;;
            --file-mb) FILE_MB="$2"; shift 2 ;;
            --snaplen) SNAPLEN="$2"; shift 2 ;;
            --check-every) CHECK_EVERY="$2"; shift 2 ;;
            --log-every) LOG_EVERY="$2"; shift 2 ;;
            --filter) FILTER="$2"; shift 2 ;;
            -h|--help) usage; exit 0 ;;
            *) die "неизвестная опция: $1 (см. --help)" ;;
        esac
    done
    for name in RESERVE_GB MAX_GB FILE_MB SNAPLEN CHECK_EVERY LOG_EVERY; do
        [[ "${!name}" =~ ^[0-9]+$ ]] || die "--$(echo "$name" | tr 'A-Z_' 'a-z-') должно быть целым числом"
    done
    # Здесь, а не в duration_seconds: die внутри $(...) завершил бы только подоболочку
    [[ "$DURATION" =~ ^[0-9]+[smhd]?$ ]] || die "--duration: ожидается 30m, 12h, 7d или число секунд"
    (( FILE_MB >= 10 )) || die "--file-mb не меньше 10"
    (( SNAPLEN >= 96 )) || die "--snaplen не меньше 96 (иначе обрежутся заголовки TCP)"
}

# 30m / 12h / 7d / 3600 -> секунды
duration_seconds() {
    local value="$1"
    case "$value" in
        0) echo 0 ;;
        *[0-9]s) echo "${value%s}" ;;
        *[0-9]m) echo $(( ${value%m} * 60 )) ;;
        *[0-9]h) echo $(( ${value%h} * 3600 )) ;;
        *[0-9]d) echo $(( ${value%d} * 86400 )) ;;
        *) echo "$value" ;;  # формат уже проверен в parse_args
    esac
}

default_iface() {
    if command -v ip >/dev/null 2>&1; then
        ip route show default 2>/dev/null | awk '/default/ {for (i=1;i<=NF;i++) if ($i=="dev") {print $(i+1); exit}}'
    fi
}

# Свободное место в каталоге, МБ
free_mb() { df -Pk "$1" | awk 'NR==2 {print int($4/1024)}'; }

# Бюджет кольца: 80% от (свободно - запас), но не больше --max-gb.
# Печатает «files budget_mb free_mb».
ring_plan() {
    local free reserve budget files
    free=$(free_mb "$DIR")
    reserve=$(( RESERVE_GB * 1024 ))
    budget=$(( (free - reserve) * 8 / 10 ))
    (( budget > MAX_GB * 1024 )) && budget=$(( MAX_GB * 1024 ))
    files=$(( budget / FILE_MB ))
    echo "$files $budget $free"
}

# || true: без сессий ls завершается с ошибкой, а pipefail + set -e оборвали бы скрипт
session_root() { { ls -1d "$DIR"/*/ 2>/dev/null || true; } | sort | tail -1 | sed 's:/*$::'; }

# Журнал sshd за [since, until) — unit называется ssh (Debian/Ubuntu) или sshd
export_ssh_log() {
    local session="$1" since="$2" until="$3" out
    out="$session/ssh_$(date -u -d "@$since" '+%Y%m%dT%H%M%SZ').log"
    if command -v journalctl >/dev/null 2>&1; then
        # -q и grep: без «-- No entries --» и прочих служебных строк
        journalctl -q -u ssh -u sshd -o short-iso --no-pager \
            --since "@$since" --until "@$until" 2>/dev/null | grep -v '^-- ' >"$out" || true
        if [[ ! -s "$out" ]]; then
            # Нет unit-а с таким именем — берём по имени процесса
            journalctl -q -t sshd -t sshd-session -o short-iso --no-pager \
                --since "@$since" --until "@$until" 2>/dev/null | grep -v '^-- ' >"$out" || true
        fi
    fi
    if [[ ! -s "$out" ]]; then
        # Без journald: новые строки auth.log/secure с прошлой выгрузки
        local source offset_file offset size
        for source in /var/log/auth.log /var/log/secure; do
            [[ -r "$source" ]] || continue
            offset_file="$session/.offset_$(basename "$source")"
            offset=$(cat "$offset_file" 2>/dev/null || echo 0)
            size=$(stat -c %s "$source")
            (( size < offset )) && offset=0   # файл ротирован
            tail -c +"$(( offset + 1 ))" "$source" | grep -a 'sshd' >"$out" || true
            echo "$size" >"$offset_file"
            break
        done
    fi
    if [[ -s "$out" ]]; then
        log "журнал SSH: $(wc -l <"$out") строк -> $(basename "$out")"
    else
        rm -f "$out"
    fi
}

write_session_json() {
    local session="$1" started="$2" status="$3"
    cat >"$session/session.json" <<EOF
{
  "host": "$(hostname)",
  "iface": "$IFACE",
  "started_utc": "$(date -u -d "@$started" '+%Y-%m-%dT%H:%M:%SZ')",
  "updated_utc": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "status": "$status",
  "server_timezone": "$(date '+%Z %z')",
  "snaplen": $SNAPLEN,
  "file_mb": $FILE_MB,
  "ring_files": ${RING_FILES:-0},
  "filter": "${FILTER//\"/\\\"}",
  "free_mb": $(free_mb "$DIR")
}
EOF
}

check_tools() {
    command -v dumpcap >/dev/null 2>&1 || die "нет dumpcap: apt install tshark (или wireshark-common)"
    if [[ $EUID -ne 0 ]] && ! id -nG | tr ' ' '\n' | grep -qx wireshark; then
        die "нужен root или группа wireshark (usermod -aG wireshark \$USER и перелогиниться)"
    fi
}

prepare() {
    mkdir -p "$DIR"
    [[ -n "$IFACE" ]] || IFACE=$(default_iface)
    [[ -n "$IFACE" ]] || die "не удалось определить интерфейс, укажите --iface (список: dumpcap -D)"
    read -r RING_FILES BUDGET_MB FREE_MB < <(ring_plan)
    if (( RING_FILES < 2 )); then
        die "мало места: свободно ${FREE_MB} МБ, запас ${RESERVE_GB} ГБ — на кольцо из 2 файлов по ${FILE_MB} МБ не хватает. Освободите место или уменьшите --reserve-gb/--file-mb"
    fi
    DURATION_S=$(duration_seconds "$DURATION")
}

dumpcap_command() {
    local session="$1"
    DUMPCAP=(dumpcap -q -i "$IFACE" -s "$SNAPLEN"
             -b "filesize:$(( FILE_MB * 1000 ))" -b "files:$RING_FILES"
             -w "$session/capture.pcapng")
    (( DURATION_S > 0 )) && DUMPCAP+=(-a "duration:$DURATION_S")
    [[ -n "$FILTER" ]] && DUMPCAP+=(-f "$FILTER")
    return 0
}

cmd_plan() {
    prepare
    dumpcap_command "$DIR/<сессия>"
    echo "Интерфейс:     $IFACE"
    echo "Свободно:      $FREE_MB МБ в $DIR (запас $RESERVE_GB ГБ остаётся всегда)"
    echo "Кольцо:        $RING_FILES файлов по $FILE_MB МБ = $(( RING_FILES * FILE_MB )) МБ (бюджет $BUDGET_MB МБ, максимум $MAX_GB ГБ)"
    echo "Длительность:  $DURATION ($DURATION_S с; 0 — до stop)"
    echo "Контроль:      место раз в $CHECK_EVERY с, журнал SSH раз в $LOG_EVERY с"
    echo "Команда:       ${DUMPCAP[*]}"
}

cmd_run() {
    check_tools
    prepare
    local session started capture_pid last_log now
    started=$(date -u +%s)
    session="$DIR/$(hostname -s)_$(date -u -d "@$started" '+%Y%m%dT%H%M%SZ')"
    mkdir -p "$session"
    chmod 700 "$session"
    dumpcap_command "$session"

    log "сессия: $session"
    log "интерфейс $IFACE, кольцо $RING_FILES x $FILE_MB МБ, свободно $FREE_MB МБ, длительность $DURATION"
    "${DUMPCAP[@]}" 2>>"$session/dumpcap.log" &
    capture_pid=$!
    sleep 2
    if ! kill -0 "$capture_pid" 2>/dev/null; then
        # Нет прав, неверный интерфейс или фильтр — сразу видно причину
        write_session_json "$session" "$started" "failed to start"
        die "dumpcap не запустился: $(tail -3 "$session/dumpcap.log" 2>/dev/null | tr '\n' ' ')"
    fi
    echo "$capture_pid" >"$session/dumpcap.pid"
    write_session_json "$session" "$started" "running"

    STOPPING=0
    trap 'STOPPING=1' INT TERM
    last_log=$started
    while kill -0 "$capture_pid" 2>/dev/null && (( STOPPING == 0 )); do
        sleep "$CHECK_EVERY" & wait $! || true
        now=$(date -u +%s)
        if (( $(free_mb "$DIR") < RESERVE_GB * 1024 )); then
            log "ВНИМАНИЕ: свободно меньше $RESERVE_GB ГБ — останавливаю запись, чтобы не заполнить диск"
            write_session_json "$session" "$started" "stopped: low disk space"
            STOPPING=2
            break
        fi
        if (( now - last_log >= LOG_EVERY )); then
            export_ssh_log "$session" "$last_log" "$now"
            last_log=$now
            write_session_json "$session" "$started" "running"
        fi
    done

    kill "$capture_pid" 2>/dev/null || true
    wait "$capture_pid" 2>/dev/null || true
    rm -f "$session/dumpcap.pid"
    export_ssh_log "$session" "$last_log" "$(date -u +%s)"
    if (( STOPPING != 2 )); then
        write_session_json "$session" "$started" "finished"
    fi
    log "готово: $(ls -1 "$session"/capture_*.pcapng 2>/dev/null | wc -l) файлов захвата, $(du -sh "$session" | cut -f1)"
}

cmd_start() {
    check_tools
    prepare
    if pgrep -f "collect_capture.sh run" >/dev/null 2>&1; then
        die "сбор уже идёт (./collect_capture.sh status)"
    fi
    mkdir -p "$DIR"
    # setsid + nohup: процесс не завершится при выходе из SSH
    setsid nohup "$0" run "${ARGS[@]}" >>"$DIR/collector.log" 2>&1 < /dev/null &
    sleep 2
    log "запущено в фоне; журнал: $DIR/collector.log"
    cmd_status
}

cmd_status() {
    local session pid
    session=$(session_root)
    if [[ -z "$session" ]]; then
        echo "Сессий в $DIR нет."
        return 0
    fi
    echo "Последняя сессия: $session"
    if [[ -f "$session/dumpcap.pid" ]] && pid=$(cat "$session/dumpcap.pid") && kill -0 "$pid" 2>/dev/null; then
        echo "Состояние:        идёт запись (dumpcap PID $pid)"
    else
        echo "Состояние:        запись не идёт"
    fi
    echo "Файлов захвата:   $(ls -1 "$session"/capture_*.pcapng 2>/dev/null | wc -l), журналов SSH: $(ls -1 "$session"/ssh_*.log 2>/dev/null | wc -l)"
    echo "Занято:           $(du -sh "$session" | cut -f1), свободно на диске: $(free_mb "$DIR") МБ"
    [[ -f "$session/session.json" ]] && grep -E '"status"' "$session/session.json" | sed 's/^ */Статус:           /'
    return 0
}

cmd_stop() {
    local pids
    pids=$(pgrep -f "collect_capture.sh run" || true)
    if [[ -z "$pids" ]]; then
        echo "Сбор не запущен."
        return 0
    fi
    # TERM сборщику: он остановит dumpcap и выгрузит последний кусок журнала
    kill $pids
    log "остановка..."
    for _ in $(seq 1 30); do
        pgrep -f "collect_capture.sh run" >/dev/null 2>&1 || break
        sleep 1
    done
    cmd_status
}

main() {
    local command="${1:-}"
    [[ -n "$command" ]] || { usage; exit 2; }
    shift
    ARGS=("$@")
    parse_args "$@"
    case "$command" in
        start) cmd_start ;;
        run) cmd_run ;;
        stop) cmd_stop ;;
        status) cmd_status ;;
        plan) cmd_plan ;;
        -h|--help|help) usage ;;
        *) die "неизвестная команда: $command (start | stop | status | run | plan)" ;;
    esac
}

main "$@"
