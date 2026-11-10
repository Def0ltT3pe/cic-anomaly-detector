"""Streamlit-приложение: загрузка CSV/Parquet и поиск аномалий трафика."""

from __future__ import annotations

import hmac
import logging
import os
import shutil
import sys
import tempfile
from typing import NamedTuple

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.alerts import (
    DEFAULT_IMMEDIATE_PORTS,
    DEFAULT_MIN_FLOWS,
    aggregate_alerts,
    parse_trusted,
    summarize,
)
from modules.audit import log_event, read_recent
from modules.auth import (
    LOGIN_THROTTLE,
    AuthError,
    authenticate,
    can_manage_servers,
    can_use_llm,
    can_view_audit,
    load_users,
)
from modules.bruteforce import BruteForceError, load_bruteforce
from modules.calibration import CALIBRATION_FILE, CalibrationError, load_calibration
from modules.integrity import IntegrityError, verify_if_listed
from modules.flow_report import (
    DEFAULT_QUESTION,
    SYSTEM_PROMPT,
    build_flow_report,
    first_prompt,
    flagged_candidates,
    without_images,
)
from modules.llm_explain import (
    LlmUnavailableError,
    config_errors,
    configured_providers,
    default_provider,
    provider_label,
    providers,
    stream_chat,
)
from modules.charts import daily_chart, deviation_chart, hourly_chart, timeline_chart, top_chart
from modules.explainer import feature_deviations
from modules.model_catalog import ModelEntry, list_models
from modules import monitor as mon
from modules.monitor import RETAIN_DAYS, daily_counts, load_incidents
from modules import servers as srv
from modules.overview import (
    ANOMALY,
    ATTACK,
    SUSPICIOUS,
    alert_timeline,
    flow_table,
    incident_peers,
    llm_verdict,
    model_verdict,
    top_peers,
    top_ports,
)
from modules.model_loader import (
    STATS_FILE,
    ModelLoadError,
    load_model_objects,
    load_normal_stats,
    resolve_model_dir,
)
from modules.pipeline import DetectionResult, FeatureValidationError, detect_anomalies
from pcap_to_flows import ConversionError, convert_pcap, find_tshark

logger = logging.getLogger(__name__)

MAX_ROWS = 2_000_000
VIEWS = ("Обзор", "Инциденты", "Потоки", "Разбор потока")
VIEW_ICONS = {
    "Обзор": ":material/monitoring:",
    "Инциденты": ":material/crisis_alert:",
    "Потоки": ":material/table_rows:",
    "Разбор потока": ":material/troubleshoot:",
}
TABLE_LIMIT = 1000
TOP_FEATURES_CHART = 10
PCAP_EXTENSIONS = (".pcap", ".pcapng", ".cap")
TABLE_EXTENSIONS = (".parquet", ".csv")

st.set_page_config(
    page_title="Детектор аномалий трафика",
    page_icon=":material/radar:",
    layout="wide",
)


class Assets(NamedTuple):
    model: object
    scaler: object
    selector: object
    feature_cols: list
    means: pd.Series
    stds: pd.Series
    calibration: object
    brute: object
    errors: list


@st.cache_resource(show_spinner="Загружаем модель…", max_entries=4)
def _load_assets(model_dir: str) -> Assets:
    model, scaler, selector, feature_cols = load_model_objects(model_dir)
    means, stds = load_normal_stats(feature_cols=feature_cols, model_dir=model_dir)
    errors = []
    try:
        calibration = load_calibration(model_dir)
    except CalibrationError as exc:
        logger.warning("Калибровка не загружена: %s", exc)
        calibration = None
        errors.append(f"Калибровка не загружена: {exc}")

    try:
        brute = load_bruteforce(model_dir)
    except BruteForceError as exc:
        logger.warning("BruteForce-модель не загружена: %s", exc)
        brute = None
        errors.append(f"BruteForce-модель не загружена: {exc}")

    try:
        unprotected = verify_if_listed(model_dir, [CALIBRATION_FILE, STATS_FILE])
    except IntegrityError:
        unprotected = []  # несовпадение хэша уже обработано загрузчиками выше
    if unprotected:
        errors.append(
            "Не защищены подписью: " + ", ".join(unprotected)
            + " — переподпишите модели: python make_checksums.py"
        )

    return Assets(model, scaler, selector, feature_cols, means, stds, calibration, brute, errors)


def _app_password() -> str:
    password = (os.environ.get("APP_PASSWORD") or "").strip()
    if password:
        return password
    try:
        return str(st.secrets.get("APP_PASSWORD", "")).strip()
    except Exception:
        return ""


def _register_failed_login(username: str) -> None:
    attempts = LOGIN_THROTTLE.failed(username)
    log_event(username or "-", "-", "login_failed", attempt=attempts)


def _login_locked(username: str) -> bool:
    """Логин заблокирован после серии неудач (в любой сессии): показать ошибку."""
    seconds = LOGIN_THROTTLE.locked_for(username)
    if seconds <= 0:
        return False
    log_event(username or "-", "-", "login_locked")
    st.error(f"Слишком много неудачных попыток. Повторите через {int(seconds) + 1} с.")
    return True


def _login_screen() -> None:
    """Логин по users.json; fallback — общий APP_PASSWORD; иначе открытый режим
    (если не запрещён ALLOW_OPEN_MODE=0)."""
    if st.session_state.get("user"):
        return

    try:
        users = load_users()
    except AuthError as exc:
        st.error(f"Ошибка конфигурации пользователей: {exc}")
        st.stop()

    shared = _app_password()
    if not users and not shared:
        if os.environ.get("ALLOW_OPEN_MODE", "1").strip() == "0":
            # В Docker открытый режим выключен: без пароля вошёл бы кто угодно как admin
            st.title("Вход не настроен")
            st.error(
                "Создайте пользователя (`python manage_users.py add admin --role admin`, "
                "файл config/users.json) или задайте APP_PASSWORD. Открыть приложение "
                "без пароля можно только явно: ALLOW_OPEN_MODE=1."
            )
            st.stop()
        st.session_state.user = "anonymous"
        st.session_state.role = "admin"
        st.session_state.auth_mode = "open"
        return

    st.title("Вход")
    if users:
        with st.form("login_form"):
            username = st.text_input("Логин")
            password = st.text_input("Пароль", type="password")
            submitted = st.form_submit_button("Войти", type="primary")
        if submitted and not _login_locked(username):
            user = authenticate(username, password)
            if user is not None:
                LOGIN_THROTTLE.succeeded(username)
                st.session_state.user = user.username
                st.session_state.role = user.role
                st.session_state.auth_mode = "users"
                log_event(user.username, user.role, "login", mode="password")
                st.rerun()
            _register_failed_login(username)
            st.error("Неверный логин или пароль")
    else:
        with st.form("login_form"):
            password = st.text_input("Пароль", type="password")
            submitted = st.form_submit_button("Войти", type="primary")
        if submitted and not _login_locked("shared"):
            if hmac.compare_digest(password.encode("utf-8"), shared.encode("utf-8")):
                LOGIN_THROTTLE.succeeded("shared")
                st.session_state.user = "shared"
                st.session_state.role = "admin"
                st.session_state.auth_mode = "shared"
                log_event("shared", "admin", "login", mode="shared")
                st.rerun()
            _register_failed_login("shared")
            st.error("Неверный пароль")

    st.stop()


def _init_state() -> None:
    defaults = {
        "analysis": None,
        "explanations": {},
        "df": None,
        "file_id": None,
        "brute": None,
        "flows_parquet": None,
        "model_dir": None,
        "llm_provider": None,
        "llm_keys": {},
        "user": None,
        "role": None,
        "auth_mode": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_audit_panel() -> None:
    with st.expander("Журнал аудита"):
        entries = read_recent(20)
        if not entries:
            st.caption("Пока пусто")
            return
        for entry in reversed(entries):
            details = {
                key: value
                for key, value in entry.items()
                if key not in ("ts", "user", "role", "action")
            }
            line = (
                f"{entry.get('ts', '')} · {entry.get('user', '')} · "
                f"{entry.get('action', '')}"
            )
            if details:
                line += " · " + " ".join(f"{key}={value}" for key, value in details.items())
            st.text(line)


def _render_identity() -> None:
    user = st.session_state.get("user")
    role = st.session_state.get("role")
    if not user:
        return

    _group_title(":material/account_circle:", "Аккаунт")
    st.markdown(f"**{user}**, роль `{role}`")
    if st.session_state.get("auth_mode") == "open":
        st.warning(
            "Аутентификация не настроена. Добавьте пользователя "
            "(`python manage_users.py add admin --role admin`) "
            "или задайте APP_PASSWORD."
        )

    if st.button("Выйти", key="logout", icon=":material/logout:"):
        log_event(user, role, "logout")
        for key in (
            "user",
            "role",
            "auth_mode",
            "analysis",
            "df",
            "file_id",
            "explanations",
            "brute",
            "llm_keys",
        ):
            st.session_state.pop(key, None)
        st.rerun()

    if can_view_audit(role):
        _render_audit_panel()


def _read_table(file) -> pd.DataFrame:
    name = (getattr(file, "name", "") or "").lower()

    if name.endswith(".parquet"):
        file.seek(0)
        df = pd.read_parquet(file)
    elif name.endswith(".csv"):
        file.seek(0)
        try:
            df = pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding="cp1252")
    else:
        raise ValueError("Поддерживаются файлы .pcap, .pcapng, .cap, .csv и .parquet.")
    return df


def _convert_pcap_upload(file, tshark: str) -> pd.DataFrame:
    """pcap из загрузки -> потоки; временная копия удаляется сразу после."""
    suffix = os.path.splitext(file.name)[1].lower()
    with st.status(f"{file.name}: считаем потоки…") as status, tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "capture" + suffix)
        file.seek(0)
        with open(path, "wb") as handle:
            shutil.copyfileobj(file, handle, 1 << 20)
        flows, stats = convert_pcap(
            path,
            tshark,
            progress=lambda count: status.update(
                label=f"{file.name}: прочитано пакетов {count:,}…"
            ),
        )
        status.update(
            label=(
                f"{file.name}: пакетов {stats.packets:,}, потоков {stats.flows:,} "
                f"(коротких отброшено {stats.dropped:,})"
            ),
            state="complete",
        )
        if stats.warning:
            st.warning(f"{file.name}: файл обрезан, прочитано сколько удалось — {stats.warning}")
    return flows


def _load_uploads(files) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Все загруженные файлы в одну таблицу и отдельно потоки из pcap (None — pcap не было).

    Потоки из pcap отдаются отдельно для выгрузки в Parquet: у CSV время бывает
    текстом, и смешанная колонка Timestamp в Parquet не записывается."""
    frames = []
    pcap_frames = []
    tshark = None
    for file in files:
        name = (file.name or "").lower()
        if name.endswith(PCAP_EXTENSIONS):
            if tshark is None:
                tshark = find_tshark(os.environ.get("TSHARK") or None)
            frame = _convert_pcap_upload(file, tshark)
            pcap_frames.append(frame)
        else:
            frame = _read_table(file)
        if frame is None or frame.empty:
            raise ValueError(f"{file.name}: нет ни одной строки данных.")
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if len(df) > MAX_ROWS:
        raise ValueError(f"Слишком много строк: {len(df):,}. Максимум — {MAX_ROWS:,}.")
    if not pcap_frames:
        return df, None
    if len(pcap_frames) == len(frames):
        return df, df
    return df, pd.concat(pcap_frames, ignore_index=True)


def _flow_title(result: DetectionResult, index: int, brute, brute_probs) -> str:
    row = result.df.iloc[index]
    parts = [f"#{index}"]
    if all(column in row.index for column in ("Src IP", "Dst IP", "Dst Port")):
        parts.append(f"{row['Src IP']} → {row['Dst IP']}:{row['Dst Port']}")
    if brute is not None and brute_probs is not None and brute_probs[index] >= brute.threshold:
        parts.append(f"{brute.display_name} p={brute_probs[index]:.2f}")
    if result.mask[index]:
        parts.append(f"IF {result.scores[index]:.3f}")
    return " · ".join(parts)


def _ask_llm(
    user, role, provider: str, api_key, chat: list[dict], question: str, report_for_llm: str,
    index: int,
) -> bool:
    """Вопрос в чат по потоку; ответ показывается по мере генерации. True — ответ сохранён."""
    content = first_prompt(report_for_llm, question) if not chat else question
    messages = [{"role": m["role"], "content": m["content"]} for m in chat]
    messages.append({"role": "user", "content": content})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(without_images(
                stream_chat(provider, messages, system=SYSTEM_PROMPT, api_key=api_key)
            ))
        except Exception as exc:
            if not isinstance(exc, LlmUnavailableError):  # неожиданная ошибка SDK
                logger.exception("LLM error")
            st.warning(str(exc) if isinstance(exc, LlmUnavailableError) else f"Ошибка LLM: {exc}")
            log_event(user, role, "llm_ask", row=index, provider=provider, ok=False, error=str(exc))
            return False
    answer = answer if isinstance(answer, str) else "".join(map(str, answer))
    chat.append({"role": "user", "content": content, "shown": question})
    chat.append({"role": "assistant", "content": answer, "shown": answer, "provider": provider})
    log_event(user, role, "llm_ask", row=index, provider=provider, ok=True, error=None)
    return True


def _render_flow_review(
    result: DetectionResult,
    means,
    stds,
    brute,
    brute_probs,
    brute_mask,
    incidents,
    provider: str | None,
) -> None:
    """Любой поток: факты моделей, вопрос к ИИ, уточнения в чате, отчёт в Markdown."""
    candidates = flagged_candidates(result, brute_probs, brute_mask)
    target = st.session_state.pop("goto_flow", None)
    if target is None and "flow_mode" not in st.session_state:
        # Streamlit забывает виджеты скрытой вкладки — возвращаем прежний поток
        target = st.session_state.get("review_flow")
    if target is not None:
        # Поток выбран кликом в таблице: ставим его в виджеты до их создания
        if target in candidates:
            st.session_state.flow_mode, st.session_state.flow_pick = "Из тревог", target
        else:
            st.session_state.flow_mode, st.session_state.flow_row = "По номеру строки", target
    modes = (["Из тревог"] if candidates else []) + ["По номеру строки"]
    mode = st.radio("Какой поток разобрать", modes, horizontal=True, key="flow_mode")
    if mode == "Из тревог":
        index = st.selectbox(
            f"Тревоги ({len(candidates):,}, сначала самые уверенные)",
            candidates,
            format_func=lambda i: _flow_title(result, i, brute, brute_probs),
            key="flow_pick",
        )
    else:
        index = int(
            st.number_input(
                "Номер строки", min_value=0, max_value=result.total - 1, value=0, step=1,
                key="flow_row",
            )
        )
    index = int(index)
    st.session_state.review_flow = index

    report_kwargs = dict(
        brute=brute, brute_probs=brute_probs, brute_mask=brute_mask, incidents=incidents
    )
    report = build_flow_report(result, index, means, stds, mask_ips=False, **report_kwargs)
    in_incident = incidents is not None and bool(len(incidents)) and bool(
        incidents["rows"].apply(lambda rows: index in rows).any()
    )
    level, text = model_verdict(
        bool(result.mask[index]),
        bool(brute_mask[index]) if brute_mask is not None else None,
        in_incident,
    )
    _verdict_box(level, f"По моделям — {text}")

    chat = st.session_state.explanations.setdefault(index, [])
    answers = [m["shown"] for m in chat if m["role"] == "assistant"]
    ai_level = llm_verdict(answers[0]) if answers else None
    if ai_level is not None:
        _verdict_box(ai_level, "По оценке ИИ — " + VERDICT_WORDS[ai_level])

    facts, chart = st.columns([3, 2])
    with facts.expander("Факты о потоке", expanded=True):
        st.markdown(report)
    deviations = feature_deviations(result.X.iloc[index], means, stds, TOP_FEATURES_CHART)
    if not deviations.empty:
        chart.caption("Чем поток отличается от нормы (наведите на столбец)")
        chart.altair_chart(deviation_chart(deviations, _theme()), width="stretch")

    for message in chat:
        with st.chat_message(message["role"]):
            st.markdown(message["shown"])

    user = st.session_state.get("user")
    role = st.session_state.get("role")
    if not can_use_llm(role):
        st.caption("Роль viewer: вопросы к ИИ отключены, доступны только факты моделей.")
    elif provider is None:
        st.caption(
            "ИИ не настроен: задайте GEMINI_API_KEY или OPENAI_API_KEY / OPENAI_BASE_URL."
        )
    else:
        # Подпись и параметры поля не меняются после первого ответа: иначе
        # Streamlit считает его новым виджетом и теряет введённый вопрос
        with st.form(key=f"ask_{index}", clear_on_submit=True):
            question = st.text_area(
                "Вопрос к ИИ (пустой первый вопрос — «почему сработало»)",
                placeholder=DEFAULT_QUESTION,
                height=80,
            )
            asked = st.form_submit_button(
                f"Спросить · {provider_label(provider)}", type="primary"
            )
        st.caption("В LLM уходят признаки и решения моделей; адреса замаскированы.")
        question = question.strip() or ("" if chat else DEFAULT_QUESTION)
        if asked and question:
            masked = build_flow_report(result, index, means, stds, mask_ips=True, **report_kwargs)
            api_key = st.session_state.llm_keys.get(provider)
            if _ask_llm(user, role, provider, api_key, chat, question, masked, index):
                st.rerun()  # история над формой, как при следующих вопросах

    document = [f"# Разбор потока #{index}", "", report]
    if chat:
        document += ["", "## Разбор ИИ"]
        for message in chat:
            who = (
                f"**Ответ ({provider_label(message['provider'])}):**"
                if message["role"] == "assistant"
                else "**Вопрос:**"
            )
            document += ["", who, "", message["shown"]]
    st.download_button(
        "Скачать отчёт (Markdown)",
        "\n".join(document),
        file_name=f"flow_{index}.md",
        mime="text/markdown",
        key=f"report_{index}",
    )


def _compute_brute(result: DetectionResult, brute):
    if brute is None:
        return None
    try:
        probabilities = brute.predict_proba(result.X)
        return probabilities, probabilities >= brute.threshold
    except BruteForceError as exc:
        st.warning(f"BruteForce-модель не применилась: {exc}")
        return None


def _compute_incidents(result: DetectionResult, brute_probs, brute_mask):
    """Инциденты по внешнему адресу за час; None — если адресов в файле нет."""
    try:
        trusted = parse_trusted(os.environ.get("TRUSTED_IPS", ""))
    except ValueError as exc:
        st.warning(f"TRUSTED_IPS не разобран ({exc}) — доверенных адресов нет.")
        trusted = []
    try:
        return aggregate_alerts(result.df, brute_mask, brute_probs, trusted=trusted)
    except Exception as exc:  # битые адреса/время в загруженном файле
        logger.exception("Инциденты не собраны")
        st.warning(f"Инциденты не собраны: {exc}")
        return None


VERDICT_WORDS = {
    ATTACK: "атака",
    SUSPICIOUS: "подозрительно",
    ANOMALY: "необычный поток",
    "normal": "вероятно, ложное срабатывание или норма",
}


def _verdict_box(level: str, text: str) -> None:
    """Плашка вердикта: цвет и значок по уровню, текст всегда рядом (не только цвет)."""
    if level == ATTACK:
        st.error(text, icon="🚨")
    elif level == SUSPICIOUS:
        st.warning(text, icon="⚠️")
    elif level == ANOMALY:
        st.info(text, icon="🔎")
    else:
        st.success(text, icon="✅")


def _theme() -> str | None:
    try:
        return st.context.theme.type
    except AttributeError:  # старый Streamlit без st.context.theme
        return None


def _goto_flow(index: int) -> None:
    """Открыть поток во вкладке «Разбор потока» (виджеты обновятся при перезапуске)."""
    st.session_state.goto_flow = int(index)
    st.session_state.goto_view = "Разбор потока"
    st.rerun()


def _jump_on_select(event, table: pd.DataFrame, key: str, column: str = "#") -> None:
    """Клик по строке таблицы -> разбор потока. Повторно та же строка не срабатывает:
    при возврате на вкладку Streamlit восстанавливает прежний выбор."""
    rows = event.selection.rows if event is not None else []
    if not rows:
        st.session_state.pop(key, None)
        return
    index = int(table.iloc[rows[0]][column])
    if st.session_state.get(key) != index:
        st.session_state[key] = index
        _goto_flow(index)


def _render_overview(result: DetectionResult, brute, brute_probs, brute_mask, incidents) -> None:
    cols = st.columns(5)
    cols[0].metric("Потоков", f"{result.total:,}", border=True)
    cols[1].metric("Аномалий IF", f"{result.n_anomalies:,}",
                   help=f"IsolationForest: {result.share:.2%} потоков", border=True)
    if brute_mask is not None:
        cols[2].metric("Тревог", f"{int(brute_mask.sum()):,}",
                       help=f"{brute.display_name}: порог {brute.threshold:.3f}, FPR ≈ "
                            f"{float(brute.metadata.get('target_fpr', 0)):.2%}", border=True)
        cols[3].metric("Инцидентов", "—" if incidents is None else f"{len(incidents):,}",
                       border=True)
        cols[4].metric("Адресов", "—" if incidents is None else f"{incident_peers(incidents):,}",
                       help="Внешние адреса, от которых есть инциденты", border=True)
    else:
        cols[2].metric("Классификатор атак", "нет модели", border=True)

    if result.info.get("inf_replaced") or result.info.get("nan_filled"):
        st.caption(
            f"В данных найдено inf: {result.info.get('inf_replaced', 0)}, "
            f"пропусков: {result.info.get('nan_filled', 0)} — заменены средними нормы."
        )

    masks = {"IsolationForest": result.mask}
    if brute_mask is not None:
        masks[brute.display_name] = brute_mask
    timeline = alert_timeline(result.df, masks)
    st.markdown("##### Тревоги по времени")
    if timeline is None:
        st.caption("В файле нет времени потоков (Timestamp) — график недоступен.")
    else:
        st.altair_chart(timeline_chart(timeline, list(masks), _theme()), width="stretch")

    flagged = brute_mask if brute_mask is not None else result.mask
    source = brute.display_name if brute_mask is not None else "IsolationForest"
    peers = top_peers(result.df, flagged)
    if peers is None:
        st.caption("В файле нет адресов и портов — топ-списки недоступны.")
        return
    left, right = st.columns(2)
    left.markdown(f"##### Адреса с наибольшим числом тревог ({source})")
    right.markdown(f"##### Порты сервера под ударом ({source})")
    if peers.empty:
        left.caption("Тревог нет.")
        return
    left.altair_chart(top_chart(peers, "Адрес", _theme()), width="stretch")
    right.altair_chart(top_chart(top_ports(result.df, flagged), "Порт", _theme()),
                       width="stretch")


def _render_incidents(brute, brute_probs, brute_mask, incidents) -> None:
    """Подозрительные потоки, сведённые в инциденты по внешнему адресу за час."""
    if brute_mask is None:
        st.info("Инциденты собираются из срабатываний классификатора атак, а его у модели нет.")
        return
    if incidents is None:
        st.caption(
            "Инциденты по адресам не собраны: в файле нет Src/Dst IP и портов "
            "(их добавляет pcap_to_flows.py)."
        )
        return
    if not len(incidents):
        st.success(
            "Ни один адрес не дал инцидента: тревог нет или они единичные — "
            "скорее всего, ложные."
        )
        return
    # Порядок как в таблице: сначала больше потоков; для перехода — самый уверенный поток
    ordered = incidents.sort_values(["flows", "max_proba"], ascending=False).reset_index(drop=True)
    ordered["top_row"] = [max(rows, key=lambda r: brute_probs[r]) for rows in ordered["rows"]]
    view = ordered.drop(columns=["rows"]).rename(columns={
        "window_start": "Час", "peer": "Адрес", "flows": "Потоков",
        "services": "Порты сервера", "max_proba": "Макс. вероятность",
        "first_seen": "Первый", "last_seen": "Последний", "reason": "Почему инцидент",
        "top_row": "#",
    })
    st.caption("Нажмите на строку — откроется разбор самого уверенного потока инцидента.")
    event = st.dataframe(
        view, hide_index=True, on_select="rerun", selection_mode="single-row",
        key="incidents_table",
        column_config={
            "Час": st.column_config.DatetimeColumn(format="DD.MM HH:mm"),
            "Первый": st.column_config.DatetimeColumn(format="HH:mm:ss"),
            "Последний": st.column_config.DatetimeColumn(format="HH:mm:ss"),
            "Макс. вероятность": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
            "#": st.column_config.NumberColumn("Поток", help="Номер самого уверенного потока"),
        },
    )
    st.caption(
        f"{summarize(incidents, brute_mask)}. Инцидент — поток на порт "
        f"{', '.join(map(str, DEFAULT_IMMEDIATE_PORTS))} или ≥{DEFAULT_MIN_FLOWS} "
        "подозрительных потоков с одного адреса за час; единичные срабатывания "
        "на других портах — почти всегда ложные."
    )
    _jump_on_select(event, view, "incidents_jump")


def _render_flows(result: DetectionResult, brute, brute_probs, brute_mask) -> None:
    filters = []
    if brute_mask is not None:
        filters.append(f"Тревоги: {brute.display_name}")
    filters += ["Аномалии IsolationForest", "Все потоки"]
    if "flows_filter" not in st.session_state and st.session_state.get("flows_choice") in filters:
        st.session_state.flows_filter = st.session_state.flows_choice  # после другой вкладки
    choice = st.radio("Показать", filters, horizontal=True, key="flows_filter",
                      label_visibility="collapsed")
    st.session_state.flows_choice = choice
    if choice.startswith("Тревоги"):
        rows = np.flatnonzero(brute_mask)
        rows = rows[np.argsort(-brute_probs[rows], kind="stable")]
    elif choice.startswith("Аномалии"):
        rows = result.indices[np.argsort(result.scores[result.indices], kind="stable")]
    else:
        rows = np.arange(result.total)
    if not len(rows):
        st.success("Таких потоков нет.")
        return
    shown = rows[:TABLE_LIMIT]
    table = flow_table(result, shown, brute_probs)
    st.caption(
        f"Показано {len(shown):,} из {len(rows):,} (сначала самые подозрительные). "
        "Нажмите на строку — откроется разбор потока."
    )
    event = st.dataframe(
        table, hide_index=True, on_select="rerun", selection_mode="single-row",
        key=f"flows_table_{filters.index(choice)}",
        column_config={
            "#": st.column_config.NumberColumn("Строка", format="%d"),
            "Вероятность атаки": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="%.2f",
                help="Уверенность классификатора; тревога — выше порога "
                     + (f"{brute.threshold:.2f}" if brute is not None else "")),
            "IF score": st.column_config.NumberColumn(
                help="Меньше — необычнее; аномалия — не выше порога IsolationForest"),
        },
    )
    _jump_on_select(event, table, f"flows_jump_{filters.index(choice)}")


def _render_analysis(
    result: DetectionResult,
    means,
    stds,
    brute=None,
    brute_payload=None,
    provider: str | None = None,
) -> None:
    brute_probs = None
    brute_mask = None
    if brute is not None and brute_payload is not None:
        brute_probs, brute_mask = brute_payload
    incidents = (
        _compute_incidents(result, brute_probs, brute_mask) if brute_mask is not None else None
    )

    pending = st.session_state.pop("goto_view", None)
    if pending is not None:
        st.session_state.view = pending  # до создания переключателя вкладок
    elif st.session_state.get("view") not in VIEWS:
        st.session_state.view = VIEWS[0]
    view = st.segmented_control(
        "Раздел", VIEWS, key="view", required=True, label_visibility="collapsed",
        format_func=lambda name: f"{VIEW_ICONS[name]} {name}",
    )
    if view == "Обзор":
        _render_overview(result, brute, brute_probs, brute_mask, incidents)
    elif view == "Инциденты":
        _render_incidents(brute, brute_probs, brute_mask, incidents)
    elif view == "Потоки":
        _render_flows(result, brute, brute_probs, brute_mask)
    else:
        _render_flow_review(
            result, means, stds, brute, brute_probs, brute_mask, incidents, provider
        )


def _reset_analysis() -> None:
    st.session_state.analysis = None
    st.session_state.explanations = {}
    st.session_state.brute = None


def _render_llm_keys(user, role, registry) -> dict:
    """Ключ API из интерфейса: только в памяти этой сессии, не на диске и не в аудите."""
    keys = st.session_state.llm_keys
    keyed = [p.id for p in registry.values() if p.key_required or p.key_env]
    with st.expander("Ключ API", expanded=not configured_providers(keys)):
        with st.form("llm_key_form", clear_on_submit=True):
            name = st.selectbox("Провайдер", keyed, format_func=lambda i: registry[i].name)
            key = st.text_input("Ключ", type="password")
            saved = st.form_submit_button("Сохранить на эту сессию")
        if saved and key.strip():
            keys[name] = key.strip()
            st.session_state.llm_provider = name
            log_event(user, role, "llm_key_set", provider=name)  # без самого ключа
        for name in keys:
            if name in registry:
                st.caption(f"{registry[name].name}: ключ введён")
        if keys and st.button("Забыть ключи", key="llm_keys_clear"):
            keys.clear()
            log_event(user, role, "llm_key_clear")
            st.rerun()
        st.caption(
            "Ключ хранится только в памяти этой сессии и пропадает при выходе. "
            "Адреса и модели — в config/llm.json и переменных окружения."
        )
    return keys


def _select_llm_provider(user, role) -> str | None:
    """LLM для разбора потоков: по умолчанию LLM_PROVIDER, выбор — на сессию."""
    _group_title(":material/smart_toy:", "ИИ")
    if not can_use_llm(role):
        st.caption("Недоступен для роли viewer")
        return None
    registry = providers()
    for problem in config_errors():
        st.warning(f"config/llm.json: {problem}")
    keys = _render_llm_keys(user, role, registry)
    ready = configured_providers(keys)
    # В списке все варианты; без ключа — с пометкой, ключ вводится выше
    options = ready + [i for i in registry if i not in ready]
    current = st.session_state.llm_provider or default_provider(keys) or options[0]
    provider = st.selectbox(
        "LLM для разбора",
        options,
        index=options.index(current) if current in options else 0,
        format_func=lambda i: provider_label(i) + ("" if i in ready else " — нужен ключ"),
    )
    if provider not in ready:
        st.caption(
            f"Для «{registry[provider].name}» введите ключ в «Ключ API»"
            + (f" или задайте {registry[provider].key_env}" if registry[provider].key_env else "")
            + "."
        )
    st.session_state.llm_provider = provider
    return provider


def _select_model(user, role) -> ModelEntry | None:
    """Выбор модели в боковой панели; действует только в этой сессии."""
    try:
        entries = list_models()
    except OSError as exc:
        st.warning(f"Не удалось прочитать список моделей: {exc}")
        entries = []
    if not entries:
        return None

    paths = [entry.path for entry in entries]
    current = st.session_state.model_dir
    index = paths.index(current) if current in paths else 0
    entry = st.selectbox(
        "Какую модель применять",
        entries,
        index=index,
        format_func=lambda item: item.label,
        help="Подписанные модели из папок models* и MODELS_ROOT",
    )
    if current is not None and current != entry.path:
        _reset_analysis()  # тревоги прежней модели к новой не относятся
        log_event(user, role, "model_select", model=entry.name)
    st.session_state.model_dir = entry.path
    return entry


def _percent(value, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value):.{digits}%}"


def _evaluation_rows(metadata: dict) -> tuple[str, list[dict]]:
    evaluation = metadata.get("evaluation_test") or metadata.get("evaluation") or {}
    title = "отложенные файлы" if metadata.get("evaluation_test") else "отложенная часть"
    rows = []
    for key, name in (("isolation_forest", "IsolationForest"), ("supervised", None)):
        block = evaluation.get(key)
        if not isinstance(block, dict) or not isinstance(block.get("adapted"), dict):
            continue
        stats = block["adapted"]
        rows.append({
            "Модель": name or block.get("display_name", "Supervised"),
            "Полнота": _percent(stats.get("recall")),
            "FPR": _percent(stats.get("fpr"), digits=3),
            "AUC": f"{stats['auc']:.3f}" if stats.get("auc") is not None else "—",
        })
    return title, rows


def _render_model_card(entry: ModelEntry) -> None:
    metadata = entry.metadata
    with st.expander("О модели"):
        st.caption(f"`{entry.name}`")
        if not metadata:
            st.caption("metadata.json нет.")
            return
        facts = {
            "Версия": metadata.get("version"),
            "Создана": str(metadata.get("created_at", ""))[:16].replace("T", " "),
            "База": metadata.get("base_model"),
            "Целевой FPR": (
                f"{float(metadata['target_fpr']):.2%}" if metadata.get("target_fpr") else None
            ),
        }
        st.markdown("\n".join(f"- {key}: {value}" for key, value in facts.items() if value))
        if metadata.get("note"):
            st.caption(metadata["note"])
        title, rows = _evaluation_rows(metadata)
        if rows:
            st.caption(f"Проверка ({title}):")
            st.dataframe(pd.DataFrame(rows), hide_index=True)


def _render_model_status(entry: ModelEntry | None, assets: Assets, user, role) -> None:
    st.caption(":green[:material/check_circle:] Модель и статистики загружены")
    if entry is not None:
        _render_model_card(entry)
    for message in assets.errors:
        st.warning(message, icon=":material/warning:")
        log_event(user, role, "model_warning", message=message)
    st.caption(f"Признаков: {len(assets.feature_cols)}")
    calibration = assets.calibration
    if calibration is not None:
        st.caption(
            f"Порог: score ≤ {calibration.threshold:.4f} "
            f"(целевой FPR ≈ {calibration.target_fpr:.1%})"
        )
    else:
        st.caption("Порог: встроенный (contamination модели)")
    if calibration is not None and calibration.source == "target network holdout":
        st.caption("Модель адаптирована под целевую сеть (adapt_model.py)")
    brute = assets.brute
    st.caption(
        (f"{brute.display_name}: supervised" if brute is not None else "Supervised: нет модели")
    )


def _handle_upload(user, role) -> tuple[pd.DataFrame, str] | None:
    """Загрузка файлов: таблица и имена файлов, или None, если анализировать нечего."""
    uploaded = st.file_uploader(
        "Файлы с трафиком: pcap, pcapng, Parquet или CSV",
        type=list(PCAP_EXTENSIONS + TABLE_EXTENSIONS),
        accept_multiple_files=True,
    )
    if not uploaded:
        st.caption(
            "Можно выбрать несколько файлов сразу: они объединятся в одну таблицу. "
            "После загрузки появится кнопка «Анализировать»."
        )
        return None

    file_id = tuple(
        getattr(item, "file_id", None) or f"{item.name}:{item.size}" for item in uploaded
    )
    names = ", ".join(item.name for item in uploaded)
    if st.session_state.file_id != file_id:
        st.session_state.file_id = None
        st.session_state.df = None
        st.session_state.flows_parquet = None
        _reset_analysis()
        try:
            df, pcap_flows = _load_uploads(uploaded)
        except ConversionError as exc:
            log_event(user, role, "upload_error", file=names, error=str(exc))
            st.error(f"Не удалось разобрать захват: {exc} (путь к tshark можно задать в TSHARK)")
            return None
        except Exception as exc:
            log_event(user, role, "upload_error", file=names, error=str(exc))
            st.error(f"Не удалось прочитать файл: {exc}")
            return None

        st.session_state.file_id = file_id
        st.session_state.df = df
        if pcap_flows is not None:
            st.session_state.flows_parquet = pcap_flows.to_parquet(index=False)
        log_event(
            user,
            role,
            "upload",
            file=names,
            rows=len(df),
            columns=len(df.columns),
            pcap=pcap_flows is not None,
        )

    df = st.session_state.df
    if df is None:
        return None
    return df, names


def _run_analysis(df: pd.DataFrame, assets: Assets, user, role) -> None:
    try:
        with st.spinner("Ищем аномалии…"):
            result = detect_anomalies(
                df,
                assets.model,
                assets.scaler,
                assets.selector,
                assets.feature_cols,
                fill_values=assets.means,
                max_rows=MAX_ROWS,
                calibration=assets.calibration,
            )
    except FeatureValidationError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        logger.exception("Ошибка детекции")
        st.error(f"Ошибка при анализе: {exc}")
        return
    st.session_state.analysis = result
    st.session_state.explanations = {}
    brute_payload = _compute_brute(result, assets.brute)
    st.session_state.brute = brute_payload
    log_event(
        user,
        role,
        "analyze",
        rows=result.total,
        anomalies=result.n_anomalies,
        brute=int(brute_payload[1].sum()) if brute_payload is not None else None,
        threshold=result.threshold,
    )


def _group_title(icon: str, title: str) -> None:
    """Заголовок блока боковой панели."""
    st.markdown(f"**{icon} {title}**")


def _render_header(entry: ModelEntry | None) -> None:
    st.title(":primary[:material/radar:] Детектор аномалий трафика")
    st.caption(
        "Ищет необычные потоки и атаки в захватах трафика. Загрузите pcap/pcapng "
        "(потоки посчитает tshark) или CSV/Parquet с признаками CICFlowMeter."
        + (f" Модель: {entry.label}." if entry is not None else "")
    )


def main() -> None:
    _init_state()
    _login_screen()

    # Блоки боковой панели создаются заранее: аккаунт внизу, но выход
    # доступен, даже если модель не загрузилась
    model_box = st.sidebar.container(border=True)
    ai_box = st.sidebar.container(border=True)
    account_box = st.sidebar.container(border=True)
    with account_box:
        _render_identity()

    user = st.session_state.get("user")
    role = st.session_state.get("role")

    with model_box:
        _group_title(":material/model_training:", "Модель")
        entry = _select_model(user, role)
        model_dir = entry.path if entry is not None else resolve_model_dir()
        try:
            assets = _load_assets(model_dir)
        except ModelLoadError as exc:
            log_event(user, role, "model_error", message=str(exc))
            st.error(str(exc), icon=":material/error:")
            st.stop()
        except Exception as exc:
            logger.exception("Не удалось загрузить модель")
            log_event(user, role, "model_error", message=str(exc))
            st.error(f"Не удалось загрузить модель: {exc}", icon=":material/error:")
            st.stop()
        _render_model_status(entry, assets, user, role)

    with ai_box:
        provider = _select_llm_provider(user, role)

    _render_header(entry)
    # Вкладки по возможностям: мониторинг — если monitor.py уже запускался,
    # серверы — админу и только там, где есть ssh (не в Docker)
    sections = {":material/upload_file: Анализ файла":
                lambda: _render_file_analysis(assets, provider, user, role)}
    monitoring = load_incidents()
    if monitoring is not None:
        sections[":material/monitor_heart: Мониторинг серверов"] = lambda: _render_monitoring(*monitoring, user, role)
    if can_manage_servers(role) and shutil.which("ssh"):
        sections[":material/dns: Серверы"] = lambda: _render_servers(user, role)
    if len(sections) == 1:
        next(iter(sections.values()))()
        return
    for tab, render in zip(st.tabs(list(sections)), sections.values()):
        with tab:
            render()


def _render_monitoring(incidents: pd.DataFrame, updated, user=None, role=None) -> None:
    """Инциденты, которые monitor.py нашёл в захватах с серверов."""
    age = pd.Timestamp.now(tz="UTC") - pd.Timestamp(updated)
    st.caption(
        f"Последняя проверка: {pd.Timestamp(updated).tz_convert(None):%d.%m.%Y %H:%M} UTC "
        f"({_ago(age)}). Проверку запускает pull_captures.ps1 после скачивания; "
        f"вручную — python monitor.py. Время инцидентов — UTC."
    )
    # Серверы — и из списка pull_captures.json (у них может не быть инцидентов)
    try:
        configured = {e.name: e for e in map(srv.entry_from_config, srv.load_config())}
    except (OSError, ValueError):
        configured = {}
    names = sorted(set(configured) | (set(incidents["server"]) if len(incidents) else set()))
    server = "Все"
    if len(names) > 1:
        server = st.segmented_control(
            "Сервер", ["Все", *names], default="Все", required=True,
            key="monitor_server", label_visibility="collapsed",
        )
    elif names:
        server = names[0]
    if server != "Все":
        incidents = incidents[incidents["server"] == server]

    stamps = pd.to_datetime(incidents["window_start"])
    day_ago = pd.Timestamp.now(tz="UTC").tz_convert(None) - pd.Timedelta(days=1)
    cols = st.columns(4, gap="small")
    cols[0].metric("За сутки", f"{int((stamps >= day_ago).sum()):,}", border=True)
    cols[1].metric(f"За {RETAIN_DAYS} дней", f"{len(incidents):,}", border=True)
    cols[2].metric("Адресов", f"{incidents['peer'].nunique():,}", border=True,
                   help="Внешние адреса, от которых были инциденты")
    pending = int((~incidents["notified"].astype(bool)).sum())
    cols[3].metric("Не отправлено", f"{pending:,}", border=True,
                   help="Инциденты, о которых ещё не ушло сообщение в Telegram: "
                        "не настроен бот или не было сети. Уйдут при следующей проверке.")
    if server == "Все":
        _render_servers_comparison(names, configured, incidents)
    else:
        _render_server_stats(server, configured.get(server), incidents, user, role)

    if not len(incidents):
        st.success(f"За {RETAIN_DAYS} дней инцидентов нет.")
        return

    st.markdown("##### Инциденты по дням")
    st.altair_chart(daily_chart(daily_counts(incidents), _theme()), width="stretch")
    view = incidents.sort_values("window_start", ascending=False).rename(columns={
        "server": "Сервер", "window_start": "Час", "peer": "Адрес", "flows": "Потоков",
        "services": "Порты сервера", "max_proba": "Макс. вероятность", "first_seen": "Первый",
        "last_seen": "Последний", "reason": "Почему инцидент", "notified": "Отправлено",
    })
    st.dataframe(
        view, hide_index=True,
        column_config={
            "Час": st.column_config.DatetimeColumn(format="DD.MM HH:mm"),
            "Первый": st.column_config.DatetimeColumn(format="HH:mm:ss"),
            "Последний": st.column_config.DatetimeColumn(format="HH:mm:ss"),
            "Макс. вероятность": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="%.2f"),
            "Отправлено": st.column_config.CheckboxColumn(),
        },
    )


def _day_ago() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_convert(None) - pd.Timedelta(days=1)


def _share(value) -> str:
    return "—" if value is None else f"{value:.2%}"


def _render_servers_comparison(names, configured, incidents) -> None:
    """«Все»: серверы рядом — трафик, доля тревог и инциденты за сутки."""
    directory = mon.state_dir()
    rows = []
    for name in names:
        day = mon.traffic_summary(mon.server_traffic(directory, name), _day_ago())
        own = incidents[incidents["server"] == name]
        entry = configured.get(name)
        downloaded = mon.last_download(entry.local) if entry and entry.local else None
        rows.append({
            "Сервер": name,
            "Потоков за сутки": day["flows"],
            "Доля тревог": day["share"],
            "Инцидентов за сутки": int((pd.to_datetime(own["window_start"]) >= _day_ago()).sum()),
            f"За {RETAIN_DAYS} дней": len(own),
            "Скачано": pd.Timestamp(downloaded).tz_convert(None) if downloaded else None,
        })
    if not rows:
        return
    st.markdown("##### Серверы за сутки")
    st.dataframe(
        pd.DataFrame(rows), hide_index=True,
        column_config={
            "Доля тревог": st.column_config.NumberColumn(
                format="percent", help="Тревожных потоков от всех за сутки"),
            "Скачано": st.column_config.DatetimeColumn(
                "Последнее скачивание", format="DD.MM HH:mm", help="UTC"),
        },
    )


def _render_server_stats(name, entry, incidents, user, role) -> None:
    """Один сервер: трафик и тревоги по часам, атакующие, порты, состояние сбора."""
    directory = mon.state_dir()
    traffic = mon.server_traffic(directory, name)
    day = mon.traffic_summary(traffic, _day_ago())
    downloaded = mon.last_download(entry.local) if entry and entry.local else None

    st.markdown(f"##### {name}: трафик за сутки")
    cols = st.columns(3, gap="small")
    cols[0].metric("Потоков", f"{day['flows']:,}", border=True)
    cols[1].metric("Тревожных", f"{day['alerts']:,}", border=True,
                   help=f"Доля от всех потоков: {_share(day['share'])}")
    cols[2].metric("Скачано", _ago(pd.Timestamp.now(tz="UTC") - pd.Timestamp(downloaded))
                   if downloaded else "—", border=True,
                   help="Когда pull_captures.ps1 последний раз скачал файл захвата")
    if len(traffic):
        left, right = st.columns(2)
        left.altair_chart(hourly_chart(traffic, "Потоков", "blue", _theme()), width="stretch")
        right.altair_chart(hourly_chart(traffic, "Тревог", "orange", _theme()), width="stretch")
        if not traffic["Потоков"].any():
            st.caption("Все потоки по часам monitor.py считает с этой версии: "
                       "график заполнится после следующей проверки.")
    else:
        st.caption("monitor.py ещё не проверял захваты этого сервера.")

    attackers = mon.top_attackers(incidents)
    ports = mon.attacked_ports(directory, name, mon.load_state(directory)["local_ip"].get(name))
    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Атакующие адреса")
        if len(attackers):
            st.dataframe(attackers, hide_index=True, column_config={
                "Последний раз": st.column_config.DatetimeColumn(format="DD.MM HH:mm")})
        else:
            st.caption("Инцидентов нет.")
    with right:
        st.markdown("##### Порты под ударом")
        if len(ports):
            st.altair_chart(top_chart(ports, "Порт", _theme()), width="stretch")
        else:
            st.caption("Тревог нет.")

    if entry is not None and can_manage_servers(role) and shutil.which("ssh"):
        _render_collection_stats(entry, user, role)


def _render_collection_stats(entry, user, role) -> None:
    """Состояние сбора с самого сервера по SSH — по кнопке: опрос занимает секунды."""
    head, button = st.columns([3, 1], vertical_alignment="center")
    head.markdown("##### Сбор на сервере")
    if button.button("Обновить", key=f"collection_{entry.name}", icon=":material/refresh:"):
        try:
            with st.spinner(f"{entry.host}: опрос…"):
                stats = srv.collection_stats(entry)
            st.session_state[f"collection_{entry.name}_stats"] = (stats, pd.Timestamp.now())
            log_event(user, role, "server_collection_stats", server=entry.name, ok=True)
        except srv.ServerError as exc:
            log_event(user, role, "server_collection_stats", server=entry.name, ok=False)
            _show_server_error(str(exc))
            return
    cached = st.session_state.get(f"collection_{entry.name}_stats")
    if cached is None:
        st.caption("Свободное место, объём записи и сколько живёт файл в кольце — "
                   "нажмите «Обновить», чтобы спросить сервер.")
        return
    stats, checked = cached
    label, color, icon = SERVER_STATES[stats["state"]]
    st.badge(f"{label}, проверено {checked:%H:%M}", color=color, icon=icon)
    gb = lambda mb: "—" if mb is None else f"{mb / 1024:.1f} ГБ"  # noqa: E731
    cols = st.columns(4, gap="small")
    cols[0].metric("Свободно на диске", gb(stats["free_mb"]), border=True)
    cols[1].metric("Записано", gb(stats["used_mb"]), border=True,
                   help=f"Кольцо: {gb(stats['ring_mb'])}, файлов {stats['files'] or 0}")
    cols[2].metric("Темп записи", "—" if stats["rate_mb_h"] is None else f"{stats['rate_mb_h']:.0f} МБ/ч",
                   border=True)
    cols[3].metric("Файл живёт", "—" if stats["ring_hours"] is None else f"{stats['ring_hours']:.1f} ч",
                   border=True, help="Через сколько часов кольцо перезапишет файл")
    if stats["started"] is not None:
        st.caption(f"Запись идёт с {stats['started']:%d.%m.%Y %H:%M} UTC.")
    if stats["ring_hours"] is not None and stats["ring_hours"] < RING_WARN_HOURS:
        st.warning(
            f"Файл перезаписывается через {stats['ring_hours']:.1f} ч, а pull_captures.ps1 "
            "по умолчанию забирает раз в 2 ч: часть захвата может пропасть. Запускайте "
            "скачивание чаще (-EveryHours 1) или перезапустите сбор с большим кольцом.",
            icon=":material/warning:",
        )


RING_WARN_HOURS = 4  # вдвое больше интервала скачивания по умолчанию


def _ago(delta: pd.Timedelta) -> str:
    minutes = int(delta.total_seconds() // 60)
    if minutes < 60:
        return f"{minutes} мин назад"
    if minutes < 48 * 60:
        return f"{minutes // 60} ч назад"
    return f"{minutes // 1440} дн. назад"


def _render_servers(user, role) -> None:
    """Серверы сбора: список из pull_captures.json и добавление нового по SSH."""
    try:
        configured = [srv.entry_from_config(item) for item in srv.load_config()]
    except (OSError, ValueError) as exc:
        st.error(f"{srv.config_path()} не прочитан: {exc}")
        return
    pending = st.session_state.get("new_server")
    st.caption(
        "Серверы из pull_captures.json: их забирает pull_captures.ps1 и проверяет monitor.py.",
        help=f"Файл: {srv.config_path()}",
    )
    # Кнопки по ширине текста и с переносом: в колонках они обрезались до «Пров…»
    toolbar = st.container(horizontal=True, horizontal_alignment="right")
    if configured and toolbar.button("Проверить все", icon=":material/refresh:"):
        with st.spinner("Опрос серверов…"):
            states = srv.check_all(configured)
        checked = pd.Timestamp.now()
        for name, (state, output) in states.items():
            st.session_state[f"server_state_{name}"] = (state, checked)
            st.session_state[f"server_out_{name}"] = (
                "error" if state == srv.UNREACHABLE else "ok", output)
        log_event(user, role, "server_check_all", servers=len(states))
    add_label = f"Продолжить: {pending['entry'].name}" if pending else "Добавить сервер"
    if toolbar.button(add_label, type="primary", icon=":material/add:"):
        _add_server_dialog(configured, user, role)

    if not configured:
        st.info("Серверов пока нет. Нажмите «Добавить сервер»: понадобятся адрес и пароль root.")
    for entry in configured:
        _render_server_card(entry, user, role)


# Состояние записи: подпись, цвет метки, значок
SERVER_STATES = {
    srv.RUNNING: ("идёт запись", "green", ":material/radio_button_checked:"),
    srv.STOPPED: ("запись не идёт", "orange", ":material/pause_circle:"),
    srv.EMPTY: ("сбор не запускался", "gray", ":material/radio_button_unchecked:"),
    srv.UNREACHABLE: ("нет связи", "red", ":material/link_off:"),
}


def _server_action(entry, user, role, action: str, run) -> None:
    """Действие по SSH: журнал, вывод под карточкой и состояние записи."""
    try:
        with st.spinner(f"{entry.host}: выполняется…"):
            output = run()
    except srv.ServerError as exc:
        log_event(user, role, f"server_{action}", server=entry.name, host=entry.host, ok=False)
        st.session_state[f"server_out_{entry.name}"] = ("error", str(exc))
        st.session_state[f"server_state_{entry.name}"] = (srv.UNREACHABLE, pd.Timestamp.now())
        return
    log_event(user, role, f"server_{action}", server=entry.name, host=entry.host, ok=True)
    st.session_state[f"server_out_{entry.name}"] = ("ok", output or "Готово.")
    st.session_state[f"server_state_{entry.name}"] = (srv.collector_state(output), pd.Timestamp.now())


def _render_server_card(entry, user, role) -> None:
    with st.container(border=True):
        info, actions = st.columns([3, 1], vertical_alignment="center")
        buttons = actions.container(horizontal=True, horizontal_alignment="right")
        state = st.session_state.get(f"server_state_{entry.name}")
        with info:
            st.markdown(f"**{entry.name}** :gray[{entry.target}]")
            if state is None:
                st.badge("не проверялся", color="gray", icon=":material/help:",
                         help="Нажмите «Статус» или «Проверить все»")
            else:
                label, color, icon = SERVER_STATES[state[0]]
                st.badge(f"{label}, {state[1]:%H:%M}", color=color, icon=icon)
            st.caption(f"{entry.remote} → {entry.local}")
        if buttons.button("Статус", key=f"server_status_{entry.name}", icon=":material/info:"):
            _server_action(entry, user, role, "status", lambda: srv.collector(entry, "status"))
            st.rerun()  # метка выше уже нарисована со старым состоянием
        with buttons.popover("", icon=":material/more_vert:", help="Остановить или убрать из списка"):
            st.markdown(f"**Остановить сбор на {entry.name}**")
            st.caption("dumpcap остановится, последний кусок журнала SSH выгрузится. "
                       "Запустить снова: уберите сервер из списка и добавьте заново, "
                       "пароль уже не понадобится.")
            if st.button("Остановить сбор", key=f"server_stop_{entry.name}", type="primary",
                         icon=":material/stop_circle:"):
                _server_action(entry, user, role, "stop", lambda: srv.collector(entry, "stop"))
                st.rerun()
            st.divider()
            st.markdown(f"**Убрать {entry.name} из списка**")
            st.caption("Сбор на сервере не останавливается, скачанные файлы остаются.")
            if st.button("Убрать из списка", key=f"server_remove_{entry.name}",
                         icon=":material/delete:"):
                srv.remove_from_config(entry.name)
                log_event(user, role, "server_remove", server=entry.name, host=entry.host)
                st.rerun()
        result = st.session_state.get(f"server_out_{entry.name}")
        if result is not None:
            kind, text = result
            if kind == "error":
                _show_server_error(text)
            else:
                st.code(text, language=None)


def _show_server_error(text: str) -> None:
    """Ошибка SSH. В тексте бывает вывод сервера, поэтому он показывается кодом:
    как Markdown взломанный сервер мог бы вставить картинку-маячок или ссылку."""
    st.error("Команда на сервере не выполнена.", icon=":material/error:")
    st.code(text, language=None, wrap_lines=True)


def _forget_finished_server() -> None:
    """Окно закрыли крестиком на последнем шаге — мастер начнётся заново."""
    pending = st.session_state.get("new_server")
    if pending is not None and pending.get("done") is not None:
        st.session_state.pop("new_server", None)
        st.session_state.pop("new_server_plan", None)


@st.dialog("Добавить сервер", width="large", on_dismiss=_forget_finished_server)
def _add_server_dialog(configured, user, role) -> None:
    """Мастер в окне: подключение, tshark (если его нет), запуск сбора, итог."""
    pending = st.session_state.get("new_server")
    if pending is None:
        _render_server_connect(configured, user, role)
    elif pending.get("done") is not None:
        _render_server_done(pending)
    elif not pending["dumpcap"]:
        _render_tshark_install(pending, user, role)
    else:
        _render_server_start(pending, user, role)


WIZARD_STEPS = ("Подключение", "Установка tshark", "Настройка записи", "Готово")


def _render_wizard_steps(current: int) -> None:
    """Шаги мастера в одну строку: пройденные — с галочкой, текущий выделен.
    Если tshark на сервере уже есть, его шаг просто отмечается пройденным."""
    parts = []
    for number, title in enumerate(WIZARD_STEPS):
        label = f"{number + 1}. {title}"
        if number < current:
            parts.append(f":green[:material/check_circle: {label}]")
        elif number == current:
            parts.append(f"**:blue[:material/radio_button_checked: {label}]**")
        else:
            parts.append(f":gray[:material/radio_button_unchecked: {label}]")
    st.markdown(" &nbsp;→&nbsp; ".join(parts))
    st.progress((current + 1) / len(WIZARD_STEPS))


def _default_local(configured, name: str) -> str:
    """Папка для захватов: рядом с папками уже добавленных серверов."""
    parents = [os.path.dirname(entry.local) for entry in configured if entry.local]
    base = parents[0] if parents else os.path.join(BASE_DIR, "captures")
    return os.path.join(base, name or "serverX")


def _render_server_connect(configured, user, role) -> None:
    _render_wizard_steps(0)
    st.caption(
        "Пароль нужен один раз: приложение входит по нему, записывает ключ этого компьютера "
        "в authorized_keys сервера и дальше работает только по ключу. Пароль нигде не "
        "сохраняется. Нужен root или пользователь с sudo без пароля."
    )
    # Поля не очищаются: при ошибке (неверный пароль, нет связи) всё введённое
    # остаётся — достаточно исправить одно поле и подключиться снова
    with st.form("server_connect", border=False):
        left, right = st.columns(2)
        name = left.text_input("Имя", placeholder="serverC", help="Латиница, цифры, «-» и «_»")
        host = right.text_input("Адрес сервера", placeholder="203.0.113.10")
        user_name = left.text_input("Пользователь", value="root")
        password = right.text_input("Пароль", type="password")
        remote = left.text_input("Папка сбора на сервере", value=srv.DEFAULT_REMOTE_DIR)
        local = right.text_input("Папка на этом компьютере", placeholder=_default_local(configured, ""),
                                 help="Пусто — рядом с папками других серверов")
        submitted = st.form_submit_button("Подключиться", type="primary", icon=":material/link:")
    if not submitted:
        return
    try:
        entry = srv.validate(name, user_name, host, remote, local or _default_local(configured, name.strip()))
        if any(item.name.lower() == entry.name.lower() or item.host == entry.host for item in configured):
            raise srv.ServerError("Сервер с таким именем или адресом уже в списке.")
    except srv.ServerError as exc:
        st.error(str(exc), icon=":material/error:")  # текст проверки, без вывода сервера
        return
    try:
        with st.status(f"Подключение к {entry.target}…", expanded=True) as status:
            key = srv.public_key(create=True)
            if password:
                st.write("Вход по паролю и запись ключа…")
                fingerprint = srv.install_key_with_password(entry, password, key)
                st.write(f"Ключ хоста: `{fingerprint}`")
            else:
                fingerprint = None
                st.write("Пароль не указан — пробую войти по уже добавленному ключу.")
            st.write("Проверка входа по ключу и прав…")
            privileges = srv.check_key_login(entry)
            st.write("Копирование collect_capture.sh…")
            srv.install_collector(entry)
            dumpcap = srv.has_dumpcap(entry)
            status.update(label=f"{entry.target}: подключено ({privileges})", state="complete")
    except srv.ServerError as exc:
        log_event(user, role, "server_connect", host=entry.host, ok=False)
        _show_server_error(str(exc))
        return
    log_event(user, role, "server_connect", server=entry.name, host=entry.host, ok=True,
              by_password=bool(password))
    st.session_state.new_server = {"entry": entry, "fingerprint": fingerprint, "dumpcap": dumpcap}
    st.rerun(scope="fragment")  # окно остаётся открытым: шаг 2


def _render_server_start(pending: dict, user, role) -> None:
    entry = pending["entry"]
    _render_wizard_steps(2)
    with st.container():
        st.success(f"{entry.name} ({entry.target}): подключено, ключ записан.",
                   icon=":material/check_circle:")
        if pending.get("fingerprint"):
            st.caption(f"Ключ хоста при первом входе: {pending['fingerprint']}")
        cols = st.columns(3)
        duration = cols[0].text_input("Сколько писать", value="7d", help="30m, 12h, 7d; 0 — до остановки")
        reserve = cols[1].number_input("Запас места, ГБ", min_value=0, max_value=1000, value=5,
                                       help="Сколько оставить свободным всегда; запись остановится раньше")
        file_mb = cols[2].number_input("Файл кольца, МБ", min_value=10, max_value=2000, value=100,
                                       help="Меньше файл — чаще забирается и быстрее проверяется")
        plan, start, cancel = st.columns([1, 1.6, 1])
        try:
            options = srv.start_options(duration, reserve, file_mb)
        except srv.ServerError as exc:
            st.error(str(exc))
            return
        if plan.button("Рассчитать место", icon=":material/calculate:", width="stretch"):
            try:
                with st.spinner("plan…"):
                    st.session_state.new_server_plan = srv.collector(entry, "plan", *options)
            except srv.ServerError as exc:
                st.session_state.new_server_plan = f"Ошибка: {exc}"
        if start.button("Запустить сбор и добавить", type="primary", icon=":material/play_arrow:",
                        width="stretch"):
            try:
                with st.spinner("Запуск сбора…"):
                    output = srv.collector(entry, "start", *options)
                srv.add_to_config(entry)
            except srv.ServerError as exc:
                log_event(user, role, "server_add", server=entry.name, host=entry.host, ok=False)
                _show_server_error(str(exc))
                return
            log_event(user, role, "server_add", server=entry.name, host=entry.host, ok=True,
                      duration=duration)
            st.session_state.pop("new_server_plan", None)
            st.session_state[f"server_out_{entry.name}"] = ("ok", output)
            pending["done"] = {"duration": duration, "reserve": reserve, "file_mb": file_mb}
            st.rerun(scope="fragment")  # окно остаётся: что будет дальше
        if cancel.button("Отмена", width="stretch"):
            st.session_state.pop("new_server", None)
            st.session_state.pop("new_server_plan", None)
            st.rerun()
        if st.session_state.get("new_server_plan"):
            st.code(st.session_state.new_server_plan, language=None)
        with st.expander("Как это будет работать", icon=":material/help:"):
            st.markdown(_how_it_works(entry, duration, reserve, file_mb))
            _render_pull_help(entry)


# Ручная установка dumpcap: пакет tshark (wireshark-cli) под разные системы
TSHARK_MANUAL = """\
# Debian / Ubuntu
sudo apt-get update && sudo apt-get install -y tshark

# RHEL / CentOS / Rocky / Alma / Fedora
sudo dnf install -y wireshark-cli

# openSUSE
sudo zypper install -y wireshark

# Arch
sudo pacman -S --noconfirm wireshark-cli

# Alpine
apk add tshark

# Проверка: должен вывести путь и версию
command -v dumpcap && dumpcap --version | head -n 1"""


def _render_tshark_install(pending: dict, user, role) -> None:
    """Шаг tshark: поставить отсюда по SSH или самому — и проверить установку."""
    entry = pending["entry"]
    _render_wizard_steps(1)
    st.success(f"{entry.name} ({entry.target}): подключено, ключ записан.",
               icon=":material/check_circle:")
    st.warning("На сервере нет dumpcap (пакет tshark) — без него запись трафика не запустится.",
               icon=":material/warning:")
    auto, manual = st.tabs([":material/download: Установить отсюда",
                            ":material/terminal: Установлю сам"])
    with auto:
        st.caption("Приложение выполнит на сервере `apt-get install tshark` (Debian и Ubuntu). "
                   "На других системах — вкладка «Установлю сам».")
        if st.button("Установить tshark (apt-get)", icon=":material/download:", type="primary"):
            try:
                with st.spinner("apt-get install tshark… это может занять пару минут"):
                    srv.install_dumpcap(entry)
                    pending["dumpcap"] = srv.has_dumpcap(entry)
                log_event(user, role, "server_install_tshark", server=entry.name, ok=True)
            except srv.ServerError as exc:
                log_event(user, role, "server_install_tshark", server=entry.name, ok=False)
                _show_server_error(str(exc))
                st.caption("Не получилось — поставьте вручную: вкладка «Установлю сам».")
            else:
                st.rerun(scope="fragment")
    with manual:
        st.markdown("1. Войдите на сервер:")
        st.code(f"ssh {entry.target}", language="bash")
        st.markdown("2. Выполните команду для своей системы (от root — без `sudo`):")
        st.code(TSHARK_MANUAL, language="bash")
        st.caption("Если установщик спросит, разрешать ли захват обычным пользователям, "
                   "ответ не важен: сбор запускается от root или через sudo.")
        st.markdown("3. Вернитесь сюда и проверьте:")
        if st.button("Проверить установку", icon=":material/fact_check:", type="primary"):
            try:
                with st.spinner("Ищу dumpcap на сервере…"):
                    found = srv.has_dumpcap(entry)
            except srv.ServerError as exc:
                _show_server_error(str(exc))
            else:
                log_event(user, role, "server_check_tshark", server=entry.name, found=found)
                if found:
                    pending["dumpcap"] = True
                    st.rerun(scope="fragment")  # дальше — настройка записи
                st.warning("dumpcap пока не найден. Проверьте вывод последней команды "
                           "из списка выше и нажмите «Проверить установку» ещё раз.",
                           icon=":material/search_off:")
    if st.button("Отмена", key="server_cancel_tshark"):
        st.session_state.pop("new_server", None)
        st.rerun()


def _how_it_works(entry, duration: str, reserve, file_mb) -> str:
    """Что происходит после запуска сбора: сервер -> этот компьютер -> мониторинг."""
    span = "до остановки" if duration == "0" else f"{duration}, затем запись остановится сама"
    return f"""\
**На сервере {entry.name}**
- `dumpcap` пишет заголовки пакетов (без содержимого) кольцом файлов по {file_mb} МБ
  в `{entry.remote}`. Когда кольцо заполнено, самый старый файл перезаписывается;
  {reserve} ГБ свободного места не трогаются никогда — при нехватке запись остановится.
- Сколько писать: {span}. Сбор идёт в фоне и не зависит от этого приложения и SSH.
- Раз в час рядом сохраняется журнал SSH (`ssh_*.log`) — по нему размечается брутфорс.

**На этом компьютере**
- `pull_captures.ps1` по ключу (без пароля) скачивает новые файлы в
  `{entry.local}`. Файл, который ещё пишется, пропускается и скачается в следующий раз.
- Сразу после скачивания `monitor.py` проверяет новые захваты моделью. Новые
  инциденты появляются во вкладке «Мониторинг серверов» и уходят в Telegram,
  если бот настроен в `.env`.

Сами по себе файлы на компьютер не приходят: скачивание нужно запускать —
вручную или по расписанию, как описано ниже. Состояние записи — кнопка «Статус»
на карточке сервера; остановить запись — меню ⋮ → «Остановить сбор»."""


# Скрипты PowerShell Windows по умолчанию не запускает: Bypass — разрешение
# только на этот запуск, системные настройки не меняются
PULL_COMMAND = "powershell -ExecutionPolicy Bypass -File .\\pull_captures.ps1"


def _render_pull_help(entry) -> None:
    """Пошагово: как запускать pull_captures.ps1 вручную и по расписанию."""
    st.subheader("Как скачивать захваты на этот компьютер", divider="gray")
    st.markdown(
        "**Шаг 1. Откройте PowerShell в папке проекта.** Нажмите Win+X → «Терминал» "
        "(или «Windows PowerShell») и перейдите в папку проекта:"
    )
    st.code(f'cd "{BASE_DIR}"', language="powershell")
    st.markdown(
        "**Шаг 2. Проверьте без скачивания.** Скрипт подключится к серверам из списка "
        "и покажет, какие файлы скачал бы. Ошибка входа здесь значит, что ключ "
        "не принят — проверьте сервер кнопкой «Статус»."
    )
    st.code(f"{PULL_COMMAND} -DryRun", language="powershell")
    st.caption(
        "`-ExecutionPolicy Bypass` нужен потому, что Windows по умолчанию запрещает "
        "запуск скриптов PowerShell; это разрешение только на один запуск, настройки "
        "системы не меняются."
    )
    st.markdown("**Шаг 3. Выберите, как запускать скачивание.**")
    manual, scheduled = st.tabs([":material/touch_app: Вручную",
                                 ":material/schedule: По расписанию"])
    with manual:
        st.markdown(
            f"Запускайте, когда нужны свежие данные. Новые файлы появятся в "
            f"`{entry.local}`, затем их проверит `monitor.py`, и инциденты будут во "
            "вкладке «Мониторинг серверов»."
        )
        st.code(PULL_COMMAND, language="powershell")
        st.caption("Минус ручного запуска: если долго не запускать, кольцо на сервере "
                   "перезапишет старые файлы, и они пропадут, не дойдя до компьютера.")
    with scheduled:
        st.markdown(
            "Одна команда ставит задачу `cic-pull-captures` в Планировщик Windows: "
            "она сама скачивает и проверяет новые файлы каждые N часов (здесь — 2), "
            "первый раз — через минуту. Если компьютер спал, задача выполнится "
            "при пробуждении; выключенный компьютер ничего не скачивает."
        )
        st.code(f"{PULL_COMMAND} -Install -EveryHours 2", language="powershell")
        st.markdown(
            "Проверить: Пуск → «Планировщик заданий» → «Библиотека планировщика "
            "заданий» → `cic-pull-captures`. Что происходило при запусках — в файле "
            "`pull.log` в папке проекта. Убрать задачу:"
        )
        st.code(f"{PULL_COMMAND} -Uninstall", language="powershell")
    st.markdown(
        "**Как часто запускать.** Чаще, чем заполняется кольцо на сервере: когда оно "
        "полное, самый старый файл перезаписывается — если его ещё не скачали, он "
        "потерян. Сколько файл живёт в кольце, зависит от трафика сервера. Узнать: "
        "примерно через час записи откройте «Мониторинг серверов» → этот сервер → "
        "«Сбор на сервере» → «Обновить», показатель «Файл живёт». Интервал "
        "`-EveryHours` берите хотя бы вдвое меньше; если не уверены — 2 часа "
        "подходят большинству серверов. Поменять интервал — снова `-Install` "
        "с другим числом."
    )


def _render_server_done(pending: dict) -> None:
    """Итог: сервер добавлен, сбор идёт — и как всё будет работать дальше."""
    entry, done = pending["entry"], pending["done"]
    _render_wizard_steps(3)
    st.success(f"{entry.name}: сбор запущен, сервер добавлен в список.",
               icon=":material/check_circle:")
    st.markdown(_how_it_works(entry, done["duration"], done["reserve"], done["file_mb"]))
    _render_pull_help(entry)
    if st.button("Готово", type="primary", icon=":material/done:"):
        st.session_state.pop("new_server", None)
        st.rerun()


def _render_file_analysis(assets: Assets, provider, user, role) -> None:
    upload = _handle_upload(user, role)
    if upload is None:
        return
    df, names = upload

    st.caption(f"Файлы: `{names}` — {len(df):,} записей, {len(df.columns)} колонок")
    if st.session_state.flows_parquet is not None:
        st.download_button(
            "Скачать потоки (Parquet)",
            st.session_state.flows_parquet,
            file_name="flows.parquet",
            mime="application/octet-stream",
            help="Потоки из pcap: в следующий раз можно загрузить их без конвертации",
        )

    if st.button("Анализировать", type="primary", icon=":material/play_arrow:"):
        _run_analysis(df, assets, user, role)

    result = st.session_state.analysis
    if result is not None:
        _render_analysis(
            result, assets.means, assets.stds, assets.brute, st.session_state.get("brute"),
            provider,
        )


if __name__ == "__main__":
    main()
