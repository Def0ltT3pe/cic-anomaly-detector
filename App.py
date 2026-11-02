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
from modules.charts import deviation_chart, timeline_chart, top_chart
from modules.explainer import feature_deviations
from modules.model_catalog import ModelEntry, list_models
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
