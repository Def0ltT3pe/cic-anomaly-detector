"""Streamlit-приложение: загрузка CSV/Parquet и поиск аномалий трафика."""

from __future__ import annotations

import hmac
import logging
import os
import shutil
import sys
import tempfile
import time

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
    first_message,
    flagged_candidates,
)
from modules.llm_explain import (
    PROVIDERS,
    LlmUnavailableError,
    configured_providers,
    default_provider,
    provider_label,
    stream_chat,
)
from modules.model_catalog import ModelEntry, list_models
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
PCAP_EXTENSIONS = (".pcap", ".pcapng", ".cap")
TABLE_EXTENSIONS = (".parquet", ".csv")

st.set_page_config(
    page_title="Детектор аномалий трафика",
    page_icon="🛰️",
    layout="wide",
)


@st.cache_resource(show_spinner="Загружаем модель…", max_entries=4)
def _load_assets(model_dir: str):
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

    return model, scaler, selector, feature_cols, means, stds, calibration, brute, errors


MAX_LOGIN_ATTEMPTS = 5
LOCK_SECONDS = 60


def _app_password() -> str:
    password = (os.environ.get("APP_PASSWORD") or "").strip()
    if password:
        return password
    try:
        return str(st.secrets.get("APP_PASSWORD", "")).strip()
    except Exception:
        return ""


def _register_failed_login(username: str) -> None:
    attempts = int(st.session_state.get("login_attempts", 0)) + 1
    st.session_state.login_attempts = attempts
    log_event(username or "-", "-", "login_failed", attempt=attempts)
    if attempts >= MAX_LOGIN_ATTEMPTS:
        st.session_state.lock_until = time.time() + LOCK_SECONDS
        st.session_state.login_attempts = 0


def _login_screen() -> None:
    """Логин по users.json; fallback — общий APP_PASSWORD; иначе открытый режим."""
    if st.session_state.get("user"):
        return

    try:
        users = load_users()
    except AuthError as exc:
        st.error(f"Ошибка конфигурации пользователей: {exc}")
        st.stop()

    shared = _app_password()
    if not users and not shared:
        st.session_state.user = "anonymous"
        st.session_state.role = "admin"
        st.session_state.auth_mode = "open"
        return

    st.title("Вход")
    lock_until = float(st.session_state.get("lock_until", 0.0))
    if time.time() < lock_until:
        st.error(
            "Слишком много неудачных попыток. Повторите через "
            f"{int(lock_until - time.time()) + 1} с."
        )
        st.stop()

    if users:
        with st.form("login_form"):
            username = st.text_input("Логин")
            password = st.text_input("Пароль", type="password")
            submitted = st.form_submit_button("Войти", type="primary")
        if submitted:
            user = authenticate(username, password)
            if user is not None:
                st.session_state.user = user.username
                st.session_state.role = user.role
                st.session_state.auth_mode = "users"
                st.session_state.login_attempts = 0
                log_event(user.username, user.role, "login", mode="password")
                st.rerun()
            _register_failed_login(username)
            st.error("Неверный логин или пароль")
    else:
        with st.form("login_form"):
            password = st.text_input("Пароль", type="password")
            submitted = st.form_submit_button("Войти", type="primary")
        if submitted:
            if hmac.compare_digest(password.encode("utf-8"), shared.encode("utf-8")):
                st.session_state.user = "shared"
                st.session_state.role = "admin"
                st.session_state.auth_mode = "shared"
                st.session_state.login_attempts = 0
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
        "login_attempts": 0,
        "lock_until": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_audit_panel() -> None:
    with st.sidebar.expander("Журнал аудита"):
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
            st.caption(line)


def _render_identity() -> None:
    user = st.session_state.get("user")
    role = st.session_state.get("role")
    if not user:
        return

    st.sidebar.markdown(f"**{user}** · `{role}`")
    if st.session_state.get("auth_mode") == "open":
        st.sidebar.warning(
            "Аутентификация не настроена. Добавьте пользователя "
            "(`python manage_users.py add admin --role admin`) "
            "или задайте APP_PASSWORD."
        )

    if st.sidebar.button("Выйти", key="logout"):
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
            "login_attempts",
            "lock_until",
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


def _load_uploads(files) -> tuple[pd.DataFrame, bool]:
    """Все загруженные файлы в одну таблицу; True — если были pcap."""
    frames = []
    tshark = None
    for file in files:
        name = (file.name or "").lower()
        if name.endswith(PCAP_EXTENSIONS):
            if tshark is None:
                tshark = find_tshark(os.environ.get("TSHARK") or None)
            frame = _convert_pcap_upload(file, tshark)
        else:
            frame = _read_table(file)
        if frame is None or frame.empty:
            raise ValueError(f"{file.name}: нет ни одной строки данных.")
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if len(df) > MAX_ROWS:
        raise ValueError(f"Слишком много строк: {len(df):,}. Максимум — {MAX_ROWS:,}.")
    return df, tshark is not None


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


def _ask_llm(provider: str, chat: list[dict], question: str, report_for_llm: str, index: int) -> bool:
    """Вопрос в чат по потоку; ответ показывается по мере генерации. True — ответ сохранён."""
    user = st.session_state.get("user")
    role = st.session_state.get("role")
    content = first_message(report_for_llm, question)["content"] if not chat else question
    messages = [{"role": m["role"], "content": m["content"]} for m in chat]
    messages.append({"role": "user", "content": content})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            api_key = st.session_state.llm_keys.get(provider)
            answer = st.write_stream(
                stream_chat(provider, messages, system=SYSTEM_PROMPT, api_key=api_key)
            )
        except LlmUnavailableError as exc:
            st.warning(str(exc))
            log_event(user, role, "llm_ask", row=index, provider=provider, ok=False, error=str(exc))
            return False
        except Exception as exc:  # неожиданная ошибка SDK
            logger.exception("LLM error")
            st.warning(f"Ошибка LLM: {exc}")
            log_event(user, role, "llm_ask", row=index, provider=provider, ok=False, error=str(exc))
            return False
    answer = answer if isinstance(answer, str) else "".join(map(str, answer))
    chat.append({"role": "user", "content": content, "shown": question})
    chat.append({"role": "assistant", "content": answer, "shown": answer, "provider": provider})
    log_event(user, role, "llm_ask", row=index, provider=provider, ok=bool(answer.strip()), error=None)
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
    st.subheader("Разбор потока")
    candidates = flagged_candidates(result, brute_probs, brute_mask)
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

    report_kwargs = dict(
        brute=brute, brute_probs=brute_probs, brute_mask=brute_mask, incidents=incidents
    )
    report = build_flow_report(result, index, means, stds, mask_ips=False, **report_kwargs)
    with st.expander("Факты о потоке", expanded=True):
        st.markdown(report)

    chat = st.session_state.explanations.setdefault(index, [])
    for message in chat:
        with st.chat_message(message["role"]):
            st.markdown(message["shown"])

    if not can_use_llm(st.session_state.get("role")):
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
            if _ask_llm(provider, chat, question, masked, index):
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


def _render_incidents(brute, brute_mask, incidents) -> None:
    """Подозрительные потоки, сведённые в инциденты по внешнему адресу за час."""
    if incidents is None:
        st.caption(
            "Инциденты по адресам не собраны: в файле нет Src/Dst IP и портов "
            "(их добавляет pcap_to_flows.py)."
        )
        return
    st.subheader(f"Инциденты: {brute.display_name}")
    if not len(incidents):
        st.success(
            "Подозрительные потоки есть, но ни один адрес не дал инцидента: "
            "единичные срабатывания, скорее всего ложные."
        )
        return
    view = incidents.drop(columns=["rows"]).rename(
        columns={
            "window_start": "Час",
            "peer": "Адрес",
            "flows": "Потоков",
            "services": "Порты сервера",
            "max_proba": "Макс. p",
            "first_seen": "Первый",
            "last_seen": "Последний",
            "reason": "Почему инцидент",
        }
    )
    st.dataframe(view, hide_index=True)
    st.caption(
        f"{summarize(incidents, brute_mask)}. Инцидент — поток на порт "
        f"{', '.join(map(str, DEFAULT_IMMEDIATE_PORTS))} или ≥{DEFAULT_MIN_FLOWS} "
        "подозрительных потоков с одного адреса за час; единичные срабатывания "
        "на других портах — почти всегда ложные."
    )


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

    cols = st.columns(4 if brute_mask is not None else 3)
    cols[0].metric("Всего записей", f"{result.total:,}")
    cols[1].metric("Аномалии (IF)", f"{result.n_anomalies:,}")
    cols[2].metric("Доля аномалий", f"{result.share:.2%}")
    if brute_mask is not None:
        cols[3].metric(brute.display_name, f"{int(brute_mask.sum()):,}")
        cols[3].caption(
            f"порог {brute.threshold:.3f} · FPR ≈ "
            f"{float(brute.metadata.get('target_fpr', 0)):.2%}"
        )

    if result.info.get("inf_replaced") or result.info.get("nan_filled"):
        st.warning(
            f"В данных найдено inf: {result.info.get('inf_replaced', 0)}, "
            f"пропусков: {result.info.get('nan_filled', 0)}. "
            "Они заменены средними значениями нормы."
        )

    if result.n_anomalies:
        st.subheader("Аномальные записи (IsolationForest)")
        st.dataframe(result.df[result.mask].head(10), hide_index=True)
    else:
        st.success("Аномалий IsolationForest не обнаружено")

    incidents = None
    if brute_mask is not None and brute_mask.any():
        incidents = _compute_incidents(result, brute_probs, brute_mask)
        _render_incidents(brute, brute_mask, incidents)
        st.subheader(f"Подозрения: {brute.display_name} (supervised)")
        flagged = np.flatnonzero(brute_mask)
        top = flagged[np.argsort(-brute_probs[flagged])][:10]
        view = result.df.iloc[top].copy()
        view[f"{brute.display_name} p"] = np.round(brute_probs[top], 3)
        st.dataframe(view, hide_index=True)
        st.caption(
            f"Порог {brute.threshold:.3f}; модель обучена на размеченных потоках "
            f"{'целевой сети' if brute.metadata.get('adaptation') else 'CICIDS'}. "
            f"Совпадений с IF: {int((result.mask & brute_mask).sum()):,}."
        )

    _render_flow_review(
        result, means, stds, brute, brute_probs, brute_mask, incidents, provider
    )


def _reset_analysis() -> None:
    st.session_state.analysis = None
    st.session_state.explanations = {}
    st.session_state.brute = None


def _render_llm_keys(user, role) -> dict:
    """Ключ API из интерфейса: только в памяти этой сессии, не на диске и не в аудите."""
    keys = st.session_state.llm_keys
    with st.sidebar.expander("Ключ API для ИИ", expanded=not configured_providers(keys)):
        with st.form("llm_key_form", clear_on_submit=True):
            name = st.selectbox("Провайдер", list(PROVIDERS), format_func=PROVIDERS.get)
            key = st.text_input("Ключ", type="password")
            saved = st.form_submit_button("Сохранить на эту сессию")
        if saved and key.strip():
            keys[name] = key.strip()
            st.session_state.llm_provider = name
            log_event(user, role, "llm_key_set", provider=name)  # без самого ключа
        for name in keys:
            st.caption(f"{PROVIDERS[name]}: ключ введён")
        if keys and st.button("Забыть ключи", key="llm_keys_clear"):
            keys.clear()
            log_event(user, role, "llm_key_clear")
            st.rerun()
        st.caption(
            "Ключ хранится только в памяти этой сессии и пропадает при выходе. "
            "Адрес OpenAI-совместимого API — OPENAI_BASE_URL (по умолчанию OpenAI)."
        )
    return keys


def _select_llm_provider(user, role) -> str | None:
    """LLM для разбора потоков: по умолчанию LLM_PROVIDER, выбор — на сессию."""
    if not can_use_llm(role):
        st.sidebar.caption("ИИ: недоступен для роли viewer")
        return None
    keys = _render_llm_keys(user, role)
    providers = configured_providers(keys)
    if not providers:
        st.sidebar.caption("ИИ: не настроен — введите ключ выше или задайте GEMINI_API_KEY / OPENAI_API_KEY")
        return None
    current = st.session_state.llm_provider or default_provider(keys)
    provider = st.sidebar.selectbox(
        "LLM для разбора",
        providers,
        index=providers.index(current) if current in providers else 0,
        format_func=provider_label,
    )
    st.session_state.llm_provider = provider
    return provider


def _select_model(user, role) -> ModelEntry | None:
    """Выбор модели в боковой панели; действует только в этой сессии."""
    try:
        entries = list_models()
    except OSError as exc:
        st.sidebar.warning(f"Не удалось прочитать список моделей: {exc}")
        entries = []
    if not entries:
        return None

    paths = [entry.path for entry in entries]
    current = st.session_state.model_dir
    index = paths.index(current) if current in paths else 0
    entry = st.sidebar.selectbox(
        "Модель",
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
    with st.sidebar.expander("О модели"):
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


def main() -> None:
    _init_state()
    _login_screen()
    _render_identity()

    user = st.session_state.get("user")
    role = st.session_state.get("role")

    entry = _select_model(user, role)
    model_dir = entry.path if entry is not None else resolve_model_dir()
    try:
        model, scaler, selector, feature_cols, means, stds, calibration, brute, load_errors = (
            _load_assets(model_dir)
        )
    except ModelLoadError as exc:
        log_event(user, role, "model_error", message=str(exc))
        st.sidebar.error(f"❌ {exc}")
        st.stop()
    except Exception as exc:
        logger.exception("Не удалось загрузить модель")
        log_event(user, role, "model_error", message=str(exc))
        st.sidebar.error(f"❌ Не удалось загрузить модель: {exc}")
        st.stop()

    st.sidebar.success("✅ Модель и статистики загружены")
    if entry is not None:
        _render_model_card(entry)
    for message in load_errors:
        st.sidebar.warning(f"⚠️ {message}")
        log_event(user, role, "model_warning", message=message)
    st.sidebar.caption(f"Признаков: {len(feature_cols)}")
    if calibration is not None:
        st.sidebar.caption(
            f"Порог: score ≤ {calibration.threshold:.4f} "
            f"(целевой FPR ≈ {calibration.target_fpr:.1%})"
        )
    else:
        st.sidebar.caption("Порог: встроенный (contamination модели)")
    if calibration is not None and calibration.source == "target network holdout":
        st.sidebar.caption("Модель адаптирована под целевую сеть (adapt_model.py)")
    st.sidebar.caption(
        (f"{brute.display_name}: supervised" if brute is not None else "Supervised: нет модели")
    )
    provider = _select_llm_provider(user, role)

    st.title("Анализатор аномалий сетевого трафика")
    st.caption(
        "Загрузите захват pcap/pcapng (потоки посчитает tshark) или CSV/Parquet "
        "с теми же признаками, что использовались при обучении."
    )

    uploaded = st.file_uploader(
        "📂 Загрузите файлы (pcap, pcapng, Parquet или CSV)",
        type=list(PCAP_EXTENSIONS + TABLE_EXTENSIONS),
        accept_multiple_files=True,
    )
    if not uploaded:
        st.info("Файл не загружен.")
        return

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
            df, converted = _load_uploads(uploaded)
        except ConversionError as exc:
            log_event(user, role, "upload_error", file=names, error=str(exc))
            st.error(f"Не удалось разобрать захват: {exc} (путь к tshark можно задать в TSHARK)")
            return
        except Exception as exc:
            log_event(user, role, "upload_error", file=names, error=str(exc))
            st.error(f"Не удалось прочитать файл: {exc}")
            return

        st.session_state.file_id = file_id
        st.session_state.df = df
        if converted:
            st.session_state.flows_parquet = df.to_parquet(index=False)
        log_event(
            user,
            role,
            "upload",
            file=names,
            rows=len(df),
            columns=len(df.columns),
            pcap=converted,
        )

    df = st.session_state.df
    if df is None:
        return

    st.caption(f"Файлы: `{names}` — {len(df):,} записей, {len(df.columns)} колонок")
    if st.session_state.flows_parquet is not None:
        st.download_button(
            "Скачать потоки (Parquet)",
            st.session_state.flows_parquet,
            file_name="flows.parquet",
            mime="application/octet-stream",
            help="Потоки из pcap: в следующий раз можно загрузить их без конвертации",
        )

    if st.button("Анализировать", type="primary"):
        try:
            with st.spinner("Ищем аномалии…"):
                result = detect_anomalies(
                    df,
                    model,
                    scaler,
                    selector,
                    feature_cols,
                    fill_values=means,
                    max_rows=MAX_ROWS,
                    calibration=calibration,
                )
        except FeatureValidationError as exc:
            st.error(str(exc))
        except Exception as exc:
            logger.exception("Ошибка детекции")
            st.error(f"Ошибка при анализе: {exc}")
        else:
            st.session_state.analysis = result
            st.session_state.explanations = {}
            brute_payload = _compute_brute(result, brute)
            st.session_state.brute = brute_payload
            log_event(
                user,
                role,
                "analyze",
                rows=result.total,
                anomalies=result.n_anomalies,
                brute=(
                    int(brute_payload[1].sum())
                    if brute_payload is not None
                    else None
                ),
                threshold=result.threshold,
            )

    result = st.session_state.analysis
    if result is not None:
        _render_analysis(
            result, means, stds, brute, st.session_state.get("brute"), provider
        )


if __name__ == "__main__":
    main()
