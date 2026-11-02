"""Пояснения через LLM: Gemini или любой OpenAI-совместимый API.

Провайдеры:
- `gemini` — ключ GEMINI_API_KEY (env или .streamlit/secrets.toml), модель
  GEMINI_MODEL; streamGenerateContent по HTTP (SSE). SDK google-genai не
  используется: его потоковый режим зависал без ответа, хотя тот же запрос
  по HTTP отвечал за секунды.
- `openai` — Chat Completions по HTTP без отдельного SDK: OPENAI_API_KEY,
  OPENAI_BASE_URL (по умолчанию api.openai.com; так же работают OpenRouter,
  DeepSeek, локальные Ollama/LM Studio — им ключ не обязателен), OPENAI_MODEL.

- `claude` — Claude через официальный SDK anthropic (потоковый ответ):
  ANTHROPIC_API_KEY, модель CLAUDE_MODEL. При отказе фильтра безопасности
  запрос сам повторяется на рекомендованной модели (fallbacks: "default").

Кроме них встроены DeepSeek и OpenRouter (DEEPSEEK_API_KEY, OPENROUTER_API_KEY),
а свои варианты — Ollama, LM Studio, другие модели того же провайдера —
описываются в config/llm.json (путь — LLM_CONFIG): id, имя, тип, адрес,
модель и имя переменной с ключом. Сами ключи в файл не пишутся: они берутся
из env/secrets или вводятся в интерфейсе.

LLM_PROVIDER выбирает провайдера по умолчанию; в приложении его можно сменить.
Ключей в исходном коде нет.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_URL = "https://api.openai.com/v1"
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_URL = "https://api.anthropic.com"
CLAUDE_MAX_TOKENS = 16000
# Модели, принимающие серверный fallback при отказе (остальные ответят 400)
CLAUDE_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
# Ожидание каждого чтения: «думающие» модели молчат до первого куска
REQUEST_TIMEOUT_SECONDS = 120
ATTEMPTS = 3

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "llm.json"
)
KINDS = ("gemini", "openai", "anthropic")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,39}$")


class LlmUnavailableError(RuntimeError):
    """LLM использовать нельзя: нет ключа, пакета или ответа модели."""


class LlmRequestError(RuntimeError):
    """Ошибка запроса к провайдеру; status — HTTP-код, если он известен."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


# Повторять имеет смысл только временные ошибки: сеть, 408/409/429, 5xx.
# Неверный ключ или модель (400/401/403/404) повтор не исправит, а у платных
# API лишние запросы ещё и тарифицируются.
RETRY_STATUSES = {408, 409, 429}


def _setting(name: str) -> str:
    """Значение из env, иначе из .streamlit/secrets.toml (NAME, name, [section] key)."""
    value = (os.environ.get(name) or "").strip()
    return value or _secret(name)


def _secret(name: str) -> str:
    try:
        import streamlit as st

        secrets = st.secrets
        for key in (name, name.lower()):
            if key in secrets:
                return str(secrets[key]).strip()
        section, _, key = name.lower().partition("_")
        if section in secrets and key in secrets[section]:
            return str(secrets[section][key]).strip()
    except Exception:
        # Streamlit не запущен (CLI) или secrets.toml отсутствует
        pass

    return ""


@dataclass(frozen=True)
class Provider:
    """Один вариант в списке LLM: kind — протокол (gemini, openai-совместимый, anthropic)."""

    id: str
    name: str
    kind: str
    model: str
    base_url: str
    key_env: str = ""
    key_required: bool = True

    @property
    def label(self) -> str:
        return f"{self.name} · {self.model}"


def _builtin_providers() -> list[Provider]:
    custom_openai = _setting("OPENAI_BASE_URL")
    return [
        Provider("gemini", "Gemini", "gemini", _setting("GEMINI_MODEL") or DEFAULT_MODEL,
                 GEMINI_URL, "GEMINI_API_KEY"),
        Provider("claude", "Claude", "anthropic", _setting("CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL,
                 CLAUDE_URL, "ANTHROPIC_API_KEY"),
        # Свой OPENAI_BASE_URL (Ollama, LM Studio) работает и без ключа
        Provider("openai", "OpenAI-совместимый" if custom_openai else "ChatGPT", "openai",
                 _setting("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
                 custom_openai or DEFAULT_OPENAI_URL, "OPENAI_API_KEY",
                 key_required=not custom_openai),
        Provider("deepseek", "DeepSeek", "openai", _setting("DEEPSEEK_MODEL") or "deepseek-chat",
                 "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
        Provider("openrouter", "OpenRouter", "openai",
                 _setting("OPENROUTER_MODEL") or "openrouter/auto",
                 "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    ]


def _config_providers() -> tuple[list[Provider], list[str]]:
    """Варианты из config/llm.json и ошибки в нём (файла может не быть)."""
    path = _setting("LLM_CONFIG") or DEFAULT_CONFIG
    if not os.path.isfile(path):
        return [], []
    try:
        with open(path, encoding="utf-8") as handle:
            entries = json.load(handle).get("providers", [])
    except (OSError, ValueError, AttributeError) as exc:
        return [], [f"{os.path.basename(path)} не прочитан: {exc}"]

    result, errors = [], []
    for number, entry in enumerate(entries if isinstance(entries, list) else [], start=1):
        if not isinstance(entry, dict):
            errors.append(f"запись {number}: ожидается объект")
            continue
        provider_id = str(entry.get("id", "")).strip().lower()
        kind = str(entry.get("type", "openai")).strip().lower()
        default_url = {"gemini": GEMINI_URL, "anthropic": CLAUDE_URL}.get(kind, "")
        base_url = str(entry.get("base_url") or default_url).strip()
        model = str(entry.get("model", "")).strip()
        key_env = str(entry.get("api_key_env", "")).strip()
        problem = (
            "id из латиницы, цифр, «-_.»" if not ID_RE.match(provider_id)
            else f"type — {' или '.join(KINDS)}" if kind not in KINDS
            else "base_url должен начинаться с http:// или https://"
            if not base_url.startswith(("http://", "https://"))
            else "не задан model" if not model
            else None
        )
        if problem:
            errors.append(f"запись {number} ({provider_id or '?'}): {problem}")
            continue
        result.append(Provider(
            provider_id, str(entry.get("name") or provider_id), kind, model, base_url.rstrip("/"),
            key_env, bool(entry.get("key_required", bool(key_env))),
        ))
    return result, errors


def providers() -> dict[str, Provider]:
    """Все варианты LLM по id: встроенные, затем из config/llm.json (с тем же id — замена)."""
    registry = {p.id: p for p in _builtin_providers()}
    registry.update({p.id: p for p in _config_providers()[0]})
    return registry


def config_errors() -> list[str]:
    return _config_providers()[1]


def _provider_key(provider: Provider, keys: dict | None = None) -> str:
    return (keys or {}).get(provider.id) or (_setting(provider.key_env) if provider.key_env else "")


def configured_providers(keys: dict | None = None) -> list[str]:
    """id вариантов, которые можно вызвать: есть ключ (env/secrets/интерфейс) или он не нужен.

    keys — ключи, введённые в интерфейсе: {id: ключ}.
    """
    return [
        p.id for p in providers().values()
        if not p.key_required or _provider_key(p, keys)
    ]


def default_provider(keys: dict | None = None) -> str | None:
    available = configured_providers(keys)
    wanted = _setting("LLM_PROVIDER").lower()
    if wanted in available:
        return wanted
    return available[0] if available else None


def is_llm_configured() -> bool:
    """Настроен ли хотя бы один провайдер (без проверки сети)."""
    return bool(configured_providers())


def provider_label(provider_id: str) -> str:
    provider = providers().get(provider_id)
    return provider.label if provider else provider_id


def _build_prompt(stats_text: str) -> str:
    return f"""Ниже — признаки сетевого потока (CICFlowMeter), сильнее всего отличающиеся \
от нормального трафика: значение, норма и отклонение в стандартных отклонениях. \
Время — в микросекундах, объёмы — в байтах.

Объясни аномалию для аналитика в 2–3 предложениях по-русски:
1. Говори масштабом, а не статистикой: «в 30 раз дольше обычного», \
«экстремально много пакетов» — без слов «сигма», «отклонение», «дисперсия».
2. Связанные признаки объединяй (несколько IAT — «паузы между пакетами»).
3. Последним предложением назови наиболее вероятную угрозу или скажи, что это \
похоже на безобидный редкий трафик.

Данные:
{stats_text}
"""


def _error_message(body: str) -> str:
    """Текст ошибки из JSON провайдера ({"error": {"message": …}}), иначе начало тела."""
    try:
        error = json.loads(body).get("error")
        message = error.get("message") if isinstance(error, dict) else error
        if message:
            return str(message)[:500]
    except (ValueError, AttributeError):
        pass
    return body[:500]


def _sse_events(url: str, payload: dict, headers: dict):
    """POST с JSON и разбор ответа text/event-stream: JSON каждого события data:."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream", **headers},
        method="POST",
    )
    try:
        response = urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise LlmRequestError(f"HTTP {exc.code}: {_error_message(body)}", status=exc.code) from exc

    with response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("error"):
                raise LlmRequestError(str(event["error"])[:500])
            yield event


def _gemini_stream(system: str | None, messages: list[dict], provider: Provider, api_key: str):
    payload = {
        "contents": [
            {"role": "model" if m["role"] == "assistant" else "user",
             "parts": [{"text": m["content"]}]}
            for m in messages
        ],
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    # Ключ — в заголовке, а не в URL: адреса запросов попадают в логи прокси
    url = f"{provider.base_url}/models/{provider.model}:streamGenerateContent?alt=sse"
    for event in _sse_events(url, payload, {"x-goog-api-key": api_key}):
        for candidate in event.get("candidates") or []:
            for part in (candidate.get("content") or {}).get("parts") or []:
                # thought=True — рассуждения «думающей» модели, их не показываем
                if part.get("text") and not part.get("thought"):
                    yield part["text"]


def _openai_stream(system: str | None, messages: list[dict], provider: Provider, api_key: str):
    base_url = provider.base_url.rstrip("/")
    payload = {
        "model": provider.model,
        "messages": ([{"role": "system", "content": system}] if system else [])
        + [{"role": m["role"], "content": m["content"]} for m in messages],
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    for event in _sse_events(f"{base_url}/chat/completions", payload, headers):
        for choice in event.get("choices") or []:
            text = (choice.get("delta") or {}).get("content")
            if text:
                yield text


def _anthropic_stream(system: str | None, messages: list[dict], provider: Provider, api_key: str):
    try:
        import anthropic
    except ImportError as exc:
        raise LlmUnavailableError(
            "Не установлен пакет anthropic. Выполните: pip install -r requirements.txt"
        ) from exc

    # Повторы делает stream_chat (только до первого куска), поэтому у SDK их нет
    client = anthropic.Anthropic(
        api_key=api_key, base_url=provider.base_url,
        timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0,
    )
    params = {
        "model": provider.model,
        "max_tokens": CLAUDE_MAX_TOKENS,
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
    }
    if system:
        params["system"] = system
    if provider.model in CLAUDE_FALLBACK_MODELS:
        # Отказ фильтра безопасности — повтор на рекомендованной модели внутри того же запроса
        params.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    try:
        with client.beta.messages.stream(**params) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
    except anthropic.APIStatusError as exc:
        raise LlmRequestError(f"HTTP {exc.status_code}: {exc.message}", status=exc.status_code) from exc
    if final.stop_reason == "refusal":
        raise LlmUnavailableError(f"{provider.name}: модель отказалась отвечать (фильтр безопасности).")


STREAMS = {"gemini": _gemini_stream, "openai": _openai_stream, "anthropic": _anthropic_stream}


def stream_chat(
    provider: str, messages: list[dict], system: str | None = None, api_key: str | None = None
):
    """Ответ провайдера по кускам; messages — [{"role": "user"|"assistant", "content"}].

    api_key — ключ из интерфейса (только на эту сессию), иначе из env/secrets.

    Повторяет попытку, только пока не пришёл первый кусок: оборванный на
    середине ответ повторять нельзя — текст уже показан.
    """
    entry = providers().get(provider)
    if entry is None:
        raise LlmUnavailableError(f"Неизвестный провайдер LLM: {provider}")
    key = _provider_key(entry, {provider: api_key} if api_key else None)
    if entry.key_required and not key:
        hint = f"задайте {entry.key_env} или " if entry.key_env else ""
        raise LlmUnavailableError(f"{entry.name} не настроен: {hint}введите ключ в интерфейсе.")
    generate = STREAMS[entry.kind]
    name = entry.name
    last_error: Exception | None = None

    for attempt in range(1, ATTEMPTS + 1):
        started = False
        try:
            for text in generate(system, messages, entry, key):
                started = True
                yield text
        except LlmUnavailableError:
            raise
        except Exception as exc:  # сеть, квоты, неверный ключ, ошибки SDK
            if started:
                raise LlmUnavailableError(f"{name}: ответ оборвался: {exc}") from exc
            last_error = exc
            if not _retryable(exc):
                break
            if attempt < ATTEMPTS:
                time.sleep(2 ** (attempt - 1))
            continue
        if not started:
            # Пустой ответ (например, заблокирован фильтром) сохранять в чат
            # нельзя: следующий запрос с пустой репликой провайдер отвергнет
            raise LlmUnavailableError(
                f"{name}: модель не вернула текст (возможно, ответ заблокирован фильтром)."
            )
        return

    raise LlmUnavailableError(f"{name} недоступен: {last_error}") from last_error


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status", None)
    if not isinstance(status, int) or not 400 <= status < 500:
        return True
    return status in RETRY_STATUSES


def explain_with_llm(stats_text: str, provider: str | None = None) -> str:
    """Краткое пояснение одной строкой (для CLI); провайдер — по умолчанию."""
    provider = provider or default_provider()
    if provider is None:
        raise LlmUnavailableError("Не настроен ни один провайдер LLM.")
    return "".join(
        stream_chat(provider, [{"role": "user", "content": _build_prompt(stats_text)}])
    ).strip()
