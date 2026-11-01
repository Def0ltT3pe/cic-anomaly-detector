"""Пояснения через LLM: Gemini или любой OpenAI-совместимый API.

Провайдеры:
- `gemini` — ключ GEMINI_API_KEY (env или .streamlit/secrets.toml), модель
  GEMINI_MODEL. SDK `google-genai`, запасной — legacy `google-generativeai`.
- `openai` — Chat Completions по HTTP без отдельного SDK: OPENAI_API_KEY,
  OPENAI_BASE_URL (по умолчанию api.openai.com; так же работают OpenRouter,
  DeepSeek, локальные Ollama/LM Studio — им ключ не обязателен), OPENAI_MODEL.

LLM_PROVIDER выбирает провайдера по умолчанию; в приложении его можно сменить.
Ключей в исходном коде нет.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_URL = "https://api.openai.com/v1"
REQUEST_TIMEOUT_SECONDS = 60
ATTEMPTS = 3

PROVIDERS = {"gemini": "Gemini", "openai": "OpenAI-совместимый"}

_clients: dict[str, object] = {}


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


def configured_providers(keys: dict | None = None) -> list[str]:
    """Провайдеры, для которых задан ключ или адрес (без проверки сети).

    keys — ключи, введённые в интерфейсе: {"gemini": "...", "openai": "..."}.
    """
    keys = keys or {}
    providers = []
    if keys.get("gemini") or _setting("GEMINI_API_KEY"):
        providers.append("gemini")
    if keys.get("openai") or _setting("OPENAI_API_KEY") or _setting("OPENAI_BASE_URL"):
        providers.append("openai")
    return providers


def default_provider(keys: dict | None = None) -> str | None:
    providers = configured_providers(keys)
    wanted = _setting("LLM_PROVIDER").lower()
    if wanted in providers:
        return wanted
    return providers[0] if providers else None


def is_llm_configured() -> bool:
    """Настроен ли хотя бы один провайдер (без проверки сети)."""
    return bool(configured_providers())


def provider_model(provider: str) -> str:
    if provider == "openai":
        return _setting("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    return _setting("GEMINI_MODEL") or DEFAULT_MODEL


def provider_label(provider: str) -> str:
    return f"{PROVIDERS.get(provider, provider)} · {provider_model(provider)}"


def _backend() -> str:
    """Выбирает установленный SDK: genai (новый) -> generativeai (legacy)."""
    try:
        import google.genai  # noqa: F401

        return "genai"
    except ImportError:
        pass

    try:
        import google.generativeai  # noqa: F401

        return "generativeai"
    except ImportError as exc:
        raise LlmUnavailableError(
            "Не установлен SDK Gemini. Выполните: pip install -r requirements.txt"
        ) from exc


def _build_prompt(stats_text: str) -> str:
    return f"""Проанализируй аномальные метрики сетевого соединения. Напиши краткое объяснение (ровно 2-3 предложения).

Строгие ограничения:
1. Запрещено использовать слова: "сигма", "отклонение", "дисперсия", "статистика".
2. Вместо математических терминов укажи масштаб аномалии понятным языком: во сколько раз превышена норма или используй оценку "критически/экстремально выше нормы".
3. Объединяй связанные метрики (например, если несколько параметров IAT завышены, опиши это как "задержки между пакетами").
4. В конце укажи наиболее вероятную сетевую угрозу.

Данные для анализа:
{stats_text}
"""


def _gemini_stream(system: str | None, messages: list[dict], model_name: str, api_key=None):
    # Клиент кэшируется только для ключа из env/secrets: ключ из интерфейса
    # не должен оставаться в памяти процесса после выхода из сессии
    cache = api_key is None
    api_key = api_key or _setting("GEMINI_API_KEY")
    if not api_key:
        raise LlmUnavailableError(
            "Не задан GEMINI_API_KEY. Добавьте его в переменные окружения "
            "или в .streamlit/secrets.toml — пояснения ИИ будут недоступны."
        )
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages
    ]

    if _backend() == "genai":
        from google import genai

        client = _clients.get(api_key) if cache else None
        if client is None:
            client = genai.Client(
                api_key=api_key, http_options={"timeout": REQUEST_TIMEOUT_SECONDS * 1000}
            )
            if cache:
                _clients[api_key] = client
        config = {"system_instruction": system} if system else None
        response = client.models.generate_content_stream(
            model=model_name, contents=contents, config=config
        )
    else:
        import google.generativeai as legacy

        legacy.configure(api_key=api_key)
        model = legacy.GenerativeModel(model_name, system_instruction=system)
        response = model.generate_content(
            contents, stream=True, request_options={"timeout": REQUEST_TIMEOUT_SECONDS}
        )

    for chunk in response:
        try:
            text = getattr(chunk, "text", None)
        except Exception:  # safety-блокировка части ответа
            text = None
        if text:
            yield str(text)


def _openai_stream(system: str | None, messages: list[dict], model_name: str, api_key=None):
    base_url = (_setting("OPENAI_BASE_URL") or DEFAULT_OPENAI_URL).rstrip("/")
    api_key = api_key or _setting("OPENAI_API_KEY")
    payload = {
        "model": model_name,
        "messages": ([{"role": "system", "content": system}] if system else [])
        + [{"role": m["role"], "content": m["content"]} for m in messages],
        "stream": True,
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        response = urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        raise LlmRequestError(f"HTTP {exc.code}: {body}", status=exc.code) from exc

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
            if event.get("error"):
                raise LlmRequestError(str(event["error"])[:500])
            for choice in event.get("choices") or []:
                text = (choice.get("delta") or {}).get("content")
                if text:
                    yield text


def stream_chat(
    provider: str, messages: list[dict], system: str | None = None, api_key: str | None = None
):
    """Ответ провайдера по кускам; messages — [{"role": "user"|"assistant", "content"}].

    api_key — ключ из интерфейса (только на эту сессию), иначе из env/secrets.

    Повторяет попытку, только пока не пришёл первый кусок: оборванный на
    середине ответ повторять нельзя — текст уже показан.
    """
    if provider not in PROVIDERS:
        raise LlmUnavailableError(f"Неизвестный провайдер LLM: {provider}")
    if provider not in configured_providers({provider: api_key}):
        variable = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY или OPENAI_BASE_URL"
        raise LlmUnavailableError(f"{PROVIDERS[provider]} не настроен: задайте {variable}.")
    generate = _gemini_stream if provider == "gemini" else _openai_stream
    model_name = provider_model(provider)
    last_error: Exception | None = None

    for attempt in range(1, ATTEMPTS + 1):
        started = False
        try:
            for text in generate(system, messages, model_name, api_key):
                started = True
                yield text
        except LlmUnavailableError:
            raise
        except Exception as exc:  # сеть, квоты, неверный ключ, ошибки SDK
            if started:
                raise LlmUnavailableError(f"{PROVIDERS[provider]}: ответ оборвался: {exc}") from exc
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
                f"{PROVIDERS[provider]}: модель не вернула текст (возможно, ответ заблокирован фильтром)."
            )
        return

    raise LlmUnavailableError(f"{PROVIDERS[provider]} недоступен: {last_error}") from last_error


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status", None)
    if not isinstance(status, int):
        status = getattr(exc, "code", None)  # google-genai APIError
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
