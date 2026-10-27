"""Пояснения аномалий через Gemini.

Поддерживаются оба SDK:
- `google-genai` (новый, основной) — ставится из requirements.txt;
- `google-generativeai` (legacy) — запасной вариант, если новый не установлен.

Ключ API берётся из переменной окружения GEMINI_API_KEY или из
.streamlit/secrets.toml. В исходном коде ключа нет.
"""

from __future__ import annotations

import os
import time

DEFAULT_MODEL = "gemini-3.8-flash"
REQUEST_TIMEOUT_SECONDS = 60
ATTEMPTS = 3

_clients: dict[str, object] = {}


class LlmUnavailableError(RuntimeError):
    """LLM использовать нельзя: нет ключа, пакета или ответа модели."""


def _read_api_key() -> str:
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if key:
        return key

    try:
        import streamlit as st

        secrets = st.secrets
        for name in ("GEMINI_API_KEY", "gemini_api_key"):
            if name in secrets:
                return str(secrets[name]).strip()
        if "gemini" in secrets and "api_key" in secrets["gemini"]:
            return str(secrets["gemini"]["api_key"]).strip()
    except Exception:
        # Streamlit не запущен (CLI) или secrets.toml отсутствует
        pass

    return ""


def is_llm_configured() -> bool:
    """Есть ли ключ API (без проверки сети)."""
    return bool(_read_api_key())


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


def _generate(prompt: str, model_name: str, stream: bool):
    api_key = _read_api_key()
    if not api_key:
        raise LlmUnavailableError(
            "Не задан GEMINI_API_KEY. Добавьте его в переменные окружения "
            "или в .streamlit/secrets.toml — пояснения ИИ будут недоступны."
        )

    if _backend() == "genai":
        from google import genai

        client = _clients.get(api_key)
        if client is None:
            client = genai.Client(api_key=api_key)
            _clients[api_key] = client

        if stream:
            return client.models.generate_content_stream(
                model=model_name, contents=prompt
            )
        return client.models.generate_content(model=model_name, contents=prompt)

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    return model.generate_content(
        prompt,
        stream=stream,
        request_options={"timeout": REQUEST_TIMEOUT_SECONDS},
    )


def _call(prompt: str, model_name: str | None = None, stream: bool = False):
    model_name = model_name or DEFAULT_MODEL
    last_error: Exception | None = None

    for attempt in range(1, ATTEMPTS + 1):
        try:
            return _generate(prompt, model_name, stream)
        except LlmUnavailableError:
            raise
        except Exception as exc:  # сеть, квоты, неверный ключ и т.п.
            last_error = exc
            if attempt < ATTEMPTS:
                time.sleep(2 ** (attempt - 1))

    raise LlmUnavailableError(
        f"Gemini недоступен после {ATTEMPTS} попыток: {last_error}"
    ) from last_error


def _response_text(response) -> str:
    """Достаёт текст ответа, не падая на safety-блокировке."""
    try:
        text = getattr(response, "text", None)
        if text:
            return str(text).strip()
    except Exception:
        # SDK бросает ValueError, если нет валидных кандидатов
        pass

    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                return str(text).strip()

    feedback = getattr(response, "prompt_feedback", None)
    detail = f" (prompt_feedback: {feedback})" if feedback else ""
    raise LlmUnavailableError(f"Модель не вернула текст{detail}.")


def enhance_report_with_gemini(stats_text: str, model_name: str | None = None) -> str:
    """Возвращает пояснение одной строкой (для CLI)."""
    response = _call(_build_prompt(stats_text), model_name=model_name, stream=False)
    return _response_text(response)


def stream_gemini_explanation(stats_text: str, model_name: str | None = None):
    """Генератор для потокового вывода в Streamlit."""
    response = _call(_build_prompt(stats_text), model_name=model_name, stream=True)

    for chunk in response:
        try:
            text = getattr(chunk, "text", None)
        except Exception:
            text = None
        if text:
            yield str(text)
