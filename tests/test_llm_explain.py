import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from modules import llm_explain
from modules.llm_explain import (
    LlmUnavailableError,
    configured_providers,
    default_provider,
    stream_chat,
)

CLEAN = {name: "" for name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
                               "OPENAI_MODEL", "LLM_PROVIDER")}


class FakeChat(BaseHTTPRequestHandler):
    """Chat Completions со стримингом: два куска и [DONE]."""

    requests: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeChat.requests.append((self.path, dict(self.headers), body))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for text in ("При", "вет"):
            event = {"choices": [{"delta": {"content": text}}]}
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


def setUpModule():
    # локальный .streamlit/secrets.toml не должен влиять на тесты
    patcher = patch.object(llm_explain, "_secret", return_value="")
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


class ProviderTests(unittest.TestCase):
    def test_configured_and_default(self):
        with patch.dict(os.environ, CLEAN):
            self.assertEqual(configured_providers(), [])
            self.assertIsNone(default_provider())
        env = {**CLEAN, "GEMINI_API_KEY": "g", "OPENAI_BASE_URL": "http://localhost:11434/v1"}
        with patch.dict(os.environ, env):
            self.assertEqual(configured_providers(), ["gemini", "openai"])
            self.assertEqual(default_provider(), "gemini")
        with patch.dict(os.environ, {**env, "LLM_PROVIDER": "OpenAI"}):
            self.assertEqual(default_provider(), "openai")

    def test_key_from_interface(self):
        with patch.dict(os.environ, CLEAN):
            self.assertEqual(configured_providers({"openai": "k"}), ["openai"])
            self.assertEqual(default_provider({"gemini": "k"}), "gemini")

    def test_unconfigured_provider_is_rejected(self):
        with patch.dict(os.environ, CLEAN), self.assertRaises(LlmUnavailableError):
            list(stream_chat("openai", [{"role": "user", "content": "x"}]))


class OpenAiStreamTests(unittest.TestCase):
    def setUp(self):
        self.server = HTTPServer(("127.0.0.1", 0), FakeChat)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        FakeChat.requests = []
        self.env = {**CLEAN, "OPENAI_BASE_URL": f"http://127.0.0.1:{self.server.server_port}/v1/",
                    "OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "local-model"}

    def test_stream_and_payload(self):
        messages = [{"role": "user", "content": "вопрос"},
                    {"role": "assistant", "content": "ответ"},
                    {"role": "user", "content": "ещё"}]
        with patch.dict(os.environ, self.env):
            text = "".join(stream_chat("openai", messages, system="роль"))
        self.assertEqual(text, "Привет")
        path, headers, body = FakeChat.requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(body["model"], "local-model")
        self.assertTrue(body["stream"])
        with patch.dict(os.environ, self.env):
            "".join(stream_chat("openai", messages, api_key="ui-key"))
        self.assertEqual(FakeChat.requests[-1][1]["Authorization"], "Bearer ui-key")
        self.assertEqual([m["role"] for m in body["messages"]],
                         ["system", "user", "assistant", "user"])

    def test_no_retry_after_partial_answer(self):
        calls = []

        def broken(system, messages, model, api_key=None):
            calls.append(1)
            yield "часть"
            raise OSError("обрыв")

        chunks = []
        with patch.dict(os.environ, self.env), \
                patch.object(llm_explain, "_openai_stream", broken), \
                self.assertRaises(LlmUnavailableError):
            for chunk in stream_chat("openai", [{"role": "user", "content": "x"}]):
                chunks.append(chunk)
        self.assertEqual((chunks, len(calls)), (["часть"], 1))

    def test_retry_before_first_chunk(self):
        attempts = []

        def flaky(system, messages, model, api_key=None):
            attempts.append(1)
            if len(attempts) < 2:
                raise OSError("429")
            yield "ок"

        with patch.dict(os.environ, self.env), \
                patch.object(llm_explain, "_openai_stream", flaky), \
                patch.object(llm_explain.time, "sleep"):
            text = "".join(stream_chat("openai", [{"role": "user", "content": "x"}]))
        self.assertEqual((text, len(attempts)), ("ок", 2))
