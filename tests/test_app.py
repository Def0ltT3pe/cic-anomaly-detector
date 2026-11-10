"""Смоук-тесты интерфейса через streamlit.testing: вход, загрузка, вкладки.

Приложение запускается целиком (настоящие модели из models/), но все пути с
состоянием — пользователи, аудит, мониторинг, список серверов — уводятся во
временную папку, а ssh скрывается, чтобы результат не зависел от машины.
"""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from modules import auth
from modules.monitor import Server, run
from tests.helpers import make_temp_dir, remove_temp_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "App.py")
TIMEOUT = 120


def synthetic_flows(rows: int = 60) -> bytes:
    """CSV с признаками модели около нормы и адресами из 203.0.113.0/24."""
    with open(os.path.join(ROOT, "models", "normal_stats.json"), encoding="utf-8") as handle:
        stats = json.load(handle)
    rng = np.random.default_rng(0)
    data = {name: np.abs(rng.normal(mean, stats["stds"][name] or 1.0, rows))
            for name, mean in stats["means"].items()}
    frame = pd.DataFrame(data)
    frame["Src IP"] = [f"203.0.113.{n % 20}" for n in range(rows)]
    frame["Src Port"] = 40000 + np.arange(rows)
    frame["Dst IP"] = "10.0.0.1"
    frame["Dst Port"] = [22 if n % 3 == 0 else 443 for n in range(rows)]
    frame["Timestamp"] = pd.date_range("2026-11-01 10:00", periods=rows, freq="30s").strftime("%Y-%m-%d %H:%M:%S")
    return frame.to_csv(index=False).encode("utf-8")


class AppSmokeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = make_temp_dir()
        self.addCleanup(remove_temp_dir, self.tmp)
        self.env = {
            "USERS_FILE": os.path.join(self.tmp, "users.json"),
            "AUDIT_ENABLED": "0",
            "MONITOR_DIR": os.path.join(self.tmp, "monitor"),
            "PULL_CONFIG": os.path.join(self.tmp, "pull_captures.json"),
            "APP_PASSWORD": "",
            "ALLOW_OPEN_MODE": "1",
            "MODEL_DIR": "models",
        }
        auth.LOGIN_THROTTLE._state.clear()  # блокировка общая на процесс

    def run_app(self, ssh: bool = False, **env) -> AppTest:
        self.env.update(env)  # и для следующих перезапусков этого же приложения
        with patch.dict(os.environ, self.env), \
                patch("shutil.which", return_value="ssh" if ssh else None):
            app = AppTest.from_file(APP, default_timeout=TIMEOUT)
            app.run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        return app

    def rerun(self, app: AppTest, ssh: bool = False) -> AppTest:
        with patch.dict(os.environ, self.env), \
                patch("shutil.which", return_value="ssh" if ssh else None):
            app.run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        return app

    def test_open_mode_starts(self):
        app = self.run_app()
        self.assertIn("Детектор аномалий трафика", app.title[0].value)
        self.assertEqual(len(app.file_uploader), 1)
        self.assertEqual(len(app.tabs), 0)  # без мониторинга и ssh вкладок нет

    def test_closed_mode_without_login(self):
        app = self.run_app(ALLOW_OPEN_MODE="0")
        self.assertEqual(app.title[0].value, "Вход не настроен")
        self.assertEqual(len(app.file_uploader), 0)

    def test_shared_password_login(self):
        app = self.run_app(APP_PASSWORD="ci-check-password")
        self.assertEqual(app.title[0].value, "Вход")
        self.assertEqual(len(app.file_uploader), 0)
        app.text_input[0].set_value("wrong")
        app.button[0].click()
        self.rerun(app)
        self.assertIn("Неверный пароль", [e.value for e in app.error])
        app.text_input[0].set_value("ci-check-password")
        app.button[0].click()
        self.rerun(app)
        self.assertEqual(len(app.file_uploader), 1)

    def test_viewer_has_no_servers_tab(self):
        auth.add_user("viewer1", "viewer-password", role="viewer", path=self.env["USERS_FILE"])
        app = self.run_app(ssh=True)
        login, password = app.text_input[0], app.text_input[1]
        login.set_value("viewer1")
        password.set_value("viewer-password")
        app.button[0].click()
        self.rerun(app, ssh=True)
        self.assertEqual(len(app.file_uploader), 1)
        self.assertFalse(any("Серверы" in tab.label for tab in app.tabs))

    def test_admin_sees_servers_tab_with_ssh(self):
        app = self.run_app(ssh=True)
        labels = [tab.label for tab in app.tabs]
        self.assertTrue(any("Серверы" in label for label in labels), labels)

    def test_monitoring_tab_appears_after_monitor_run(self):
        local = os.path.join(self.tmp, "captures", "serverA", "session1")
        os.makedirs(local)
        with open(os.path.join(local, "capture_1.pcapng"), "wb") as handle:
            handle.write(b"pcap")
        rows = pd.DataFrame([{"Src IP": "203.0.113.5", "Src Port": 40001, "Dst IP": "10.0.0.1",
                              "Dst Port": 22, "Timestamp": pd.Timestamp.now().floor("min"), "proba": 0.9}])
        run([Server("serverA", os.path.dirname(local))], lambda path, server: (rows, "10.0.0.1"),
            None, directory=self.env["MONITOR_DIR"], log=lambda message: None)
        app = self.run_app()
        labels = [tab.label for tab in app.tabs]
        monitoring = next((label for label in labels if "Мониторинг" in label), None)
        self.assertIsNotNone(monitoring, labels)
        # Закрытая вкладка не считается
        self.assertNotIn("За сутки", [metric.label for metric in app.metric])
        app.session_state["main_tab"] = monitoring
        self.rerun(app)
        self.assertIn("За сутки", [metric.label for metric in app.metric])

    def test_analysis_survives_switching_tabs(self):
        app = self.run_app(ssh=True)
        servers = next(tab.label for tab in app.tabs if "Серверы" in tab.label)
        app.file_uploader[0].set_value(("flows.csv", synthetic_flows(), "text/csv"))
        self.rerun(app, ssh=True)
        next(b for b in app.button if b.label == "Анализировать").click()
        self.rerun(app, ssh=True)
        app.segmented_control(key="view").set_value("Потоки")
        self.rerun(app, ssh=True)
        app.session_state["main_tab"] = servers
        self.rerun(app, ssh=True)
        self.assertEqual(len(app.segmented_control), 0)  # анализ на закрытой вкладке не рисуется
        app.session_state["main_tab"] = app.tabs[0].label
        self.rerun(app, ssh=True)
        self.assertEqual(app.segmented_control(key="view").value, "Потоки")
        self.assertEqual(len(app.radio(key="flows_filter").options), 3)  # раздел «Потоки» снова на месте

    def test_upload_analyze_and_switch_views(self):
        app = self.run_app()
        app.file_uploader[0].set_value(("flows.csv", synthetic_flows(), "text/csv"))
        self.rerun(app)
        analyze = [button for button in app.button if button.label == "Анализировать"]
        self.assertEqual(len(analyze), 1)
        analyze[0].click()
        self.rerun(app)
        self.assertEqual(app.metric[0].label, "Потоков")
        self.assertEqual(app.metric[0].value, "60")
        for view in ("Инциденты", "Потоки", "Разбор потока", "Обзор"):
            with self.subTest(view=view):
                # Элемент берётся заново: после перезапуска старый уже не связан с деревом
                app.segmented_control(key="view").set_value(view)
                self.rerun(app)
                self.assertEqual(app.segmented_control(key="view").value, view)
        # Таблица с переходом по клику (на этих данных тревог нет — берём все потоки)
        app.segmented_control(key="view").set_value("Потоки")
        self.rerun(app)
        app.radio(key="flows_filter").set_value("Все потоки")
        self.rerun(app)
        self.assertGreaterEqual(len(app.dataframe), 1)

    def test_programmatic_jump_to_flow_review(self):
        app = self.run_app()
        app.file_uploader[0].set_value(("flows.csv", synthetic_flows(), "text/csv"))
        self.rerun(app)
        next(b for b in app.button if b.label == "Анализировать").click()
        self.rerun(app)
        # Так переходит клик по строке таблицы: goto_view/goto_flow до создания виджетов
        app.session_state["goto_view"] = "Разбор потока"
        app.session_state["goto_flow"] = 7
        self.rerun(app)
        self.assertEqual(app.segmented_control(key="view").value, "Разбор потока")
        self.assertEqual(app.session_state["review_flow"], 7)


class _State(dict):
    """Замена st.session_state: доступ и по ключу, и через атрибут."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class RowClickTests(unittest.TestCase):
    def setUp(self):
        import App  # noqa: PLC0415 — модуль приложения, main() не запускается
        self.app = App
        self.state = _State()
        patcher = patch.object(App.st, "session_state", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_click_opens_flow_review(self):
        self.state["flows_table_0"] = SimpleNamespace(selection=SimpleNamespace(rows=[1]))
        self.app._open_selected_flow("flows_table_0", [5, 9, 12])
        self.assertEqual((self.state["goto_flow"], self.state["goto_view"]), (9, "Разбор потока"))

    def test_cleared_selection_does_nothing(self):
        self.state["incidents_table"] = SimpleNamespace(selection=SimpleNamespace(rows=[]))
        self.app._open_selected_flow("incidents_table", [5])
        self.assertNotIn("goto_flow", self.state)


if __name__ == "__main__":
    unittest.main()
