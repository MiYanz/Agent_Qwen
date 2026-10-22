"""启动回归：真实 CMD 解析与启用统计分支的 Gradio 启动，不启动 MCP、不写正式数据。"""

import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from urllib.request import ProxyHandler, build_opener


ROOT = Path(__file__).resolve().parent


class StartupTests(unittest.TestCase):
    def test_all_builtin_theme_comparisons(self):
        import app
        from gradio import utils

        # 执行 launch 报错中的真实比较表达式，不能只检查 CSS 里出现了字体名字。
        is_custom = not any(app.APP_THEME.to_dict() == theme.to_dict()
                            for theme in utils.BUILT_IN_THEMES.values())
        self.assertTrue(is_custom)
        self.assertTrue(all(font.stylesheet() == {"url": None, "css": None}
                            for font in [*app.APP_THEME._font, *app.APP_THEME._font_mono]))

    def test_gradio_launch_with_analytics_branch_enabled(self):
        import app

        # 使用真正的页面组件和 HTTP 服务，但不传生产 lifespan，因此不会创建 MCP 或打开 SQLite。
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        app.demo.analytics_enabled = True
        try:
            # 只阻止统计上报发往外网，主题比较仍正常执行；这里不能关闭 analytics 绕过异常分支。
            with patch("gradio.analytics.launched_analytics") as analytics:
                app.demo.launch(server_name="127.0.0.1", server_port=port,
                    theme=app.APP_THEME, css=app.CSS, prevent_thread_lock=True, quiet=True)
                analytics.assert_called_once()
                self.assertTrue(analytics.call_args.args[1]["is_custom_theme"])
                opener = build_opener(ProxyHandler({}))
                with opener.open(f"http://127.0.0.1:{port}/config", timeout=10) as response:
                    config = json.load(response)
                self.assertIn("Times New Roman", config["css"])
                self.assertIn("SimHei", config["css"])
        finally:
            app.demo.close(verbose=False)

    @unittest.skipUnless(os.name == "nt", "CMD 解析验证仅适用于 Windows")
    def test_cmd_launcher_parses_without_chinese_command_fragments(self):
        data = (ROOT / "start_agent.bat").read_bytes()
        text = data.decode("ascii")
        self.assertEqual(data.count(b"\n"), data.count(b"\r\n"))
        # 临时副本保留真实 CMD/PowerShell 续行语法；替换有副作用的入口，绝不停止现有 Agent。
        text = text.replace(
            "$connection = @(Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue)[0]; ",
            "$connection = $null; ")
        self.assertNotIn("Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue)[0]", text)
        text = text.replace(r"call C:\ProgramData\anaconda3\Scripts\activate.bat qwen_agent",
                            'call "%~dp0activate_stub.bat"')
        text = text.replace("python app.py", "echo APP_LAUNCH_OK")
        text = text.replace("    pause", "    REM no pause during tests").replace("\r\npause", "\r\nREM no pause during tests")
        with tempfile.TemporaryDirectory() as directory:
            mirror = Path(directory) / "launcher.bat"
            mirror.write_bytes(text.encode("ascii"))
            (mirror.parent / "activate_stub.bat").write_bytes(b"@echo off\r\necho ACTIVATION_OK\r\nexit /b 0\r\n")
            completed = subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", str(mirror)],
                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
                creationflags=subprocess.CREATE_NO_WINDOW)
        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, output)
        self.assertIn("ACTIVATION_OK", output)
        self.assertIn("APP_LAUNCH_OK", output)
        self.assertIn("[4/4] Starting Agent...", output)
        self.assertNotIn("is not recognized", output)


if __name__ == "__main__":
    unittest.main()
