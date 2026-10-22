# 离线网络边界测试：Mock 系统解析和 HTTP 客户端，不发送真实 DNS 或网页请求。
# 既检查失败回退，也检查成功路径、缓存、启动参数和错误分类，防止修复扩大访问权限。
"""离线检查备用 DNS、上游校验保留、启动参数和错误分类。"""

import json
import socket
import unittest
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mcp_client.client import MCPClient, _prefer_cached_uv
from mcp_client.tools import CONFIG_PATH, _make_tool_coroutine, _web_failure_payload
from mcp_servers.duckduckgo_entry import DNSFallbackSocket


# 同步验证解析器和配置构造，网络和系统 DNS 都以 Mock 隔离，不依赖代理服务实际运行。
class DNSFallbackTests(unittest.TestCase):
    def test_uv_cache_first_and_online_fallback(self):
        args = ["-m", "uv", "tool", "run", "--from", "example==1", "entry", "--flag"]
        for outcome, offline in [(SimpleNamespace(returncode=0), True),
                                 (SimpleNamespace(returncode=1), False),
                                 (subprocess.TimeoutExpired("probe", 8), False)]:
            with patch("mcp_client.client.subprocess.run", side_effect=outcome if isinstance(outcome, Exception) else None,
                       return_value=outcome) as run:
                selected = _prefer_cached_uv("python", args, {})
                self.assertEqual("--offline" in selected, offline)
                self.assertEqual(run.call_args.args[0][-3:], ["python", "-c", "pass"])
                self.assertTrue(run.call_args.kwargs["capture_output"])
        self.assertNotIn("--offline", args)
        with patch("mcp_client.client.subprocess.run") as run:
            self.assertEqual(_prefer_cached_uv("python", args, {"UV_OFFLINE": "false"}), args)
            run.assert_not_called()
    # 系统解析正常时必须直接使用原结果，不能无故向备用解析器发送域名。
    def test_working_system_dns_does_not_use_doh(self):
        resolver = DNSFallbackSocket("http://127.0.0.1:7897")
        resolver._resolve = Mock()
        # 上下文退出后自动恢复被替换对象，不会永久改动机器 DNS 或 httpx。
        with patch("mcp_servers.duckduckgo_entry.socket.getaddrinfo", return_value=["system-result"]):
            self.assertEqual(resolver.getaddrinfo("example.com", 443), ["system-result"])
        resolver._resolve.assert_not_called()

    # 首次本机解析失败才调用备用解析；第二次应使用短期缓存，减少重复失败等待。
    def test_dns_failure_uses_doh_and_short_cache(self):
        resolver = DNSFallbackSocket("http://127.0.0.1:7897")
        resolver._resolve = Mock(return_value=(["1.1.1.1"], 60))
        # 让域名解析报错，而数字 IP 解析成功，模拟备用结果重新包装为标准 getaddrinfo 元组。
        def lookup(host, *args):
            if host == "example.com":
                raise socket.gaierror("local DNS unavailable")
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (host, 443))]
        with patch("mcp_servers.duckduckgo_entry.socket.getaddrinfo", side_effect=lookup) as mock:
            first = resolver.getaddrinfo("example.com", 443, 0, socket.SOCK_STREAM)
            second = resolver.getaddrinfo("example.com", 443, 0, socket.SOCK_STREAM)
            self.assertEqual(first, second)
            self.assertEqual(mock.call_count, 3)
        resolver._resolve.assert_called_once()

    # 没有代理或明确要求数字地址解析时，应保留原错误，不触发备用网络解析。
    def test_no_proxy_or_numeric_lookup_never_uses_doh(self):
        for proxy, flags in ((None, 0), ("http://proxy", socket.AI_NUMERICHOST)):
            resolver = DNSFallbackSocket(proxy)
            resolver._resolve = Mock()
            with patch("mcp_servers.duckduckgo_entry.socket.getaddrinfo", side_effect=socket.gaierror("failed")):
                with self.assertRaises(socket.gaierror):
                    resolver.getaddrinfo("example.com", 443, flags=flags)
            resolver._resolve.assert_not_called()

    # 备用解析也失败时必须继续抛 gaierror，不能伪造地址或悄悄绕过校验。
    def test_doh_failure_remains_dns_failure(self):
        resolver = DNSFallbackSocket("http://proxy")
        resolver._resolve = Mock(side_effect=ValueError("NXDOMAIN"))
        with patch("mcp_servers.duckduckgo_entry.socket.getaddrinfo", side_effect=socket.gaierror("failed")):
            with self.assertRaisesRegex(socket.gaierror, "DNS-over-HTTPS failed"):
                resolver.getaddrinfo("example.com", 443)

    # 故意返回公网和回环混合结果；解析器应交回全部地址，由上游拒绝不安全集合。
    def test_doh_records_are_not_filtered_before_original_guard(self):
        resolver = DNSFallbackSocket("http://proxy")
        resolver._resolve = Mock(return_value=(["127.0.0.1", "1.1.1.1"], 30))
        # 这里只模拟底层解析格式，不替代上游公网判断，返回地址供后续集合断言。
        def lookup(host, *args):
            if host == "example.com":
                raise socket.gaierror("failed")
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (host, 443))]
        with patch("mcp_servers.duckduckgo_entry.socket.getaddrinfo", side_effect=lookup):
            infos = resolver.getaddrinfo("example.com", 443)
        self.assertEqual({info[4][0] for info in infos}, {"127.0.0.1", "1.1.1.1"})

    # 模拟解析服务失败状态和非法 IP 字符串，确保两者都不能作为成功记录返回。
    def test_doh_resolver_requires_success_and_valid_records(self):
        resolver = DNSFallbackSocket("http://proxy")
        for payload in ({"Status": 3}, {"Status": 0, "Answer": [{"type": 1, "data": "bad-ip"}]}):
            with patch("mcp_servers.duckduckgo_entry.httpx.Client") as client:
                client.return_value.__enter__.return_value.get.return_value.json.return_value = payload
                with self.assertRaises(ValueError):
                    resolver._resolve("example.com", socket.AF_INET)

    # 对比调用前后的配置序列化结果，验证薄入口只改变返回参数，不改原配置或其他服务命令。
    def test_entry_preserves_config_and_other_server_commands(self):
        client = MCPClient(CONFIG_PATH)
        before = json.dumps(client.servers, sort_keys=True)
        with patch("mcp_client.client.subprocess.run", return_value=SimpleNamespace(returncode=0)):
            params = client.get_server_params("duckduckgo-search")
        self.assertIn("duckduckgo_entry.py", " ".join(params.args))
        self.assertIn("duckduckgo-mcp-server[browser]==0.7.0", params.args)
        self.assertEqual(params.args[-2:], ["--search-backend", "auto"])
        self.assertEqual(client.get_server_params("filesystem").args, client.servers["filesystem"]["args"])
        self.assertEqual(json.dumps(client.servers, sort_keys=True), before)

    # 构造上游带误导尾部的 DNS 错误，确认优先分类为解析失败，并与实际非公网错误区分。
    def test_dns_failure_is_not_reported_as_private_address(self):
        result = json.loads(_web_failure_payload(
            "Error: Refusing to fetch https://example.com (could not resolve host 'example.com': getaddrinfo failed). "
            "This server blocks requests to private/internal addresses to prevent SSRF."))
        self.assertEqual(result["error_code"], "dns_resolution_failed")
        self.assertNotIn("This server blocks", result["error"])
        private = json.loads(_web_failure_payload("Error: refusing to fetch host: it resolves to non-public address 127.0.0.1"))
        self.assertEqual(private["error_code"], "non_public_address_blocked")


# 异步验证真实 MCP 工具包装器如何处理上游普通文本错误，不启动真实服务。
class WebFailureTests(unittest.IsolatedAsyncioTestCase):
    # 即使 isError=False，fetch_content 返回 Error 文本时也必须转换成稳定的失败 JSON。
    async def test_plain_text_mcp_error_is_structured(self):
        # 最小管理器替身，仅提供包装器需要的异步 call_tool 接口。
        class Manager:
            # 返回 SDK 风格文本块，专门复现“内容报错但错误标志未设置”的上游情况。
            async def call_tool(self, *args, **kwargs):
                return SimpleNamespace(isError=False, content=[SimpleNamespace(text="Error: could not resolve host example.com")])
        result = await _make_tool_coroutine(Manager(), "duckduckgo-search", "fetch_content")(url="https://example.com")
        self.assertEqual(json.loads(result)["error_code"], "dns_resolution_failed")


if __name__ == "__main__":
    unittest.main()
