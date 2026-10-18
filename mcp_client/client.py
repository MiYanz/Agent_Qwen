import json
import os
import sys
import tempfile
from pathlib import Path
from urllib.request import getproxies

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


NETWORK_MCP_SERVERS = {
    "duckduckgo-search",
    "browser",
    "pdf",
    "office",
}
# 只有需要访问网络的子进程继承系统代理；Filesystem 等本地服务不受代理影响。


def _masked_proxy(proxy):
    """隐藏代理 URL 中可能包含的账号信息，避免日志泄露凭据。"""
    if not proxy:
        return "disabled"
    if "@" not in proxy:
        return proxy
    scheme, remainder = proxy.split("://", 1)
    return f"{scheme}://***@{remainder.rsplit('@', 1)[-1]}"


class MCPClient:
    """读取 MCP 配置，并按配置构造短生命周期客户端连接。"""

    def __init__(self, config_path):
        # Path.open 配合 with 使用，读取结束后文件句柄会自动关闭。
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as file:
            self.config = json.load(file)

        self.servers = self.config["mcpServers"]

    def get_server_params(self, server_name):
        """把单个 Server 配置转换为 MCP stdio 启动参数。"""
        server = self.servers[server_name]
        command = server["command"]
        if command in {"python", "python3"}:
            # 使用当前 Conda 环境的解释器，避免子进程落到系统 Python。
            command = sys.executable
        # 从父进程环境复制一份，后续修改不会污染当前 Python 进程的 os.environ。
        server_env = os.environ.copy()
        server_env.setdefault(
            "NPM_CONFIG_CACHE",
            str(Path(tempfile.gettempdir()) / "46-agent-npm-cache"),
        )
        if server_name in NETWORK_MCP_SERVERS:
            system_proxies = getproxies()
            http_proxy = system_proxies.get("http")
            https_proxy = system_proxies.get("https") or http_proxy
            all_proxy = system_proxies.get("all") or https_proxy or http_proxy
            proxy_values = {
                "HTTP_PROXY": http_proxy,
                "HTTPS_PROXY": https_proxy,
                "ALL_PROXY": all_proxy,
            }
            for key, proxy in proxy_values.items():
                if not proxy:
                    continue
                # Python, Node and native subprocesses do not consistently use
                # the same casing, so provide both variants.
                server_env.setdefault(key, proxy)
                server_env.setdefault(key.lower(), proxy)
            server_env.setdefault("NO_PROXY", "127.0.0.1,localhost")
            server_env.setdefault("no_proxy", server_env["NO_PROXY"])
            print(
                f"[MCP Proxy] server={server_name} "
                f"http={_masked_proxy(http_proxy)} "
                f"https={_masked_proxy(https_proxy)}",
                flush=True,
            )
        server_env.update(server.get("env", {}))
        # 配置文件中的 env 最后覆盖默认值，便于为单个 Server 指定专用环境变量。
        return StdioServerParameters(
            command=command,
            args=server["args"],
            env=server_env,
        )

    async def get_tools(self, server_name="filesystem"):
        """临时启动一个 Server，初始化 Session，并返回它公开的工具定义。"""
        server_params = self.get_server_params(server_name)
        # async with 会在离开代码块时异步关闭 stdio 连接、Session 和子进程。
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                return tools.tools

    async def get_all_tools(self):
        """依次发现配置中的全部工具；该方法每次都会临时连接各 Server。"""
        all_tools = []
        for server_name in self.servers:
            server_tools = await self.get_tools(server_name)
            all_tools.extend((server_name, tool) for tool in server_tools)
        return all_tools

    async def call_tool(
        self,
        tool_name,
        arguments,
        server_name="filesystem",
    ):
        """通过一次临时 MCP Session 调用工具；长期复用由 MCPManager 负责。"""
        server_params = self.get_server_params(server_name)
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(tool_name, arguments)
