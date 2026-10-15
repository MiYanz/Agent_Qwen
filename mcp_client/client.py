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


def _masked_proxy(proxy):
    if not proxy:
        return "disabled"
    if "@" not in proxy:
        return proxy
    scheme, remainder = proxy.split("://", 1)
    return f"{scheme}://***@{remainder.rsplit('@', 1)[-1]}"


class MCPClient:
    def __init__(self, config_path):
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as file:
            self.config = json.load(file)

        self.servers = self.config["mcpServers"]

    def get_server_params(self, server_name):
        server = self.servers[server_name]
        command = server["command"]
        if command in {"python", "python3"}:
            command = sys.executable
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
        return StdioServerParameters(
            command=command,
            args=server["args"],
            env=server_env,
        )

    async def get_tools(self, server_name="filesystem"):
        server_params = self.get_server_params(server_name)
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                return tools.tools

    async def get_all_tools(self):
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
        server_params = self.get_server_params(server_name)
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(tool_name, arguments)
