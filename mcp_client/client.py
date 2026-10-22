# 配置和连接参数层：读取 JSON，准备 Python/Node 进程、代理环境与服务入口。
# 本类也提供临时会话方法；应用正常运行时主要由 MCPManager 持有并复用长连接。
import json
import os
import sys
import subprocess
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


# 只生成代理的日志展示值，隐藏 URL 中的认证部分；实际连接仍使用原始代理。
def _masked_proxy(proxy):
    """隐藏代理 URL 中可能包含的账号信息，避免日志泄露凭据。"""
    if not proxy:
        return "disabled"
    if "@" not in proxy:
        return proxy
    scheme, remainder = proxy.split("://", 1)
    return f"{scheme}://***@{remainder.rsplit('@', 1)[-1]}"


def _prefer_cached_uv(command, args, env):
    """缓存可完整解析时离线启动；首次安装仍走原有联网命令，不修改配置。"""
    if args[:4] != ["-m", "uv", "tool", "run"] or "UV_OFFLINE" in env:
        return args
    index = 4
    # 只识别项目当前使用的明确包参数，遇到未知选项保留原命令。
    while index + 1 < len(args) and args[index] in {"--from", "--with"}:
        index += 2
    if index == 4 or index >= len(args) or args[index].startswith("-"):
        return args
    prefix = args[:4] + ["--offline"] + args[4:index]
    try:
        # 只执行空 Python 程序验证同一套精确依赖能从缓存解析，不启动第二套 MCP。
        # 离线探测不会访问 PyPI；冷缓存或损坏缓存直接回到原来的联网安装路径。
        result = subprocess.run([command, *prefix, "python", "-c", "pass"], env=env,
            capture_output=True, timeout=8,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode == 0:
            print("[MCP Runtime] using verified uv cache (offline)", flush=True)
            return args[:4] + ["--offline"] + args[4:]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return args


# 负责理解配置并构造 stdio 连接参数；不缓存长连接，长连接所有权属于 MCPManager。
class MCPClient:
    """读取 MCP 配置，并按配置构造短生命周期客户端连接。"""

    # config_path 指向 JSON；读取后保留完整配置和 mcpServers 映射，暂不启动任何子进程。
    def __init__(self, config_path):
        # Path.open 配合 with 使用，读取结束后文件句柄会自动关闭。
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as file:
            self.config = json.load(file)

        self.servers = self.config["mcpServers"]

    # 按服务名生成 StdioServerParameters，包括命令、参数和独立环境副本。
    # 可能读取系统代理并打印脱敏诊断，但不会改写磁盘配置或父进程环境变量。
    def get_server_params(self, server_name):
        """把单个 Server 配置转换为 MCP stdio 启动参数。"""
        server = self.servers[server_name]
        command = server["command"]
        if command in {"python", "python3"}:
            # 使用当前 Conda 环境的解释器，避免子进程落到系统 Python。
            command = sys.executable
        # 从父进程环境复制一份，后续修改不会污染当前 Python 进程的 os.environ。
        # 复制而不是直接修改 os.environ，使不同服务的 env 覆盖互不污染。
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
        # 用户在配置中的服务环境值优先于自动推导的默认值。
        server_env.update(server.get("env", {}))
        # 参数也取新列表，避免为 DuckDuckGo 包装入口时把内存配置本身改掉。
        server_args = list(server["args"])
        if server_name in {"duckduckgo-search", "pdf", "office"}:
            server_args = _prefer_cached_uv(command, server_args, server_env)
        if server_name == "duckduckgo-search" and "duckduckgo-mcp-server" in server_args:
            # 保留 uv 包版本和 Server 参数，只将入口切换到项目维护的薄包装器。
            # python 由 uv 工具环境解析，确保能够导入其已隔离安装的依赖。
            entry_index = server_args.index("duckduckgo-mcp-server")
            server_args[entry_index:entry_index + 1] = [
                "python", str(Path(__file__).resolve().parents[1] / "mcp_servers" / "duckduckgo_entry.py")
            ]
        # 配置文件中的 env 最后覆盖默认值，便于为单个 Server 指定专用环境变量。
        return StdioServerParameters(
            command=command,
            args=server_args,
            env=server_env,
        )

    # 临时连接单个服务，握手后查询原始工具；退出两层上下文时关闭会话及 stdio。
    # 适合独立发现操作；主应用重复调用工具时不使用此方法反复启动服务。
    async def get_tools(self, server_name="filesystem"):
        """临时启动一个 Server，初始化 Session，并返回它公开的工具定义。"""
        server_params = self.get_server_params(server_name)
        # async with 会在离开代码块时异步关闭 stdio 连接、Session 和子进程。
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                return tools.tools

    # 遍历配置中的服务，顺序执行临时工具发现；返回带服务名的二元组列表。
    # 当前方法不隔离单个服务异常，任一 get_tools 失败会向调用者传播。
    async def get_all_tools(self):
        """依次发现配置中的全部工具；该方法每次都会临时连接各 Server。"""
        all_tools = []
        for server_name in self.servers:
            server_tools = await self.get_tools(server_name)
            all_tools.extend((server_name, tool) for tool in server_tools)
        return all_tools

    # 单次临时调用入口；arguments 原样传给指定服务，退出上下文即释放资源。
    # 返回 MCP 原始结果，不在此转换成给模型看的 ToolMessage。
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
