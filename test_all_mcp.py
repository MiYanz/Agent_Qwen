import asyncio
import sys

from mcp_client.manager import MCPManager
from mcp_client.tools import CONFIG_PATH


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


async def main():
    """启动全部 MCP Server，按 Server 输出健康状态和工具清单。"""
    async with MCPManager(CONFIG_PATH) as manager:
        # setdefault 在首次遇到 Server 时创建列表，随后把工具归组。
        tools_by_server = {}
        for server_name, tool in await manager.get_tools():
            tools_by_server.setdefault(server_name, []).append(tool)

        for server_name in manager.servers:
            print("=" * 72)
            print(f"MCP Server:\n\n{server_name}\n")
            health = manager.health.get(server_name)
            print(f"Status: {health.get('state', 'unknown')}\n")
            print("Tools:\n")
            for tool in tools_by_server.get(server_name, []):
                print(f"- {tool.name}")
            print()

        # 列表推导式收集所有非 healthy 状态，使测试以异常退出。
        unhealthy = [
            name
            for name, health in manager.health.get().items()
            if health["state"] != "healthy"
        ]
        if unhealthy:
            raise RuntimeError(
                f"MCP server startup failed: {', '.join(unhealthy)}"
            )


if __name__ == "__main__":
    asyncio.run(main())
