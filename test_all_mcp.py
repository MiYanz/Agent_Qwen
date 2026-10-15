import asyncio
import sys

from mcp_client.manager import MCPManager
from mcp_client.tools import CONFIG_PATH


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


async def main():
    async with MCPManager(CONFIG_PATH) as manager:
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
