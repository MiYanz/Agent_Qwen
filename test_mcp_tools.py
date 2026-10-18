import asyncio

from mcp_client.tools import load_mcp_tools
from mcp_client.manager import MCPManager
from mcp_client.tools import CONFIG_PATH


async def main():
    """加载全部 LangChain 包装工具，并检查关键工具是否存在。"""
    async with MCPManager(CONFIG_PATH) as manager:
        tools = await load_mcp_tools(manager)
        await verify_tools(tools)


async def verify_tools(tools):
    """按 MCP Server 整理工具名称并执行发现结果断言。"""
    print("Loaded MCP Tools:")
    for tool in tools:
        server_name = (tool.metadata or {}).get("mcp_server", "unknown")
        print(f"- {tool.name} (server: {server_name})")

    tools_by_server = {}
    for tool in tools:
        server_name = (tool.metadata or {}).get("mcp_server", "unknown")
        tools_by_server.setdefault(server_name, set()).add(tool.name)

    assert "list_directory" in tools_by_server["filesystem"]
    assert "search" in tools_by_server["duckduckgo-search"]
    assert "browser_navigate" in tools_by_server["browser"]
    assert "browser_snapshot" in tools_by_server["browser"]
    assert "pdf_convert_to_markdown" in tools_by_server["pdf"]
    assert "office_convert_to_markdown" in tools_by_server["office"]
    assert "office_create_docx" in tools_by_server["office-writer"]
    assert "search_nodes" in tools_by_server["memory"]
    assert "add_observations" in tools_by_server["memory"]
    assert tools_by_server["computer"] == {"Snapshot"}


if __name__ == "__main__":
    asyncio.run(main())
