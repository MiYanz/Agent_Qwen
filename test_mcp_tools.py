# 真实工具发现回归：检查包装后的名称、前缀和服务分组是否符合预期。
# Memory 工具不进入通用模型工具列表，但仍应能通过管理器的原始发现结果查询。
import asyncio

from mcp_client.tools import load_mcp_tools
from mcp_client.manager import MCPManager
from mcp_client.tools import CONFIG_PATH


# 启动真实管理器并加载生产 StructuredTool，再交给 verify_tools 检查；退出时统一关闭。
async def main():
    """加载全部 LangChain 包装工具，并检查关键工具是否存在。"""
    async with MCPManager(CONFIG_PATH) as manager:
        tools = await load_mcp_tools(manager)
        await verify_tools(tools, manager)


# tools 是包装后列表，manager 提供包含 Memory 的原始工具缓存。
# 分别断言展示名、服务归属及 Memory 的暴露边界，不调用工具的实际业务功能。
async def verify_tools(tools, manager):
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
    # Memory MCP 由 MemoryManager 统一访问，不应绕过记忆策略暴露给通用 ToolNode。
    assert "memory" not in tools_by_server
    # 再次查询管理器的原始发现结果；与模型包装列表相比，这一层仍应能看到 Memory 服务。
    manager_memory_tools = {
        tool.name
        for server_name, tool in await manager.get_tools()
        if server_name == "memory"
    }
    assert {"create_entities", "add_observations", "search_nodes"} <= manager_memory_tools
    assert tools_by_server["computer"] == {"Snapshot"}


if __name__ == "__main__":
    asyncio.run(main())
