import asyncio
import json
import sys

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import create_graph
from mcp_client.manager import MCPManager
from mcp_client.tools import CONFIG_PATH


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


async def main():
    """执行一次完整 Graph，并验证最终消息和工具消息的基本结构。"""
    # 命令行参数会拼成测试问题；未提供参数时使用默认新闻问题。
    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "搜索今天的重要新闻，列10条即可"
    )
    # async with 确保测试结束后关闭全部 MCP Session 和子进程。
    async with MCPManager(CONFIG_PATH) as manager:
        graph = await create_graph(manager)
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=query)]}
        )

    messages = result["messages"]
    # 打印完整消息链，便于确认 HumanMessage → ToolMessage → AIMessage 流程。
    for message in messages:
        print("=" * 72)
        print(type(message).__name__)
        if isinstance(message, AIMessage) and message.tool_calls:
            print(json.dumps(message.tool_calls, ensure_ascii=False, indent=2))
        if isinstance(message, ToolMessage):
            print(f"tool_name: {message.name}")
        print(message.content)

    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[-1], AIMessage)
    assert not messages[-1].tool_calls
    assert str(messages[-1].content).strip()

    # 如果模型请求过工具，消息历史中至少应出现一条对应的 ToolMessage。
    tool_calls = [
        message
        for message in messages
        if isinstance(message, AIMessage) and message.tool_calls
    ]
    if tool_calls:
        assert any(isinstance(message, ToolMessage) for message in messages)


if __name__ == "__main__":
    asyncio.run(main())
