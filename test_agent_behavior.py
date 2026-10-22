# 真实模型的多场景行为回归：依次尝试新闻、热搜、本地文档、文本和目录任务。
# 案例中的路径是实际读取目标；断言观察的是主图消息，不会自动展开子 Agent 内部消息。
import asyncio
import sys

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import create_graph
from mcp_client.manager import MCPManager
from mcp_client.tools import CONFIG_PATH


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


CASES = (
    (
        "news",
        "搜索今天10条重要新闻，按标题、简短摘要、来源整理，不要给原始URL列表",
        {"search"},
    ),
    (
        "weibo",
        "搜索今天微博热搜前10，给出排名和实际热搜内容，不要介绍榜单网站",
        {"search"},
    ),
    (
        "pptx",
        r"总结 D:\BaiduSyncdisk\Lab\02-晨会\2026.9.7.pptx 的内容",
        {"office_convert_to_markdown"},
    ),
    (
        "pdf",
        r"总结 D:\BaiduSyncdisk\Lab\02-晨会\2026.10.13.pdf 的内容",
        {"pdf_convert_to_markdown"},
    ),
    (
        "text",
        r"读取 D:\46-Agent\workspace\routing_test.txt 并概括内容",
        {"read_text_file"},
    ),
    (
        "directory",
        r"列出 D:\46-Agent 目录文件",
        {"list_directory"},
    ),
)
# 每个用例依次保存名称、用户问题，以及预期至少出现的一个工具名称。


# query 是实际任务，expected_tools 是期望看到的主图 ToolMessage 名称集合。
# 检查最终回答经过模型处理，并打印调用链；不会展开嵌套专业 Agent 的内部消息。
async def run_case(graph, name, query, expected_tools):
    """执行单个行为用例，并验证工具结果最终经过 Agent 二次处理。"""
    result = await graph.ainvoke({"messages": [HumanMessage(content=query)]})
    messages = result["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    tool_names = {m.name for m in tool_messages}
    final = messages[-1]

    assert isinstance(final, AIMessage) and not final.tool_calls
    assert str(final.content).strip()
    # 检查集合交集非空，表示至少出现一个期望工具；这里检查的是主图 ToolMessage 而非嵌套 trace。
    assert expected_tools & tool_names, (name, expected_tools, tool_names)
    # 最终回答不能与任一 ToolMessage 完全相同，防止工具原始结果直出。
    assert all(str(final.content).strip() != str(m.content).strip() for m in tool_messages)
    assert not str(final.content).lstrip().startswith("Found ")

    print("=" * 72)
    print(f"CASE={name}")
    print(f"TOOLS={sorted(tool_names)}")
    print("MESSAGE_FLOW=" + " -> ".join(type(m).__name__ for m in messages))
    print("FINAL=" + str(final.content)[:2000])


# 同一管理器和无 Checkpointer 的主图处理多组独立输入，避免每个用例重复启动整套服务。
async def main():
    """复用一个 MCPManager 和 Graph 顺序执行全部回归用例。"""
    async with MCPManager(CONFIG_PATH) as manager:
        graph = await create_graph(manager)
        for case in CASES:
            await run_case(graph, *case)


if __name__ == "__main__":
    asyncio.run(main())
