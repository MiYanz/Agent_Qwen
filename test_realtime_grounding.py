import argparse
import asyncio
import json
import sys

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import create_graph
from mcp_client.manager import MCPManager
from mcp_client.tools import CONFIG_PATH, _make_tool_coroutine


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


class EmptyResult:
    content = []
    isError = False


class EmptySearchManager:
    async def call_tool(self, *args, **kwargs):
        return EmptyResult()


async def test_empty_result():
    search = _make_tool_coroutine(
        EmptySearchManager(),
        "duckduckgo-search",
        "search",
    )
    empty_result = await search(query="offline test")
    payload = json.loads(empty_result)
    assert payload["status"] == "failed"
    assert payload["error"] == "工具没有返回内容"
    print("EMPTY_RESULT=PASS")


async def test_live_hot_search():
    async with MCPManager(CONFIG_PATH) as manager:
        graph = await create_graph(manager)
        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content="给我当前微博热搜前10，只列真实热搜内容"
                    )
                ]
            }
        )

    messages = result["messages"]
    names = [m.name for m in messages if isinstance(m, ToolMessage)]
    final = messages[-1]
    assert "search" in names
    assert any(
        name in {"fetch_content", "browser_snapshot", "browser_evaluate"}
        for name in names
    )
    assert isinstance(final, AIMessage) and not final.tool_calls
    assert "标题1" not in str(final.content)
    assert "当前未获取到有效" not in str(final.content)
    print(f"LIVE_TOOLS={names}")
    print("LIVE_HOT_SEARCH=PASS")


async def main(live):
    await test_empty_result()
    if live:
        await test_live_hot_search()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.live))
