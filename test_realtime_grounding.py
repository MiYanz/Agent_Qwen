# 搜索证据回归：默认只测试空结果转换；--live 才验证实时热搜。
# 实时分支依赖模型、网络和各 MCP 服务，不能把离线通过理解为实时榜单已经核验。
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


# 模拟 MCP 返回空 content 且不设置失败标志，用来检查包装层的兜底判断。
class EmptyResult:
    """模拟 MCP 成功返回但没有任何内容块的结果。"""
    content = []
    isError = False


# 只提供调用接口的最小 Fake；不读取配置、不建立 Session。
class EmptySearchManager:
    """提供最小 call_tool 接口，用于离线测试空结果包装逻辑。"""
    # 任何请求都返回 EmptyResult，保证测试不依赖真实搜索结果。
    async def call_tool(self, *args, **kwargs):
        return EmptyResult()


# 调用生产包装协程并解析失败 JSON，验证空内容不会被伪装为搜索成功。
async def test_empty_result():
    """确认空工具结果会转换为明确的 failed JSON。"""
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


# 真实运行主图，检查搜索与正文工具都出现，且最终回答没有测试占位标题。
# 该分支具有网络和正式服务依赖，默认执行路径不会进入这里。
async def test_live_hot_search():
    """联网验证热搜必须经过搜索和正文获取，且不能生成占位标题。"""
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


# 总是运行离线空结果检查，仅 live=True 时才追加真实热搜验证。
async def main(live):
    """默认只跑离线测试；--live 才会执行真实网络搜索。"""
    await test_empty_result()
    # 只有主动开启实时验证才进入网络分支；默认测试可用于安全的离线回归。
    if live:
        await test_live_hot_search()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.live))
