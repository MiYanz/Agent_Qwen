# 真实协作冒烟测试，需要显式 --live 才会启动 MCP 和访问网络。
# NoMemory 用来阻止测试接触正式长期记忆；文件读取仍会访问示例中的真实 README 路径。
"""显式 --live 才启动真实 MCP/Ollama；不访问用户记忆，不创建文档。"""

import argparse
import asyncio
import json
import re
import sys

from langchain_core.messages import HumanMessage, ToolMessage

from agent.graph import create_graph
from agent.nodes import llm
from agent.subagents import AgentRegistry
from mcp_client.manager import MCPManager
from mcp_client.tools import CONFIG_PATH, load_mcp_tools
from memory.models import MemoryOperationResult

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


# 真实服务测试中的记忆隔离层，保留主图所需接口但不访问正式存储。
class NoMemory:
    # 所有输入按普通任务处理，避免实际执行显式记忆命令。
    async def handle_command(self, query):
        return MemoryOperationResult(handled=False)

    # 不注入用户历史资料，返回空记录列表。
    async def retrieve_memories(self, query):
        return []

    # 不向真实模型额外添加长期记忆文本。
    def render_context(self, records):
        return ""

    # 测试完成后仍保持 skipped，防止验证任务触发正式记忆写入。
    async def process_automatic_memory(self, query):
        return "skipped"


# discover_only 只核对相关服务权限；否则用真实模型要求连续委托并检查主图结果。
# 管理器退出上下文时关闭服务；仅在命令行显式启用 --live 后调用。
async def main(discover_only=False, query=None):
    manager = MCPManager(CONFIG_PATH)
    # 复用原启动命令，只在本测试实例中选择只读任务需要的 Server。
    selected = {name: config for name, config in manager.servers.items()
                if name in {"filesystem", "duckduckgo-search", "pdf", "office"}}
    manager.servers = manager.client.servers = selected
    async with manager:
        tools = await load_mcp_tools(manager)
        registry = AgentRegistry(tools, llm)
        for name in ("research", "file"):
            print("PERMISSIONS=" + json.dumps({name: [t.name for t in registry.tools_for(name)]}))
        if discover_only:
            return
        graph = await create_graph(manager, memory_manager=NoMemory())
        # 不在用户输入中点名 Agent，验证普通搜索/文件请求也会经过专业角色，而非依赖演示措辞。
        query = query or (r"先搜索 LangGraph Agent-as-Tool 的官方文档并读取正文，"
                 r"再读取 D:\46-Agent\README.md，比较官方方案与本项目的状态隔离设计。"
                 r"必须依据实际证据回答；失败请如实说明。")
        result = await graph.ainvoke({"messages": [HumanMessage(content=query)]},
                                    config={"recursion_limit": 60})
        delegated = []
        incomplete = False
        acquired = {"research": False, "file": False}
        for message in result["messages"]:
            if isinstance(message, ToolMessage):
                assert message.name not in {"search", "fetch_content", "read_text_file",
                                             "pdf_convert_to_markdown", "office_convert_to_markdown"}, message.name
            if isinstance(message, ToolMessage) and message.name.startswith("call_"):
                payload = json.loads(message.content)
                # 不能再仅检查两个委托入口出现：之前空 File 报告也会让这种弱断言通过。
                # 此脚本的混合任务需要网页正文及文件正文，必须确实取得两类工具证据。
                read_tools = {"fetch_content"} if payload["agent_name"] == "research" else {
                    "read_text_file", "read_file", "pdf_convert_to_markdown", "office_convert_to_markdown"}
                acquired[payload["agent_name"]] |= any(item.get("tool") in read_tools
                    for item in payload.get("evidence", []))
                delegated.append(message.name)
                incomplete |= payload["status"] != "success"
                # 运行验证日志仅输出状态和工具轨迹，不输出本地文件摘录。
                print("AGENT_RESULT=" + json.dumps({
                    "agent_name": payload["agent_name"], "task_id": payload["task_id"],
                    "status": payload["status"], "error": payload["error"],
                    "retrieval_status": payload.get("retrieval_status"),
                    "analysis_status": payload.get("analysis_status"),
                    "tools": payload["trace"].get("tools", []),
                    "parent_run_id": payload["trace"].get("parent_run_id"),
                }, ensure_ascii=False))
                assert payload["trace"].get("parent_run_id") == result["main_run_id"]
        print("FINAL=" + str(result["messages"][-1].content))
        assert "call_research_agent" in delegated and "call_file_agent" in delegated, delegated
        assert delegated.index("call_research_agent") < delegated.index("call_file_agent"), delegated
        assert all(acquired.values()), f"Missing actual source evidence: {acquired}"
        if incomplete:
            assert re.search(r"部分|不足|失败|无法|受限|限制|未能|缺口", str(result["messages"][-1].content))
        print("LIVE_COLLABORATION=PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--query", help="显式指定只读验证任务；仍使用隔离的 NoMemory")
    args = parser.parse_args()
    # 不带 --live 时只打印跳过说明；避免仅执行脚本就默认启动服务或发送网络请求。
    if args.live:
        asyncio.run(main(args.discover_only, args.query))
    else:
        print("SKIPPED: use --live to start MCP/Ollama and access network")
