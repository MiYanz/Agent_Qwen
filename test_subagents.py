# 离线多 Agent 回归：保留真实 StateGraph、ToolNode 和工具包装器，只替换模型及 MCP 后端。
# 测试中的固定回答用于验证调用链，不代表生产系统使用硬编码任务顺序；临时 SQLite 不涉及正式会话。
"""离线验证真实 Main ToolNode → 子图 → MCP 包装器 → ToolMessage 链路。"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

from agent.graph import create_graph
from agent.state import AgentState
from agent.roles import select_main_tools
from skills.loader import SkillLoader
from agent.subagents import AgentRegistry, AgentResult, Limits, SPECS, create_supervisor_tools_node, select_tools
from mcp_client.tools import load_mcp_tools, _prepare_arguments
from memory.models import MemoryOperationResult


# 构造带工具名、参数及调用 ID 的 AIMessage，模拟模型发出的结构化请求，不直接执行工具。
def request(name, args, ident="call-1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": ident, "type": "tool_call"}])


# 模拟 MCPManager 的最小接口，并记录真实包装器传来的调用，供断言检查路由和参数转换。
class FakeManager:
    # failed 控制 MCP isError；服务前缀沿用真实配置风格，calls 保存每次调用的参数副本。
    def __init__(self, failed=False):
        self.failed = failed
        self.calls = []
        self.servers = {name: {} for name in ("duckduckgo-search", "filesystem", "pdf", "office", "memory", "browser", "office-writer", "computer")}
        self.servers["pdf"]["toolPrefix"] = "pdf"
        self.servers["office"]["toolPrefix"] = "office"
        self.servers["office-writer"]["toolPrefix"] = "office"

    # 保留异步启动接口但不创建子进程，使真实 create_graph 能在离线环境继续运行。
    async def start(self):
        pass

    # 同时提供允许与禁止的工具定义，检验权限筛选确实依据来源和 Schema 执行。
    async def get_tools(self):
        rows = [
            ("duckduckgo-search", "search", "query"),
            ("duckduckgo-search", "fetch_content", "url"),
            ("filesystem", "read_text_file", "path"),
            ("filesystem", "write_file", "path"),
            ("pdf", "convert_to_markdown", "uri"),
            ("office", "convert_to_markdown", "uri"),
            ("office-writer", "create_docx", "output_path"),
            ("memory", "search_nodes", "query"),
            ("browser", "browser_evaluate", "function"),
            ("computer", "Snapshot", "query"),
        ]
        return [(server, SimpleNamespace(name=name, description="fake tool", inputSchema={
            "type": "object", "properties": {arg: {"type": "string"}}, "required": [arg]}))
            for server, name, arg in rows]

    # 记录服务、工具和参数后返回 SDK 风格对象；文本固定，不实际搜索或读取文件。
    async def call_tool(self, name, arguments, server_name):
        # 记录的是包装层实际传入的参数，因此能检查本地路径是否已经转成 file URI。
        self.calls.append((server_name, name, dict(arguments)))
        text = "URL: https://example.com/paper\nTitle: paper" if name == "search" else "有依据的文件或网页正文：技术路线采用权限隔离。"
        return SimpleNamespace(content=[SimpleNamespace(text=text)], isError=self.failed)


# 提供 bind_tools/ainvoke 两种模型接口，用不同 mode 确定性触发成功、失败、超时及循环。
class FakeModel:
    # 保存模式以及收到的消息和工具名，用于检查上下文隔离、工具暴露及调用顺序。
    def __init__(self, mode="normal"):
        self.mode = mode
        self.inputs = []
        self.bound_names = []

    # 根据绑定工具名区分主 Agent 与专业 Agent，返回模拟已绑定模型的对象。
    def bind_tools(self, tools):
        names = {tool.name for tool in tools}
        self.bound_names.append(names)
        model = self

        # 闭包捕获当前 tools 的名字和外层 model，模拟绑定工具后的独立调用对象。
        class Bound:
            # 先深拷贝输入供事后断言，再按测试模式返回工具请求或摘要。
            # 这里的 Research → File 顺序只是测试脚本，不会替换生产 Main 的动态模型决策。
            async def ainvoke(self, messages):
                # 必须保存消息深拷贝；否则被测代码后续修改同一对象可能让隔离断言失真。
                model.inputs.append([message.model_copy(deep=True) for message in messages])
                # 使用真正的异步等待制造超时，让被测 asyncio.timeout 有机会取消当前任务。
                if model.mode == "slow":
                    await asyncio.sleep(1)
                if model.mode == "raise":
                    # 错误正文刻意带私有信息标记，生产调度器只回传异常类型，避免原文进入结果。
                    raise RuntimeError("private content should not be logged")
                results = [m for m in messages if isinstance(m, ToolMessage)]
                if "call_research_agent" in names:
                    # 主模型每轮只参考最近用户输入后的结果，避免第二轮直接命中上一轮的 done 集合。
                    last_user = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
                    results = [m for m in messages[last_user:] if isinstance(m, ToolMessage)]
                    if model.mode == "direct":
                        return AIMessage(content="直接回答")
                    if model.mode == "main_mcp":
                        return (AIMessage(content="界面摘要") if results else
                                request("Snapshot", {"query": "window"}))
                    # Fake 根据已经收到的 ToolMessage 选择下一次请求，使测试经过真实工具结果回传，而非直接跳到最终答案。
                    done = {m.name for m in results}
                    if "call_research_agent" not in done:
                        return request("call_research_agent", {"task": "研究权限隔离"})
                    if "call_file_agent" not in done:
                        return request("call_file_agent", {"task": "分析本地说明", "context": "D:\\46-Agent\\README.md"})
                    return AIMessage(content="根据研究和文件证据比较，两者都采用权限隔离。")
                if model.mode == "unknown":
                    # 故意请求白名单之外的写工具，预期应在 ToolNode 被拦截，而不是进入 FakeManager。
                    return request("write_file", {"path": "forbidden"})
                if model.mode == "loop":
                    # 不管已收到什么结果都继续请求，用于证明工具和图步数上限确实能够终止循环。
                    return request("search", {"query": "loop"})
                if "search" in names:
                    if not results:
                        return request("search", {"query": "agent security"})
                    if len(results) == 1 and model.mode != "search_only":
                        return request("fetch_content", {"url": "https://example.com/paper"})
                elif not results:
                    return request("read_text_file", {"path": "D:\\46-Agent\\README.md"})
                return AIMessage(content=json.dumps({"status": "success", "summary": "证据表明技术路线采用权限隔离。"}, ensure_ascii=False))
        return Bound()

    # 未绑定工具的调用用于 Skill 等选择场景，固定 NONE，避免测试依赖真实 Ollama。
    async def ainvoke(self, messages):
        if any("重复工具请求已复用本次结果" in str(message.content) for message in messages):
            return AIMessage(content=json.dumps({"status": "success", "summary": "基于已读取正文分析：技术路线采用权限隔离。"}, ensure_ascii=False))
        return AIMessage(content="NONE")


# 模拟主图记忆接口，使用专门标记检查记忆是否泄露到子模型，不访问正式存储。
class FakeMemory:
    # 仅记录自动更新被调用的次数，用于确认子图没有重复触发主图记忆节点。
    def __init__(self):
        self.updates = 0

    # 默认表示普通任务而非显式记忆命令；需要命令分支的测试会单独替换此方法。
    async def handle_command(self, query):
        return MemoryOperationResult(handled=False)

    # 返回空记录，避免测试依赖用户已有记忆；render_context 另外提供可追踪标记。
    async def retrieve_memories(self, query):
        return []

    # 人为注入只应出现在主模型中的标记，后续扫描子模型输入确认没有默认透传。
    def render_context(self, records):
        return "PRIVATE_MEMORY_MARKER"

    # 只计数并返回 skipped，不写任何记忆；次数可以揭示子图是否误用了主工作流。
    async def process_automatic_memory(self, text):
        self.updates += 1
        return "skipped"


# IsolatedAsyncioTestCase 为各测试提供独立事件循环，避免前一用例资源状态干扰后一用例。
class SubagentTests(unittest.IsolatedAsyncioTestCase):
    # 把待测工具节点放入真正的最小 StateGraph；ToolNode 所需的运行时由图提供。
    def tool_graph(self, node):
        builder = StateGraph(AgentState)
        builder.add_node("tools", node)
        builder.set_entry_point("tools")
        builder.add_edge("tools", END)
        return builder.compile()

    # 每个用例新建 Fake 后端、真实 StructuredTool 和注册表，避免权限或调用记录串线。
    async def asyncSetUp(self):
        self.manager = FakeManager()
        self.tools = await load_mcp_tools(self.manager)
        self.model = FakeModel()
        self.registry = AgentRegistry(self.tools, self.model)

    # 验证两种角色可查找、工具入口齐全，且重复注册不能覆盖已有定义。
    async def test_registry_and_main_tools(self):
        self.assertEqual(self.registry.get("research"), SPECS[0])
        self.assertEqual(self.registry.get("file"), SPECS[1])
        self.assertEqual({t.name for t in self.registry.as_tools()}, {"call_research_agent", "call_file_agent"})
        with self.assertRaises(ValueError):
            self.registry.register(SPECS[0], [])

    # 同时验证模型可见列表和执行端白名单，不能只靠一条系统提示要求 Main 自觉委托。
    async def test_main_binding_excludes_professional_mcp(self):
        with patch("agent.nodes.llm", self.model):
            await create_graph(self.manager, memory_manager=FakeMemory())
        main_names = self.model.bound_names[0]
        self.assertTrue({"call_research_agent", "call_file_agent", "Snapshot", "office_create_docx"} <= main_names)
        self.assertFalse({"search", "fetch_content", "read_text_file", "pdf_convert_to_markdown",
                          "office_convert_to_markdown", "browser_evaluate"} & main_names)
        self.assertEqual(main_names, {t.name for t in select_main_tools(self.tools + self.registry.as_tools())})

    async def test_main_execution_rejects_raw_search_file_and_browser(self):
        # 故意把完整 MCP 列表交给执行节点，模拟旧初始化或旧检查点；节点仍必须重新筛选。
        node = self.tool_graph(create_supervisor_tools_node(self.tools + self.registry.as_tools(), self.registry))
        for name, arguments in (("search", {"query": "news"}), ("fetch_content", {"url": "https://example.com"}),
                                ("read_text_file", {"path": "D:\\paper.txt"}),
                                ("pdf_convert_to_markdown", {"uri": "D:\\paper.pdf"}),
                                ("office_convert_to_markdown", {"uri": "D:\\slides.pptx"}),
                                ("browser_evaluate", {"function": "fetch('https://example.com')"})):
            result = await node.ainvoke({"messages": [request(name, arguments)], "main_run_id": "root-id"})
            payload = json.loads(result["messages"][-1].content)
            self.assertEqual(payload["error"], "main_tool_forbidden")
            self.assertEqual(result["agent_events"][0]["event_type"], "tool_blocked")
        self.assertFalse(self.manager.calls)

    async def test_plain_file_analysis_must_delegate(self):
        with patch("agent.nodes.llm", FakeModel("direct")):
            graph = await create_graph(self.manager, memory_manager=FakeMemory())
            result = await graph.ainvoke({"messages": [HumanMessage(content=r"分析 D:\46-Agent\README.md 的架构")]})
        replies = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual([m.name for m in replies], ["call_file_agent"])
        self.assertEqual(json.loads(replies[0].content)["trace"]["tools"][0]["tool"], "read_text_file")

    async def test_local_file_search_does_not_trigger_research(self):
        with patch("agent.nodes.llm", FakeModel("direct")):
            graph = await create_graph(self.manager, memory_manager=FakeMemory())
            result = await graph.ainvoke({"messages": [HumanMessage(content="搜索本地文件夹中的说明文件")]})
        self.assertEqual([m.name for m in result["messages"] if isinstance(m, ToolMessage)], ["call_file_agent"])
        self.assertFalse(any(server == "duckduckgo-search" for server, _, _ in self.manager.calls))

    async def test_output_path_does_not_force_file_read(self):
        from agent.nodes import _pending_specialists
        pending = _pending_specialists({"messages": [HumanMessage(content=r"创建 D:\reports\new.docx")],
                                        "file_route": "office"})
        self.assertEqual(pending, [])

    async def test_research_owns_news_skill_and_internal_search(self):
        loader = SkillLoader(Path(__file__).parent / "skills")
        loader.discover()
        registry = AgentRegistry(self.tools, self.model, skill_loader=loader)
        result = await registry.dispatch("research", {"task": "搜索今天新闻并读取正文", "parent_run_id": "root-id"})
        self.assertEqual(result["status"], "success")
        events = result["trace"]["events"]
        self.assertTrue(any(e["event_type"] == "skill_loaded" and e["target"] == "news-search" for e in events))
        self.assertTrue(all(e["agent_name"] == "research" for e in events if e["event_type"].startswith("mcp_tool_")))
        self.assertTrue(any("当前 Skill：news-search" in str(message.content)
                            for inputs in self.model.inputs for message in inputs))

    async def test_main_does_not_load_specialist_skills(self):
        from agent.nodes import create_skill_node
        loader = SkillLoader(Path(__file__).parent / "skills")
        loader.discover()
        for query, route in (("搜索今天新闻", "general"), (r"总结 D:\slides.pptx", "office")):
            selected = await create_skill_node(loader)(
                {"messages": [HumanMessage(content=query)], "file_route": route})
            self.assertEqual(selected["active_skill"], "")
            self.assertEqual(selected["skill_instruction"], "")

    async def test_file_owns_ppt_skill(self):
        loader = SkillLoader(Path(__file__).parent / "skills")
        loader.discover()
        registry = AgentRegistry(self.tools, self.model, skill_loader=loader)
        result = await registry.dispatch("file", {"task": r"总结 D:\slides.pptx"})
        self.assertTrue(any(e["event_type"] == "skill_loaded" and e["target"] == "ppt-summary"
                            and e["agent_name"] == "file" for e in result["trace"]["events"]))

    async def test_trace_has_parent_ids_and_complete_order(self):
        with patch("agent.nodes.llm", self.model):
            graph = await create_graph(self.manager, memory_manager=FakeMemory())
            updates = [update async for update in graph.astream({"messages": [HumanMessage(content="搜索论文再分析本地文件")]})]
        events = [event for update in updates for output in update.values() for event in output.get("agent_events", [])]
        root_id = next(e["task_id"] for e in events if e["event_type"] == "main_reasoning_started")
        children = [e for e in events if e["event_type"] == "delegation_started"]
        self.assertEqual([e["target"] for e in children], ["research", "file"])
        for child in children:
            chain = [e for e in events if e["task_id"] == child["task_id"]]
            self.assertTrue(all(e["parent_run_id"] == root_id for e in chain))
            types = [e["event_type"] for e in chain]
            self.assertLess(types.index("delegation_started"), types.index("agent_started"))
            self.assertLess(types.index("agent_started"), types.index("mcp_tool_started"))
            self.assertLess(types.index("mcp_tool_finished"), types.index("agent_finished"))
            self.assertLess(types.index("agent_finished"), types.index("delegation_finished"))
        # 专业 Agent 返回后确实还有 Main 的推理节点，而不只是伪造一个“返回主 Agent”的展示文本。
        final_child = max(i for i, e in enumerate(events) if e["event_type"] == "delegation_finished")
        self.assertTrue(any(e["event_type"] == "main_reasoning_started" for e in events[final_child + 1:]))

    # 除核对正常白名单，还模拟 Schema 增加写入参数或丢失，确认默认拒绝不明权限。
    async def test_exact_permissions(self):
        self.assertEqual({t.name for t in self.registry.tools_for("research")}, {"search", "fetch_content"})
        self.assertEqual({t.name for t in self.registry.tools_for("file")},
                         {"read_text_file", "pdf_convert_to_markdown", "office_convert_to_markdown"})
        changed = self.tools[0].model_copy(deep=True)
        changed.args_schema["properties"]["output_path"] = {"type": "string"}
        self.assertEqual(select_tools("research", [changed]), [])
        changed.args_schema = None
        self.assertEqual(select_tools("research", [changed]), [])

    # 验证二进制文档拒绝纯文本读取，以及路径转换和缺失文件失败。
    # 临时 PDF 仅用于路径检查，Fake MCP 不执行真实 PDF 解析。
    async def test_document_wrapper_keeps_path_guards(self):
        by_name = {t.name: t for t in self.tools}
        with self.assertRaises(ValueError):
            _prepare_arguments("filesystem", "read_text_file", {"path": "D:\\paper.pdf"})
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "test.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            await by_name["pdf_convert_to_markdown"].ainvoke({"uri": str(pdf)})
            self.assertEqual(self.manager.calls[-1][2]["uri"], pdf.resolve().as_uri())
            missing = await by_name["pdf_convert_to_markdown"].ainvoke({"uri": str(pdf.with_name("missing.pdf"))})
            self.assertEqual(json.loads(missing)["status"], "failed")

    # 分别执行两类子图，校验统一结果、证据和轨迹，并确认初始任务消息不是同一个对象。
    async def test_success_trace_and_fresh_context(self):
        for name in ("research", "file"):
            result = await self.registry.dispatch(name, {"task": "分析", "task_id": name})
            AgentResult.model_validate(result)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["task_id"], name)
            self.assertEqual(result["trace"]["target"], name)
            self.assertTrue(result["trace"]["tools"])
            self.assertTrue(result["evidence"])
            self.assertEqual(result["artifacts"], [])
        # 系统提示可以增加，但每个角色第一次模型调用都只有自己的任务、没有工具历史。
        first_calls = [items for items in self.model.inputs
                       if not any(isinstance(message, ToolMessage) for message in items)]
        self.assertEqual(len(first_calls), 2)
        tasks = [next(message for message in items if isinstance(message, HumanMessage))
                 for items in first_calls]
        self.assertIsNot(tasks[0], tasks[1])

    # 覆盖未知目标、空任务、额外字段、上下文超长，以及没有受限工具可用的情况。
    async def test_failures_and_invalid_parameters(self):
        cases = [("missing", {"task": "test"}, "unknown_agent"),
                 ("file", {"task": " "}, "invalid_parameters"),
                 ("file", {"task": "test", "unexpected": True}, "invalid_parameters"),
                 ("file", {"task": "test", "context": "x" * 6001}, "invalid_parameters")]
        for name, payload, error in cases:
            result = await self.registry.dispatch(name, payload)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error"], error)
        registry = AgentRegistry([], self.model)
        self.assertEqual((await registry.dispatch("file", {"task": "test"}))["error"], "no_permitted_tools_available")

    # 用慢模型、抛异常模型和无限请求模型，分别验证超时、错误、工具次数及图递归上限。
    async def test_timeout_exception_and_limits(self):
        for mode, limits, error in [
            ("slow", Limits(timeout=.01), "timeout"),
            ("raise", Limits(), "RuntimeError"),
            ("loop", Limits(max_steps=2), "GraphRecursionError"),
        ]:
            registry = AgentRegistry(self.tools, FakeModel(mode), limits)
            result = await registry.dispatch("research", {"task": "test"})
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error"], error)

    # 已成功取到证据后才触及限额属于部分完成；验证证据保留，而非放宽成功判定。
    async def test_interrupted_work_preserves_evidence(self):
        for limits, error in [(Limits(max_tools=1), "tool_call_limit"),
                              (Limits(max_steps=4), "GraphRecursionError")]:
            result = await AgentRegistry(self.tools, FakeModel("loop"), limits).dispatch(
                "research", {"task": "test"})
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["error"], error)
            self.assertTrue(result["evidence"])
            self.assertIn("尚未完成综合分析", result["summary"])
        # 两种上限场景各只执行过一次 MCP，之后的中断不能丢掉这份证据。
        self.assertEqual(len(self.manager.calls), 2)

    async def test_missing_tools_explains_server_health(self):
        from mcp_client.health import MCPHealth
        health = MCPHealth()
        health.update("duckduckgo-search", "unhealthy", "private diagnostic not for prompt")
        result = await AgentRegistry([], self.model, server_health=health).dispatch("research", {"task": "搜索"})
        self.assertEqual(result["error"], "no_permitted_tools_available")
        self.assertIn("duckduckgo-search=unhealthy", result["summary"])
        self.assertNotIn("private diagnostic", json.dumps(result))
        self.assertFalse(self.manager.calls)

    # 复现 File 模型第一次直接返回空字符串：程序仍先读指定文件，再有限修复报告。
    async def test_blank_file_report_reads_and_repairs_once(self):
        class BlankModel(FakeModel):
            def __init__(self, valid=True):
                super().__init__()
                self.valid = valid
                self.repairs = 0

            def bind_tools(self, tools):
                return SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content="")))

            def bind(self, **kwargs):
                self.assert_schema = kwargs["format"]
                self.repairs += 1
                text = json.dumps({"status": "success", "summary": "正文表明采用权限隔离。"}) if self.valid else ""
                return SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content=text)))

        for valid, status, error in [(True, "success", None), (False, "partial", "analysis_output_invalid")]:
            model = BlankModel(valid)
            result = await AgentRegistry(self.tools, model).dispatch("file", {
                "task": "分析状态隔离", "context": r"D:\46-Agent\README.md"})
            self.assertEqual(result["status"], status)
            self.assertEqual(result["error"], error)
            self.assertEqual(result["retrieval_status"], "success")
            self.assertEqual(result["analysis_status"], "success" if valid else "failed")
            self.assertEqual(model.repairs, 1)
            self.assertEqual(result["evidence"][0]["tool"], "read_text_file")
        self.assertEqual(len(self.manager.calls), 2)

    async def test_requested_url_is_read_without_search(self):
        result = await self.registry.dispatch("research", {
            "task": "读取 https://example.com/paper，比较状态隔离。不要搜索其他文档。"})
        self.assertTrue(result["evidence"])
        self.assertEqual([call[1] for call in self.manager.calls], ["fetch_content"])

    async def test_supervisor_restores_missing_input_path(self):
        graph = self.tool_graph(create_supervisor_tools_node(self.registry.as_tools(), self.registry))
        result = await graph.ainvoke({"messages": [HumanMessage(content=r"分析 D:\project\specified.md"),
            request("call_file_agent", {"task": "分析文件状态隔离"})]})
        payload = json.loads(result["messages"][-1].content)
        self.assertEqual(self.manager.calls[0][2]["path"], r"D:\project\specified.md")
        self.assertEqual(payload["evidence"][0]["source"], r"D:\project\specified.md")

    async def test_analysis_partial_is_not_retrieval_failure(self):
        class PartialModel(FakeModel):
            def bind_tools(self, tools):
                return SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content=json.dumps({
                    "status": "partial", "summary": "读取了正文，但分析尚缺少更多细节。"}))))
        result = await AgentRegistry(self.tools, PartialModel()).dispatch("research", {
            "task": "读取 https://example.com/paper"})
        self.assertEqual(result["retrieval_status"], "success")
        self.assertEqual(result["analysis_status"], "partial")
        self.assertEqual(result["error"], "analysis_incomplete")

    async def test_main_discloses_limit_only_once(self):
        from agent.nodes import create_chat_node
        bound = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(
            content="已有证据可以支持部分结论。\n\n执行限制：旧说明\n\n执行限制：旧说明")))
        model = SimpleNamespace(bind_tools=lambda tools: bound)
        payload = {"agent_name": "research", "status": "partial", "error": "analysis_incomplete",
                   "evidence": [{"tool": "fetch_content", "source": "https://example.com/paper"}]}
        with patch("agent.nodes.llm", model):
            node = create_chat_node(self.registry.as_tools(), Limits())
            result = await node({"messages": [HumanMessage(content="继续综合"),
                request("call_research_agent", {"task": "研究"}),
                ToolMessage(content=json.dumps(payload), name="call_research_agent", tool_call_id="call-1")]})
        self.assertEqual(result["messages"][-1].content.count("执行限制："), 1)
        self.assertIn("已实际读取", str(bound.ainvoke.call_args.args[0]))

    async def test_repeated_read_reuses_result_and_finishes_analysis(self):
        class RepeatRead(FakeModel):
            def bind_tools(self, tools):
                class Bound:
                    async def ainvoke(inner, messages):
                        count = sum(isinstance(message, ToolMessage) for message in messages)
                        return request("read_text_file", {"path": r"D:\46-Agent\README.md"}, f"read-{count}")
                return Bound()

            async def ainvoke(self, messages):
                self.final_messages = messages
                return await super().ainvoke(messages)

        model = RepeatRead()
        result = await AgentRegistry(self.tools, model).dispatch("file", {"task": "分析状态隔离"})
        self.assertEqual(result["status"], "success")
        self.assertIn("权限隔离", result["summary"])
        self.assertEqual(len(self.manager.calls), 1)
        self.assertEqual(len(result["trace"]["tools"]), 1)
        self.assertTrue(any(event["event_type"] == "tool_reused" for event in result["trace"]["events"]))
        replies = [message for message in model.final_messages if isinstance(message, ToolMessage)]
        self.assertEqual(len(replies), 2)
        self.assertEqual({message.tool_call_id for message in replies}, {"read-0", "read-1"})
        self.assertEqual(len({message.id for message in replies}), 2)

    async def test_web_paths_never_reach_filesystem(self):
        for path in ("https://langgraph.github.io", r"D:\https:\langgraph.github.io"):
            with self.assertRaisesRegex(ValueError, "本地路径"):
                _prepare_arguments("filesystem", "search_files", {"path": path, "pattern": "README"})
            # 走真实 StructuredTool 包装器也必须在请求 MCP 前失败，而非仅验证辅助函数。
            reader = next(tool for tool in self.tools if tool.name == "read_text_file")
            result = json.loads(await reader.ainvoke({"path": path}))
            self.assertEqual(result["status"], "failed")
        self.assertFalse(self.manager.calls)
        self.assertEqual(_prepare_arguments("filesystem", "search_files", {
            "path": r"D:\46-Agent", "pattern": "README"})["path"], r"D:\46-Agent")

    async def test_timeout_keeps_completed_tool_evidence(self):
        original = self.manager.call_tool

        async def slow_fetch(name, arguments, server_name):
            if name == "fetch_content":
                await asyncio.sleep(5)
            return await original(name, arguments, server_name)

        self.manager.call_tool = slow_fetch
        # 给真实图初始化留出时间，避免并行运行本地模型时 50ms 在首个工具前就超时。
        # 正文工具仍远慢于预算，因此实际异步取消与证据保留的断言保持不变。
        result = await AgentRegistry(self.tools, self.model, Limits(timeout=2)).dispatch(
            "research", {"task": "test"})
        self.assertEqual((result["status"], result["error"]), ("partial", "timeout"))
        self.assertEqual(result["evidence"][0]["tool"], "search")
        self.assertEqual(result["trace"]["tools"][-1]["status"], "failed")

    # 复现日志：Main 在收到 partial 后不断改写 task。真实子图应只执行两次研究，
    # 然后继续 File 并回答，不能因任务文字变化而把六次预算全部用在同一个故障上。
    async def test_reworded_partial_retries_finish_and_continue_file(self):
        class RetryingModel(FakeModel):
            def bind_tools(self, tools):
                if "call_research_agent" not in {tool.name for tool in tools}:
                    return super().bind_tools(tools)

                class Main:
                    async def ainvoke(inner, messages):
                        count = sum(isinstance(m, ToolMessage) for m in messages)
                        return request("call_research_agent", {"task": f"改写研究任务 {count}"}, f"retry-{count}")
                return Main()

            async def ainvoke(self, messages):
                # Skill 选择仍返回 NONE；只有收尾指令使用未绑定工具的同一客户端。
                if "本轮恢复性委托" in str(messages[-1].content):
                    return AIMessage(content="根据已取得的研究与文件证据进行有限比较。")
                return AIMessage(content="NONE")

        original = self.manager.call_tool

        async def failing_fetch(name, arguments, server_name):
            if name == "fetch_content":
                return SimpleNamespace(content=[SimpleNamespace(text="Error: DNS unavailable")], isError=False)
            return await original(name, arguments, server_name)

        self.manager.call_tool = failing_fetch
        with patch("agent.nodes.llm", RetryingModel()):
            graph = await create_graph(self.manager, memory_manager=FakeMemory())
            result = await graph.ainvoke({"messages": [HumanMessage(
                content=r"搜索官方文档，读取 D:\46-Agent\README.md 并比较")]}, {"recursion_limit": 40})
        replies = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual([m.name for m in replies],
            ["call_research_agent", "call_research_agent", "call_file_agent"])
        self.assertEqual(result["delegation_count"], 3)
        self.assertIn("有限比较", result["messages"][-1].content)
        self.assertIn("执行限制", result["messages"][-1].content)

    async def test_executor_blocks_reworded_incomplete_retries(self):
        prior = [HumanMessage(content="搜索研究")]
        for i in range(2):
            prior.append(ToolMessage(content=json.dumps({"status": "partial", "agent_name": "research"}),
                                     name="call_research_agent", tool_call_id=str(i)))
        prior.append(request("call_research_agent", {"task": "使用不同文字再次研究"}))
        graph = self.tool_graph(create_supervisor_tools_node(self.registry.as_tools(), self.registry))
        result = await graph.ainvoke({"messages": prior})
        self.assertEqual(json.loads(result["messages"][-1].content)["error"], "incomplete_retry_limit")
        self.assertFalse(self.manager.calls)

    # 即使模型声称成功，只要实际工具全部失败，包装器也必须返回 failed 并记录失败轨迹。
    async def test_tool_errors_never_success(self):
        self.manager.failed = True
        result = await self.registry.dispatch("research", {"task": "test"})
        self.assertEqual(result["status"], "failed")
        self.assertTrue(all(t["status"] == "failed" for t in result["trace"]["tools"]))

    # 搜索候选列表不等于网页正文；缺少 fetch_content 的研究最多只能部分完成。
    async def test_search_without_body_is_partial(self):
        # 没有正文获取能力时仍保留 partial 判定；具备 fetch_content 时现在会主动读取候选。
        registry = AgentRegistry([tool for tool in self.tools if tool.name != "fetch_content"], FakeModel("search_only"))
        result = await registry.dispatch("research", {"task": "test"})
        self.assertEqual(result["status"], "partial")

    # 用 AsyncMock 返回非空的“无搜索结果”文字，检查它不会被当作有效证据。
    async def test_empty_search_never_success(self):
        self.manager.call_tool = AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(text="No results were found for your search query.")], isError=False))
        result = await self.registry.dispatch("research", {"task": "test"})
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["evidence"])

    # 保留搜索成功，只替换正文获取为错误，验证有部分依据时返回 partial。
    async def test_partial_tool_failure(self):
        original = self.manager.call_tool

        # 只拦截 fetch_content，其他调用继续用原 FakeManager，构造可控的混合成功场景。
        async def call(name, arguments, server_name):
            if name == "fetch_content":
                return SimpleNamespace(content=[SimpleNamespace(text="Error: timed out")], isError=False)
            return await original(name, arguments, server_name)

        self.manager.call_tool = call
        result = await self.registry.dispatch("research", {"task": "test"})
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["trace"]["tools"][-1]["status"], "failed")

    # 让子模型请求写文件，确认 ToolNode 拒绝且调用未到达 MCP 后端。
    async def test_unknown_child_tool_is_denied(self):
        registry = AgentRegistry(self.tools, FakeModel("unknown"), Limits(max_tools=1))
        result = await registry.dispatch("file", {"task": "test"})
        self.assertEqual(result["status"], "failed")
        self.assertFalse(self.manager.calls)
        self.assertEqual(result["trace"]["tools"][0]["status"], "failed")

    # 把前一次返回的计数放回图状态，检查重复任务和总委托次数两种限制。
    async def test_delegation_budget_and_repeat(self):
        registry = AgentRegistry(self.tools, self.model, Limits(max_delegations=3, max_same_task=1))
        node = create_supervisor_tools_node(self.tools + registry.as_tools(), registry)
        node = self.tool_graph(node).ainvoke
        state = {"messages": [request("call_file_agent", {"task": "test"})]}
        one = await node(state)
        two = await node({**one, "messages": state["messages"]})
        self.assertEqual(json.loads(two["messages"][-1].content)["error"], "duplicate_call_limit")
        over = await node({"messages": [request("call_file_agent", {"task": "new"})], "delegation_count": 3})
        self.assertEqual(json.loads(over["messages"][-1].content)["error"], "delegation_limit")

    # 从主图真实工具节点传入空任务，检查输入错误仍按 Agent 结果协议返回。
    async def test_invalid_wrapper_params_return_protocol(self):
        node = create_supervisor_tools_node(self.registry.as_tools(), self.registry)
        node = self.tool_graph(node).ainvoke
        result = await node({"messages": [request("call_file_agent", {"task": ""})]})
        self.assertEqual(json.loads(result["messages"][-1].content)["error"], "invalid_parameters")

    # 同一 AIMessage 中发出多次委托，检查顺序消费预算；再验证重复任务 ID 被拒绝。
    async def test_duplicate_id_and_multiple_calls_budget(self):
        registry = AgentRegistry(self.tools, self.model, Limits(max_delegations=1))
        node = self.tool_graph(create_supervisor_tools_node(registry.as_tools(), registry))
        first = request("call_file_agent", {"task": "one"}, "a").tool_calls[0]
        second = request("call_research_agent", {"task": "two"}, "b").tool_calls[0]
        result = await node.ainvoke({"messages": [AIMessage(content="", tool_calls=[first, second])]})
        replies = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(json.loads(replies[0].content)["status"], "success")
        self.assertEqual(json.loads(replies[1].content)["error"], "delegation_limit")
        node = self.tool_graph(create_supervisor_tools_node(self.registry.as_tools(), self.registry))
        one = await node.ainvoke({"messages": [request("call_file_agent", {"task": "one", "task_id": "same"})]})
        two = await node.ainvoke({**one, "messages": [request("call_file_agent", {"task": "two", "task_id": "same"})]})
        self.assertEqual(json.loads(two["messages"][-1].content)["error"], "duplicate_task_id")

    # 直接回答和非专业的 Snapshot 操作仍可执行；文件读取已改为 File 专属，不能再用它测试 Main 直调。
    async def test_main_direct_answer_and_allowed_mcp_loop(self):
        for mode, expected in (("direct", "直接回答"), ("main_mcp", "界面摘要")):
            model = FakeModel(mode)
            with patch("agent.nodes.llm", model):
                graph = await create_graph(self.manager, memory_manager=FakeMemory())
                result = await graph.ainvoke({"messages": [HumanMessage(content="查看当前窗口")]})
            self.assertEqual(result["messages"][-1].content, expected)
            self.assertEqual(result["delegation_count"], 0)
            if mode == "main_mcp":
                self.assertEqual(result["messages"][-2].name, "Snapshot")

    # 普通搜索必须先经过委托；search/fetch_content 只出现在 Research 的内部轨迹和 Fake 后端。
    async def test_ordinary_search_uses_research_and_reads_body(self):
        with patch("agent.nodes.llm", FakeModel("direct")):
            graph = await create_graph(self.manager, memory_manager=FakeMemory())
            result = await graph.ainvoke({"messages": [HumanMessage(content="搜索最新公开资料")]})
        names = [m.name for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(names, ["call_research_agent"])
        payload = json.loads(next(m.content for m in result["messages"] if isinstance(m, ToolMessage)))
        self.assertEqual([event["tool"] for event in payload["trace"]["tools"]], ["search", "fetch_content"])
        self.assertEqual(payload["agent_name"], "research")

    # 用户明确要求专业 Agent 时，搜索规则不能抢先把任务改为普通主图搜索。
    async def test_explicit_delegation_not_overridden_by_search_rule(self):
        with patch("agent.nodes.llm", self.model):
            graph = await create_graph(self.manager, memory_manager=FakeMemory())
            result = await graph.ainvoke({"messages": [HumanMessage(content="请 Research Agent 搜索公开资料再分析文件")]})
        names = [m.name for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(names, ["call_research_agent", "call_file_agent"])

    # 把命令处理结果设为已处理，验证图直接回复可信操作结果，不调用模型或自动记忆更新。
    async def test_memory_command_still_bypasses_model(self):
        memory = FakeMemory()
        memory.handle_command = AsyncMock(return_value=MemoryOperationResult(handled=True, response="记忆结果"))
        with patch("agent.nodes.llm", self.model):
            graph = await create_graph(self.manager, memory_manager=memory)
            result = await graph.ainvoke({"messages": [HumanMessage(content="查看记忆")]})
        self.assertEqual(result["messages"][-1].content, "记忆结果")
        self.assertEqual(memory.updates, 0)
        self.assertFalse(self.model.inputs)

    # 用真实 app.chat 异步生成器和 Fake 图，检查最终页面含子 Agent、内部 MCP 和结束状态。
    async def test_gradio_nested_trace(self):
        import app

        with patch("agent.nodes.llm", self.model):
            graph = await create_graph(self.manager, memory_manager=FakeMemory())
            # 只替换应用资源获取入口，app.chat、异步更新迭代和页面格式化仍走生产代码。
            with patch.object(app, "get_agent", AsyncMock(return_value=graph)), patch.object(app, "_checkpointer", None):
                updates = [value async for value in app.chat("比较本地与网络", [], "trace-test")]
        self.assertIn("Main Agent | delegation_started | 发起委托 → research", updates[-1])
        self.assertIn("Research Agent | mcp_tool_started | 调用 MCP Tool → fetch_content", updates[-1])
        self.assertIn("Research Agent | agent_finished | 返回子任务结果 → main | success", updates[-1])
        self.assertIn("Main Agent | main_reasoning_started | 继续推理", updates[-1])
        self.assertIn("parent_run_id=", updates[-1])
        self.assertIn("比较", updates[-1])

    # 使用临时 SQLite 验证主图完整协作、消息恢复、会话隔离、子图无独立持久化及下一轮预算重置。
    async def test_full_graph_sqlite_isolation_and_recovery(self):
        memory = FakeMemory()
        # 专门放置只属于主会话的标记，之后在每次子模型输入里搜索，检查没有隐式透传完整历史。
        history = HumanMessage(content="PARENT_HISTORY_MARKER")
        with tempfile.TemporaryDirectory() as directory:
            async with AsyncSqliteSaver.from_conn_string(str(Path(directory) / "test.sqlite")) as saver:
                with patch("agent.nodes.llm", self.model):
                    graph = await create_graph(self.manager, memory_manager=memory, checkpointer=saver)
                    config = {"configurable": {"thread_id": "parent"}, "recursion_limit": 60}
                    result = await graph.ainvoke({"messages": [history]}, config)
                    self.assertIn("比较", result["messages"][-1].content)
                    results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
                    self.assertEqual([m.name for m in results], ["call_research_agent", "call_file_agent"])
                    self.assertEqual(result["delegation_count"], 2)
                    for inputs in self.model.inputs:
                        for message in inputs:
                            if isinstance(message, ToolMessage) and message.name in {"call_research_agent", "call_file_agent"}:
                                # 模型只需要结果，不需要全部 UI 事件；持久化原消息仍应有完整轨迹。
                                self.assertNotIn("events", json.loads(message.content).get("trace", {}))
                        if any(isinstance(m, HumanMessage) and '"task":' in str(m.content) for m in inputs):
                            text = str(inputs)
                            self.assertNotIn("PARENT_HISTORY_MARKER", text)
                            self.assertNotIn("PRIVATE_MEMORY_MARKER", text)
                    self.assertEqual(memory.updates, 1)
                    snapshot = await graph.aget_state(config)
                    self.assertEqual(len(snapshot.values["messages"]), len(result["messages"]))
                    other = await graph.aget_state({"configurable": {"thread_id": "other"}})
                    self.assertFalse(other.values)
                    # 检查所有已保存 checkpoint 的命名空间，确保子图没有借用父图 saver 写入嵌套检查点。
                    checkpoints = [item async for item in saver.alist(None)]
                    self.assertTrue(all(item.config["configurable"].get("checkpoint_ns", "") == ""
                                        for item in checkpoints))
                    # 同一会话下一轮预算重置；子图无历史，仍从 search/read 开始。
                    next_turn = await graph.ainvoke({"messages": [HumanMessage(content="第二轮")]}, config)
                    self.assertEqual(next_turn["delegation_count"], 2)

    # 让全部 MCP 返回错误，断言 Main 仍执行后续委托并生成回答，而且程序补充实际失败限制。
    async def test_child_failure_main_continues(self):
        self.manager.failed = True
        with patch("agent.nodes.llm", self.model):
            graph = await create_graph(self.manager, memory_manager=FakeMemory())
            result = await graph.ainvoke({"messages": [HumanMessage(content="比较文件和网络")]})
        results = [json.loads(m.content) for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual([r["status"] for r in results], ["failed", "failed"])
        self.assertIsInstance(result["messages"][-1], AIMessage)
        self.assertIn("执行限制：research Agent：failed", result["messages"][-1].content)
        self.assertIn("file Agent：failed", result["messages"][-1].content)


if __name__ == "__main__":
    unittest.main()
