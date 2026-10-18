import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

sys.dont_write_bytecode = True
# 禁止把字节码缓存成 .pyc 文件，保持项目目录干净

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent.graph import create_graph
from mcp_client.manager import MCPManager
from mcp_client.tools import CONFIG_PATH
from memory.manager import MemoryManager


MODEL_NAME = "Qwen3-8B"
# 展示用模型名：主要给用户看，出现在最终回答的“执行信息”里。
# 它不等于 Ollama 内部真正使用的模型标识。
OLLAMA_MODEL = "qwen3:8b"
# Ollama 实际模型标识：用于检查本地是否已安装该模型。
# 调用 Ollama / LangChain 时通常也需要这个带 tag 的名字。
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"
# 本地 Ollama 的模型列表 API 地址。
# 通过它检查 Ollama 是否启动、已安装哪些模型。
EXECUTION_INFO_MARKER = "\n\n---\n### 执行信息"
# 自定义分隔标记：用来把“最终回答”和“执行信息”分开。
# 最终展示时拼在回答后面；写入对话历史前会用它把执行信息剥离。
PROJECT_ROOT = Path(__file__).resolve().parent
SESSION_DB_PATH = PROJECT_ROOT / "memory" / "conversations.sqlite3"


# 这些模块级变量共同保存应用运行期间可复用的 Agent、MCP、Memory 和初始化状态。
_agent = None   # 此时没有 Agent 可用，所以用 None 表示“尚未就绪”。
_agent_init_task = None # 保存初始化 Agent 的异步任务。
_mcp_manager = MCPManager(CONFIG_PATH)  # 创建一个 MCP 管理器实例。
_memory_manager = MemoryManager(_mcp_manager)
_checkpointer = None
_runtime_ready = None   # “运行时初始化完成”的信号
_runtime_error = None   # 保存初始化过程中发生的异常


def log_event(label, detail=None):
    """打印日志"""
    print(f"\n[{label}]", flush=True)
    # \n 是先空一行，免得跟上面的字挤在一起。[{label}] 就是把标题套上方括号。flush=True 意思是：立刻显示出来，不要等。
    if detail:
        print(detail, flush=True)
        # 如果填了详细内容，就再打印一行。


def is_port_available(host, port):
    """看端口有没有被占用"""
    # with 是同步上下文管理器：离开代码块时会自动关闭临时 socket。
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def _ollama_models():
    """返回 Ollama 已安装的模型，而且不走系统代理。"""
    opener = build_opener(ProxyHandler({})) 
    # 代理设置是空的，不走代理。 Ollama 就在你电脑本地，走代理反而可能连不上。

    with opener.open(OLLAMA_TAGS_URL, timeout=2) as response:
        # 打开网址http://127.0.0.1:11434/api/tags
        payload = json.loads(response.read().decode("utf-8"))
        # 先按 UTF-8 解码成文字，再当成 JSON 解析。
    return {model.get("name") for model in payload.get("models", [])}
    # 输出已安装模型


def ensure_ollama_ready():
    """保证 Ollama 已经准备好，能用。"""
    try:
        # 看 Ollama 是不是已经开着
        models = _ollama_models()
    except (OSError, TimeoutError, URLError, json.JSONDecodeError): # 问不到就报错
        # 在系统里找找有没有叫 ollama 的程序
        executable = shutil.which("ollama")
        if not executable:  # 去电脑默认路径碰碰运气
            local_app_data = os.environ.get("LOCALAPPDATA")
            if local_app_data:
                candidate = (
                    Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
                )
                if candidate.is_file():
                    executable = str(candidate)
        if not executable:
            raise RuntimeError(
                "无法连接 Ollama，且未找到 ollama 可执行文件。请先安装或启动 Ollama。"
            )
        # 启动 Ollama
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            [executable, "serve"],          # 在命令行敲 ollama serve
            stdout=subprocess.DEVNULL,      # 正常输出丢掉
            stderr=subprocess.DEVNULL,      # 错误输出也丢掉
            creationflags=creation_flags,   # Windows 下别弹黑框。
        )
        deadline = time.monotonic() + 30    # 从现在起，最多等 30 秒。 
        last_error = None
        # 只要还没到 30 秒，就每隔 0.5 秒问一次 Ollama：“你好了没？”
        while time.monotonic() < deadline:
            try:
                models = _ollama_models()
                break
            except (OSError, TimeoutError, URLError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(0.5)
        else:
            raise RuntimeError(
                "Ollama 启动后 30 秒内未就绪。请运行 `ollama serve` 检查错误。"
            ) from last_error
    # 看看 models 这个集合里有没有 qwen3:8b
    # 没有的话，报错，告诉用户去敲 ollama pull qwen3:8b 把它下载下来。
    if OLLAMA_MODEL not in models:
        raise RuntimeError(
            f"Ollama 中未安装 {OLLAMA_MODEL}，请运行 `ollama pull {OLLAMA_MODEL}`。"
        )

# async 异步的意思就是：它里面的事情可以等，不着急，可以一边等一边干别的
async def _initialize_runtime():
    """在当前事件循环中按需初始化 Ollama、MCP 和 LangGraph Agent。"""
    # 接下来要修改外面的 _agent 和 _agent_init_task 这两个全局变量
    global _agent, _agent_init_task
    # asyncio.to_thread 的意思是：把它丢到另一个线程去跑，别堵在这里。
    # await 的意思是：我在这里等它跑完，但等的时候不占地方，别人可以干别的。
    await asyncio.to_thread(ensure_ollama_ready)
    # 启动 MCP 工具管理器。
    # MCP 就是各种工具，比如搜网页、看文件、开浏览器。
    await _mcp_manager.start()
    # 先看看是不是已经造过了。造过就不重复造。
    if _agent is None:
        # 根据 MCP 工具，造一个 LangGraph Agent。
        _agent = await create_graph(
            _mcp_manager,
            memory_manager=_memory_manager,
            checkpointer=_checkpointer,
        )
    return _agent


async def get_agent():
    """返回应用生命周期里初始化好的 Agent，如果没有就懒加载一个"""
    global _agent, _agent_init_task
    if _agent is not None:
        return _agent

    # 应用已经正式启动了，正在初始化或者已经初始化完
    if _runtime_ready is not None:
        # 直接等待
        await _runtime_ready.wait()
        if _runtime_error is not None:
            raise _runtime_error
        if _agent is None:
            raise RuntimeError("MCP runtime finished without creating a graph")
        return _agent
    # 应用生命周期还没启动，拿到当前正在跑的事件循环
    # MCP Session 必须在创建它的事件循环中使用，因此需要记录并比较当前 loop。
    current_loop = asyncio.get_running_loop()

    task_failed = (
        _agent_init_task is not None    
        and _agent_init_task.done()
        and (
            _agent_init_task.cancelled()
            or _agent_init_task.exception() is not None
        )
    )
    # 如果满足下面任意一条，就重新派一个新建agent的任务
    if (
        _agent_init_task is None
        or task_failed
        or _agent_init_task.get_loop() is not current_loop
    ):
        _agent_init_task = current_loop.create_task(_initialize_runtime())

    _agent = await _agent_init_task
    return _agent


async def _run_mcp_lifecycle(shutdown_event, checkpointer):
    """Start and stop MCP resources in one long-lived asyncio task."""
    # MCP 生命周期管理
    global _agent, _runtime_error
    try:
        # 检查 Ollama
        log_event("Ollama", "checking http://127.0.0.1:11434")
        await asyncio.to_thread(ensure_ollama_ready)
        log_event("Ollama", f"ready ({OLLAMA_MODEL})")
        await _mcp_manager.start()
        _agent = await create_graph(
            _mcp_manager,
            memory_manager=_memory_manager,
            checkpointer=checkpointer,
        )
    # 如果上面任何一步出错了，就把错误记下来
    except BaseException as exc:
        _runtime_error = exc
    finally:
        _runtime_ready.set()
    # 停在这里，等“关门信号”。
    # Event.wait() 只等待信号，不会阻塞整个事件循环。
    await shutdown_event.wait()
    if _mcp_manager.started:
        await _mcp_manager.stop()


@asynccontextmanager
async def app_lifespan(_app):
    """Own MCP resources for exactly the lifetime of the Gradio ASGI app."""
    # @asynccontextmanager 把 yield 前后的代码分别变成异步进入和退出阶段。
    # MCP 资源由当前函数管理，其生命周期与 Gradio ASGI 应用的生命周期一致。
    # 管理 MCP 资源的生命周期
    global _agent, _agent_init_task, _checkpointer, _runtime_ready, _runtime_error
    # 日志记录
    log_event("MCP Lifecycle", "startup")
    # Python 异步编程中的一种任务同步机制，等待直到其他协程调用_runtime_ready.set()
    _runtime_ready = asyncio.Event()
    # 在本次应用启动时，将之前记录的运行时错误状态清空
    _runtime_error = None
    SESSION_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Checkpointer 与 MCP Session 一样由 ASGI lifespan 持有，避免连接跨事件循环使用。
    async with AsyncSqliteSaver.from_conn_string(str(SESSION_DB_PATH)) as saver:
        await saver.setup()
        _checkpointer = saver
        shutdown_event = asyncio.Event()
        lifecycle_task = asyncio.create_task(
            # 调用的是你的 MCP 生命周期协程，并传入关闭事件
            _run_mcp_lifecycle(shutdown_event, saver),
            name="mcp-lifecycle",
        )
        try:
            # 会将协程安排为一个可以与其他异步任务并发执行的 Task
            # yield 期间 Gradio 正常处理请求；应用关闭后才会进入 finally。
            yield
        finally:
            log_event("MCP Lifecycle", "shutdown")
            shutdown_event.set()
            await lifecycle_task
            _agent = None
            _agent_init_task = None
            _checkpointer = None
            _runtime_ready = None
            _runtime_error = None


def _answer_only(content):
    # 从 Agent 的回复中提取真正的回答内容，去掉附加的执行信息（例如耗时、调用工具、执行轨迹）。
    # 统一转换为字符串，兼容模型返回的非字符串 content。
    text = content if isinstance(content, str) else str(content)
    # 分割并提取回答
    return text.split(EXECUTION_INFO_MARKER, 1)[0]


def history_to_messages(history):
    # 把 Gradio 保存的聊天记录转换成 LangChain 能识别的消息对象，然后传给 LangGraph，让 Qwen3-8B 能够理解之前聊过什么。
    # 例如：history = [
    # {"role": "user", "content": "我正在学习LangGraph"},
    # {"role": "assistant", "content": "LangGraph是一个Agent工作流框架"}]
    # 创建一个空列表，用于保存转换后的 LangChain 消息对象。
    messages = []
    for item in history or []:
        # 判断当前消息是不是字典
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content", "")
        else:
            role = getattr(item, "role", None)
            content = getattr(item, "content", "")

        if role == "user":
            messages.append(HumanMessage(content=str(content)))
        elif role == "assistant":
            # Execution diagnostics are UI metadata, not conversation context.
            # 页面执行信息不属于对话语义，重新送给模型前要先剥离。
            messages.append(AIMessage(content=_answer_only(content)))
    return messages


def _normalise_thread_id(thread_id):
    """限制会话 ID 字符，避免空值或任意长文本进入 Checkpointer 配置。"""
    cleaned = "".join(
        character
        for character in str(thread_id or "")
        if character.isalnum() or character in "-_"
    )[:80]
    return cleaned or uuid4().hex


def _new_session():
    """创建空白短期会话，同时更新页面字段和浏览器持久状态。"""
    thread_id = uuid4().hex
    return thread_id, thread_id, [], f"已创建新会话：`{thread_id}`"


def _load_browser_session(stored_thread_id):
    """页面刷新后从 BrowserState 恢复上次使用的会话 ID。"""
    thread_id = _normalise_thread_id(stored_thread_id)
    return thread_id, thread_id


async def restore_session(thread_id):
    """从 SQLite Checkpointer 读取可展示的用户消息和最终回答。"""
    thread_id = _normalise_thread_id(thread_id)
    agent = await get_agent()
    snapshot = await agent.aget_state(
        {"configurable": {"thread_id": thread_id}}
    )
    messages = snapshot.values.get("messages", []) if snapshot.values else []
    history = []
    for message in messages:
        if isinstance(message, HumanMessage):
            history.append({"role": "user", "content": str(message.content)})
        elif (
            isinstance(message, AIMessage)
            and not (message.tool_calls or [])
            and str(message.content or "").strip()
        ):
            history.append(
                {"role": "assistant", "content": _answer_only(message.content)}
            )
    status = (
        f"已恢复会话 `{thread_id}`，共 {len(history)} 条可展示消息。"
        if history
        else f"会话 `{thread_id}` 尚无历史消息。"
    )
    return history, status, thread_id


def _clock():
    """返回执行轨迹使用的本地时分秒。"""
    return datetime.now().strftime("%H:%M:%S")


def _tool_label(tool_name):
    # 用于将 LangGraph 中实际调用的工具名称转换成用户容易理解的 MCP 服务名称
    if tool_name in {"search", "fetch_content"}:
        return "DuckDuckGo Search MCP"
    if tool_name.startswith("browser_"):
        return "Browser MCP"
    if tool_name.startswith("pdf_"):
        return "PDF MCP"
    if tool_name.startswith("office_"):
        return "Office MCP"
    if tool_name in {"create_entities", "add_observations", "search_nodes"}:
        return "Memory MCP"
    if tool_name == "Snapshot":
        return "Computer MCP"
    if tool_name in {
        "read_file", "read_text_file", "read_media_file", "read_multiple_files",
        "write_file", "edit_file", "create_directory", "list_directory",
        "list_directory_with_sizes", "directory_tree", "move_file",
        "search_files", "get_file_info", "list_allowed_directories",
    }:
        return "Filesystem MCP"
    return f"Unknown Tool ({tool_name})"


def _render_progress(
    events,
    answer=None,
    elapsed=None,
    tools=None,
    skills=None,
    memory_statuses=None,
):
    # Agent 执行状态渲染函数，主要负责将 Agent 的运行过程、最终回答、耗时和 MCP 工具调用记录整理成 Markdown 文本，显示在 Gradio 聊天界面中。
    # 同一个函数负责处理中状态和完成状态的页面渲染
    process = "\n\n".join(f"[{timestamp}]  {event}" for timestamp, event in events)
    if answer is None:
        return f"### Agent 正在执行\n\n{process}"

    tool_text = "、".join(dict.fromkeys(tools or [])) or "未调用工具"
    skill_text = "、".join(dict.fromkeys(skills or [])) or "未使用 Skill"
    memory_text = "、".join(dict.fromkeys(memory_statuses or [])) or "无相关记忆"
    return (
        f"{answer}{EXECUTION_INFO_MARKER}\n\n"
        f"- 模型：{MODEL_NAME}\n"
        f"- 总耗时：{elapsed:.1f}s\n"
        f"- Skill：{skill_text}\n"
        f"- Memory：{memory_text}\n"
        f"- 调用工具：{tool_text}\n\n"
        f"### 执行过程\n\n{process}"
    )


async def chat(message, history, thread_id):
    """处理一轮 Web 对话，并通过异步生成器持续返回节点级进度。"""
    # message 是用户本轮输入；history 仅用于无 Checkpointer 的测试回退路径。
    # time.perf_counter() 适合测量经过时间，不受系统时钟调整影响。
    start_time = time.perf_counter()
    # 分别保存执行事件、调用过的工具以及本轮激活的 Skill。
    events = []
    called_tools = []
    active_skills = []
    memory_statuses = []
    # 定义内部日志函数
    def record(event):
        events.append((_clock(), event))

    # 请求日志只记录长度和会话前缀，避免把密码等敏感记忆原文写入终端日志。
    log_event(
        "Agent Start",
        f"input_chars={len(str(message))} "
        f"thread={_normalise_thread_id(thread_id)[:12]}",
    )
    record("收到用户请求")
    # yield 会把当前 Markdown 交给 Gradio，同时保留函数现场以便继续执行。
    yield _render_progress(events)
    # 进入异常处理块，任一初始化、模型或工具异常都会转成页面可读信息。
    try:
        record("加载 Agent 与 MCP 工具")
        yield _render_progress(events)
        agent = await get_agent()

        thread_id = _normalise_thread_id(thread_id)
        # SQLite Checkpointer 已保存该 thread_id 的历史，因此每轮只追加当前用户消息。
        # 仅在没有 Checkpointer 的直接函数测试中才回退到 Gradio history。
        messages = [HumanMessage(content=message)]
        if _checkpointer is None:
            messages = history_to_messages(history) + messages
        record("Agent 分析任务")
        yield _render_progress(events)

        final_answer = ""
        # LangGraph 流式执行
        async for update in agent.astream(
            # 传给 LangGraph 的初始 State
            {"messages": messages},
            config={"configurable": {"thread_id": thread_id}},
            # updates 表示按节点输出状态更新
            # 目前实现的是节点级进度更新，不是模型逐 Token 输出
            stream_mode="updates",
        ):
            # 遍历本次节点更新
            # 每次 update 通常包含刚执行完成的节点名称及该节点返回的状态增量。
            for node_name, node_output in update.items():
                node_messages = node_output.get("messages", [])
                if not isinstance(node_messages, list):
                    node_messages = [node_messages]

                if node_name == "skill_detector":
                    # 这里只展示节点实际返回的匹配结果，不根据用户文本猜测。
                    active_skill = node_output.get("active_skill")
                    if active_skill:
                        active_skills.append(active_skill)
                        record(f"匹配 Skill：{active_skill}")
                        record(f"加载 Skill：{active_skill}")
                elif node_name == "memory":
                    memory_status = node_output.get("memory_status", "")
                    if memory_status.startswith("retrieved:"):
                        count = memory_status.partition(":")[2]
                        label = f"检索到 {count} 条相关长期记忆"
                    elif memory_status.startswith("command:"):
                        label = f"执行记忆操作：{memory_status.partition(':')[2]}"
                    else:
                        label = "长期记忆检索降级"
                    memory_statuses.append(label)
                    record(label)
                elif node_name == "memory_update":
                    memory_status = node_output.get("memory_status", "")
                    label = {
                        "automatic:created": "已自动保存稳定信息",
                        "automatic:updated": "已自动更新长期记忆",
                        "automatic:duplicate": "长期记忆已存在",
                        "automatic:failed": "自动记忆保存失败",
                    }.get(memory_status)
                    if label:
                        memory_statuses.append(label)
                        record(label)
                elif node_name == "agent":
                    response = node_messages[-1] if node_messages else None
                    tool_calls = getattr(response, "tool_calls", None) or []
                    # 如果模型确实生成了工具调用请求，遍历所有工具调用请求。
                    if tool_calls:
                        for call in tool_calls:
                            tool_name = call.get("name", "unknown")
                            tool_label = _tool_label(tool_name)
                            called_tools.append(tool_label)
                            record(f"调用 {tool_label}（{tool_name}）")
                    # 如果没有工具调用，就提取回答
                    elif response is not None:
                        final_answer = _answer_only(response.content)
                        record("生成最终回答")
                elif node_name == "memory_reply":
                    response = node_messages[-1] if node_messages else None
                    if response is not None:
                        final_answer = _answer_only(response.content)
                        record("生成记忆操作结果")
                elif node_name == "tools":
                    # ToolMessage 表示工具节点已经返回消息，但内容仍可能是失败信息。
                    for tool_message in node_messages:
                        if isinstance(tool_message, ToolMessage):
                            record(f"工具返回结果（{tool_message.name}）")
                # 继续更新 Gradio 页面
                yield _render_progress(events)

        if not final_answer:
            final_answer = "Agent 已完成执行，但没有生成文本回答。"
        elapsed = time.perf_counter() - start_time
        log_event("Total Time", f"{elapsed:.2f} s")
        log_event("FINAL RESPONSE", final_answer)
        yield _render_progress(
            events,
            final_answer,
            elapsed,
            called_tools,
            active_skills,
            memory_statuses,
        )
    # 异常处理部分
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        log_event("Agent Error", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        record(f"执行失败：{type(exc).__name__}")
        error_text = str(exc)
        if type(exc).__name__ == "ConnectError":
            error_text = (
                "无法连接本地 Ollama 服务。请确认 Ollama 正在运行，并检查 "
                "http://127.0.0.1:11434。"
            )
        yield _render_progress(
            events,
            f"Agent 运行失败：\n\n{error_text}",
            elapsed,
            called_tools,
            active_skills,
            memory_statuses,
        )


CSS = """
.gradio-container { max-width: 1050px !important; margin: 0 auto !important; }
#hero { text-align: center; margin: 12px 0 4px; }
#hero h1 { font-size: 2rem; margin-bottom: 0.35rem; }
#hero p { color: var(--body-text-color-subdued); }
"""


with gr.Blocks() as demo:
    # Blocks 是 Gradio 的页面容器；with 代码块中声明的组件会加入同一个页面。
    gr.Markdown(
        """
        # Local Personal Agent
        基于 **LangGraph + Ollama Qwen3-8B + MCP** 的个人助手
        """,
        elem_id="hero",
    )
    # BrowserState 把最近会话 ID 保存在本机浏览器；SQLite 保存真正的消息状态。
    browser_thread_id = gr.BrowserState(
        default_value="",
        storage_key="local-personal-agent-thread-id",
        secret="local-personal-agent-thread-state-v1",
    )
    with gr.Row():
        thread_id_box = gr.Textbox(
            label="会话 ID（复制此值可在重启后恢复）",
            placeholder="留空时自动创建",
            scale=7,
        )
        new_session_button = gr.Button("新建会话", scale=1)
        restore_session_button = gr.Button("恢复会话", scale=1)
    session_status = gr.Markdown("会话状态将在这里显示。")
    chatbot = gr.Chatbot(
        height=600,
        layout="bubble",
        placeholder="例如：查看当前目录有哪些文件",
    )
    gr.ChatInterface(
        # additional_inputs 把稳定 thread_id 一并交给 chat()。
        fn=chat,
        chatbot=chatbot,
        textbox=gr.Textbox(
            placeholder="输入问题，按 Enter 发送",
            container=False,
            scale=7,
        ),
        additional_inputs=[thread_id_box],
        examples=[
            ["你好"],
            ["请查看 D:\\46-Agent 目录下有哪些文件"],
            ["帮我搜索最近一年关于 AI Agent Security 的论文"],
        ],
        # 历史由 LangGraph SQLite Checkpointer 管理，避免与 Gradio 再重复保存一份。
        save_history=False,
        concurrency_limit=4,
    )
    demo.load(
        _load_browser_session,
        inputs=[browser_thread_id],
        outputs=[thread_id_box, browser_thread_id],
    )
    thread_id_box.change(
        lambda value: _normalise_thread_id(value),
        inputs=[thread_id_box],
        outputs=[browser_thread_id],
    )
    new_session_button.click(
        _new_session,
        outputs=[thread_id_box, browser_thread_id, chatbot, session_status],
    )
    restore_session_button.click(
        restore_session,
        inputs=[thread_id_box],
        outputs=[chatbot, session_status, browser_thread_id],
    )


if __name__ == "__main__":
    if not is_port_available("127.0.0.1", 7860):
        raise SystemExit(
            "Port 7860 is already in use. Close the existing Local Personal "
            "Agent process or run start_agent.bat to restart it safely."
        )
    demo.queue(default_concurrency_limit=4).launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=CSS,
        app_kwargs={"lifespan": app_lifespan},
    )
