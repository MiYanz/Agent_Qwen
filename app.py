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
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

sys.dont_write_bytecode = True

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import create_graph
from mcp_client.manager import MCPManager
from mcp_client.tools import CONFIG_PATH


MODEL_NAME = "Qwen3-8B"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"
EXECUTION_INFO_MARKER = "\n\n---\n### 执行信息"

_agent = None
_agent_init_task = None
_mcp_manager = MCPManager(CONFIG_PATH)
_runtime_ready = None
_runtime_error = None


def log_event(label, detail=None):
    print(f"\n[{label}]", flush=True)
    if detail:
        print(detail, flush=True)


def is_port_available(host, port):
    """Fail before ASGI startup so MCP processes are not started needlessly."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def _ollama_models():
    """Return installed Ollama models without using the system proxy."""
    opener = build_opener(ProxyHandler({}))
    with opener.open(OLLAMA_TAGS_URL, timeout=2) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {model.get("name") for model in payload.get("models", [])}


def ensure_ollama_ready():
    """Start Ollama when needed and wait until its local API is ready."""
    try:
        models = _ollama_models()
    except (OSError, TimeoutError, URLError, json.JSONDecodeError):
        executable = shutil.which("ollama")
        if not executable:
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

        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            [executable, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
        deadline = time.monotonic() + 30
        last_error = None
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

    if OLLAMA_MODEL not in models:
        raise RuntimeError(
            f"Ollama 中未安装 {OLLAMA_MODEL}，请运行 `ollama pull {OLLAMA_MODEL}`。"
        )


async def _initialize_runtime():
    global _agent, _agent_init_task
    await asyncio.to_thread(ensure_ollama_ready)
    await _mcp_manager.start()
    if _agent is None:
        _agent = await create_graph(_mcp_manager)
    return _agent


async def get_agent():
    """Return the graph initialized by the app lifespan or initialize lazily."""
    global _agent, _agent_init_task
    if _agent is not None:
        return _agent

    if _runtime_ready is not None:
        await _runtime_ready.wait()
        if _runtime_error is not None:
            raise _runtime_error
        if _agent is None:
            raise RuntimeError("MCP runtime finished without creating a graph")
        return _agent

    current_loop = asyncio.get_running_loop()
    task_failed = (
        _agent_init_task is not None
        and _agent_init_task.done()
        and (
            _agent_init_task.cancelled()
            or _agent_init_task.exception() is not None
        )
    )
    if (
        _agent_init_task is None
        or task_failed
        or _agent_init_task.get_loop() is not current_loop
    ):
        _agent_init_task = current_loop.create_task(_initialize_runtime())

    _agent = await _agent_init_task
    return _agent


async def _run_mcp_lifecycle(shutdown_event):
    """Start and stop MCP resources in one long-lived asyncio task."""
    global _agent, _runtime_error
    try:
        log_event("Ollama", "checking http://127.0.0.1:11434")
        await asyncio.to_thread(ensure_ollama_ready)
        log_event("Ollama", f"ready ({OLLAMA_MODEL})")
        await _mcp_manager.start()
        _agent = await create_graph(_mcp_manager)
    except BaseException as exc:
        _runtime_error = exc
    finally:
        _runtime_ready.set()

    await shutdown_event.wait()
    if _mcp_manager.started:
        await _mcp_manager.stop()


@asynccontextmanager
async def app_lifespan(_app):
    """Own MCP resources for exactly the lifetime of the Gradio ASGI app."""
    global _agent, _agent_init_task, _runtime_ready, _runtime_error
    log_event("MCP Lifecycle", "startup")
    _runtime_ready = asyncio.Event()
    _runtime_error = None
    shutdown_event = asyncio.Event()
    lifecycle_task = asyncio.create_task(
        _run_mcp_lifecycle(shutdown_event),
        name="mcp-lifecycle",
    )
    try:
        yield
    finally:
        log_event("MCP Lifecycle", "shutdown")
        shutdown_event.set()
        await lifecycle_task
        _agent = None
        _agent_init_task = None
        _runtime_ready = None
        _runtime_error = None


def _answer_only(content):
    text = content if isinstance(content, str) else str(content)
    return text.split(EXECUTION_INFO_MARKER, 1)[0]


def history_to_messages(history):
    messages = []
    for item in history or []:
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
            messages.append(AIMessage(content=_answer_only(content)))
    return messages


def _clock():
    return datetime.now().strftime("%H:%M:%S")


def _tool_label(tool_name):
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
    return "Filesystem MCP"


def _render_progress(events, answer=None, elapsed=None, tools=None):
    process = "\n\n".join(f"[{timestamp}]  {event}" for timestamp, event in events)
    if answer is None:
        return f"### Agent 正在执行\n\n{process}"

    tool_text = "、".join(dict.fromkeys(tools or [])) or "未调用工具"
    return (
        f"{answer}{EXECUTION_INFO_MARKER}\n\n"
        f"- 模型：{MODEL_NAME}\n"
        f"- 总耗时：{elapsed:.1f}s\n"
        f"- 调用工具：{tool_text}\n\n"
        f"### 执行过程\n\n{process}"
    )


async def chat(message, history):
    start_time = time.perf_counter()
    events = []
    called_tools = []

    def record(event):
        events.append((_clock(), event))

    log_event("Agent Start", f"input={message}")
    record("收到用户请求")
    yield _render_progress(events)

    try:
        record("加载 Agent 与 MCP 工具")
        yield _render_progress(events)
        agent = await get_agent()

        messages = history_to_messages(history)
        messages.append(HumanMessage(content=message))
        record("Agent 分析任务")
        yield _render_progress(events)

        final_answer = ""
        async for update in agent.astream(
            {"messages": messages},
            stream_mode="updates",
        ):
            for node_name, node_output in update.items():
                node_messages = node_output.get("messages", [])
                if not isinstance(node_messages, list):
                    node_messages = [node_messages]

                if node_name == "agent":
                    response = node_messages[-1] if node_messages else None
                    tool_calls = getattr(response, "tool_calls", None) or []
                    if tool_calls:
                        for call in tool_calls:
                            tool_name = call.get("name", "unknown")
                            tool_label = _tool_label(tool_name)
                            called_tools.append(tool_label)
                            record(f"调用 {tool_label}（{tool_name}）")
                    elif response is not None:
                        final_answer = _answer_only(response.content)
                        record("生成最终回答")
                elif node_name == "tools":
                    for tool_message in node_messages:
                        if isinstance(tool_message, ToolMessage):
                            record(f"工具返回结果（{tool_message.name}）")

                yield _render_progress(events)

        if not final_answer:
            final_answer = "Agent 已完成执行，但没有生成文本回答。"
        elapsed = time.perf_counter() - start_time
        log_event("Total Time", f"{elapsed:.2f} s")
        log_event("FINAL RESPONSE", final_answer)
        yield _render_progress(events, final_answer, elapsed, called_tools)
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
        )


CSS = """
.gradio-container { max-width: 1050px !important; margin: 0 auto !important; }
#hero { text-align: center; margin: 12px 0 4px; }
#hero h1 { font-size: 2rem; margin-bottom: 0.35rem; }
#hero p { color: var(--body-text-color-subdued); }
"""


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Local Personal Agent
        基于 **LangGraph + Ollama Qwen3-8B + MCP** 的个人助手
        """,
        elem_id="hero",
    )
    gr.ChatInterface(
        fn=chat,
        chatbot=gr.Chatbot(
            height=600,
            layout="bubble",
            placeholder="例如：查看当前目录有哪些文件",
        ),
        textbox=gr.Textbox(
            placeholder="输入问题，按 Enter 发送",
            container=False,
            scale=7,
        ),
        examples=[
            "你好",
            "请查看 D:\\46-Agent 目录下有哪些文件",
            "帮我搜索最近一年关于 AI Agent Security 的论文",
        ],
        save_history=True,
        concurrency_limit=4,
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
