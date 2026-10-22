"""角色边界的集中配置：工具权限与 Skill 归属不由模型自行决定。"""

import json
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.messages import HumanMessage, ToolMessage

from mcp_client.tools import _safe_print


# Main 只直接保留现有的操作型能力。网络浏览、搜索以及文件读取/解析全部留给专业 Agent。
# 这里没有新增任何 MCP 权限；工具仍需先通过原配置的 includeTools 才会进入候选列表。
MAIN_MCP_ALLOWLIST = frozenset({
    ("computer", "Snapshot"),
    ("office-writer", "create_docx"),
    ("filesystem", "write_file"),
    ("filesystem", "edit_file"),
    ("filesystem", "create_directory"),
    ("filesystem", "move_file"),
})

# Skill 是给角色的操作说明，不是另一个 Agent，也不能授予工具权限。
# 没有声明归属的用户自定义 Skill 仍由 Main 发现和选择，保留现有扩展方式。
SKILL_OWNERS = {"news-search": "research", "ppt-summary": "file"}


def select_main_tools(tools):
    """模型绑定和执行端共用同一白名单，防止只隐藏工具而未限制真实执行。"""
    selected = []
    for tool in tools:
        metadata = tool.metadata or {}
        agent_name = metadata.get("agent_name")
        is_agent_tool = agent_name and tool.name == f"call_{agent_name}_agent"
        origin = (metadata.get("mcp_server"), metadata.get("mcp_tool"))
        if is_agent_tool or origin in MAIN_MCP_ALLOWLIST:
            selected.append(tool)
    return selected


def agent_event(agent_name, event_type, *, task_id, parent_run_id, target,
                status, elapsed_seconds=0, **details):
    """生成可持久化、可展示的角色事件；终端只输出身份和状态，不打印任务/正文。"""
    event = {
        "event_id": uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_name": agent_name,
        "task_id": task_id,
        "parent_run_id": parent_run_id,
        "event_type": event_type,
        "target": target,
        "status": status,
        "elapsed_seconds": round(elapsed_seconds, 3),
        **details,
    }
    # 任务摘要和错误详情可在本地页面查看；终端不无条件输出它们，避免泄漏敏感文件信息。
    public = {key: value for key, value in event.items()
              if key not in {"task", "summary", "error"}}
    _safe_print("[Agent Event] " + json.dumps(public, ensure_ascii=False))
    return event


def exhausted_agents(messages, limit=2):
    """只统计本轮未完成的委托，改写任务文字或换 task_id 也不能无限重试。"""
    start = next((i for i in range(len(messages) - 1, -1, -1)
                  if isinstance(messages[i], HumanMessage)), 0)
    counts = {}
    for message in messages[start:]:
        if not isinstance(message, ToolMessage) or message.name not in {
                "call_research_agent", "call_file_agent"}:
            continue
        try:
            result = json.loads(str(message.content))
            if result.get("status") in {"partial", "failed"}:
                name = message.name.removeprefix("call_").removesuffix("_agent")
                counts[name] = counts.get(name, 0) + 1
        except (ValueError, AttributeError):
            continue
    return {name for name, count in counts.items() if count >= limit}

