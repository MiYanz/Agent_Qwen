from pathlib import Path

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.nodes import create_chat_node, create_skill_node, detect_file_route
from agent.state import AgentState
from mcp_client.tools import load_mcp_tools
from skills.loader import SkillLoader


SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"
# 根据 graph.py 的绝对位置计算项目 skills 目录，避免依赖启动命令的当前目录。


async def create_graph(mcp_manager):
    """加载工具和 Skill，并编译出可异步调用的 LangGraph Agent。"""
    # await 会暂停当前协程，直到 Manager 完成启动；等待期间事件循环仍可处理其他任务。
    await mcp_manager.start()
    # 这里拿到的是已经绑定到长生命周期 MCPManager 的 LangChain 工具。
    tools = await load_mcp_tools(mcp_manager)
    # 启动时只发现 Skill 元数据；完整 SKILL.md 由 skill_detector 按需读取。
    skill_loader = SkillLoader(SKILLS_DIR)
    skill_loader.discover()

    # StateGraph 使用 AgentState 描述各节点共同读写的状态结构。
    graph = StateGraph(AgentState)
    # add_node 只注册节点函数；真正执行顺序由下面的边决定。
    graph.add_node("file_detector", detect_file_route)
    graph.add_node("skill_detector", create_skill_node(skill_loader))
    graph.add_node("agent", create_chat_node(tools))
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("file_detector")
    graph.add_edge("file_detector", "skill_detector")
    graph.add_edge("skill_detector", "agent")
    # tools_condition 检查最新 AIMessage 是否包含 tool_calls：
    # 有工具请求时进入 ToolNode，否则结束本轮图执行。
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    # 工具结果以 ToolMessage 写回 State 后必须回到 Agent，让模型生成最终回答。
    graph.add_edge("tools", "agent")

    # compile() 把节点和边构造成可 ainvoke/astream 的可执行图。
    return graph.compile()
