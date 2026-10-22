# 主工作流装配层：把文件检测、记忆、Skill、模型与工具节点连接起来。
# 这里只定义节点和路由；实际执行由返回的已编译 LangGraph 在 ainvoke/astream 时驱动。
from pathlib import Path

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition

from agent.nodes import (
    create_chat_node,
    create_memory_node,
    create_memory_update_node,
    create_skill_node,
    detect_file_route,
    memory_reply,
)
from agent.state import AgentState
from agent.subagents import AgentRegistry, create_supervisor_tools_node
from agent import nodes
from agent.roles import select_main_tools
from mcp_client.tools import load_mcp_tools
from memory.manager import MemoryManager
from skills.loader import SkillLoader

# 确定项目中 skills 文件夹的绝对路径。
SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"


# 输入是记忆节点更新后的状态；返回下一节点名称。
# 显式记忆命令已有可信操作结果时走 memory_reply，普通任务才继续选择 Skill。
def _route_after_memory(state):
    """显式记忆管理已经得到真实结果时，不再让模型改写操作结论。"""
    return "memory_reply" if state.get("memory_direct_response") else "skill_detector"


# mcp_manager 提供现成会话，memory_manager 可注入测试替身，checkpointer 决定主会话是否持久化。
# 返回已编译图，不在此执行用户任务；加载工具和发现 Skill 是构图阶段的副作用。
async def create_graph(mcp_manager, memory_manager=None, checkpointer=None):
    """加载工具和 Skill，并编译出可异步调用的 LangGraph Agent。"""
    # 启动 MCP 管理器
    await mcp_manager.start()
    # 加载可供 Agent 使用的工具
    tools = await load_mcp_tools(mcp_manager)
    # 发现阶段读取 Skill 文件以建立目录；完整说明只给拥有该 Skill 的角色注入。
    skill_loader = SkillLoader(SKILLS_DIR)
    skill_loader.discover()
    registry = AgentRegistry(tools, nodes.llm, skill_loader=skill_loader,
                             server_health=getattr(mcp_manager, "health", None))
    # 不能再把全部 MCP 直接追加给 Main，否则模型或旧规则仍可跳过专业 Agent。
    tools = select_main_tools(tools + registry.as_tools())
    memory_manager = memory_manager or MemoryManager(mcp_manager)

    # StateGraph 使用 AgentState 描述各节点共同读写的状态结构。
    graph = StateGraph(AgentState)
    # add_node 只注册节点函数；真正执行顺序由下面的边决定。
    graph.add_node("file_detector", detect_file_route)
    graph.add_node("memory", create_memory_node(memory_manager))
    graph.add_node("memory_reply", memory_reply)
    graph.add_node("skill_detector", create_skill_node(skill_loader))
    # 给多 Agent 系统增加一个“委托次数限制器”，防止 Main Agent 不断调用 Research Agent、File Agent，导致任务反复执行、陷入循环。
    graph.add_node("agent", create_chat_node(tools, delegation_limits=registry.limits))
    # create_supervisor_tools_node这个新节点的目的，是在执行工具调用时结合 registry 检查 Agent 委托预算。
    graph.add_node("tools", create_supervisor_tools_node(tools, registry))
    graph.add_node("memory_update", create_memory_update_node(memory_manager))

    # 每次正常启动这张图时，从 file_detector 开始执行。
    graph.set_entry_point("file_detector")

    # 普通边可以理解为一条固定的流程规则
    graph.add_edge("file_detector", "memory")   # 文件检测完成后进入记忆节点。
    graph.add_conditional_edges(    
        "memory",
        _route_after_memory,
        {"memory_reply": "memory_reply", "skill_detector": "skill_detector"},
    )
    graph.add_edge("memory_reply", END)
    graph.add_edge("skill_detector", "agent")   # Skill 选择完成后进入主 Agent。
    
    # 条件边
    # tools_condition 检查最新 AIMessage 是否包含 tool_calls：
    # 有工具请求时进入 ToolNode，否则结束本轮图执行。
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: "memory_update"},
    )
    
    # 工具结果以 ToolMessage 写回 State 后必须回到 Agent，让模型生成最终回答。
    graph.add_edge("tools", "agent")
    graph.add_edge("memory_update", END)

    # compile() 把节点和边构造成可 ainvoke/astream 的可执行图。
    return graph.compile(checkpointer=checkpointer)
