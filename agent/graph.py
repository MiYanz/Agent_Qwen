from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.nodes import create_chat_node, detect_file_route
from agent.state import AgentState
from mcp_client.tools import load_mcp_tools


async def create_graph(mcp_manager):
    await mcp_manager.start()
    tools = await load_mcp_tools(mcp_manager)

    graph = StateGraph(AgentState)
    graph.add_node("file_detector", detect_file_route)
    graph.add_node("agent", create_chat_node(tools))
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("file_detector")
    graph.add_edge("file_detector", "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile()
