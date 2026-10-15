from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages



class AgentState(TypedDict, total=False):

    messages: Annotated[
        list,
        add_messages
    ]
    file_route: str
