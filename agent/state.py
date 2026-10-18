from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

# messages：保存 Agent 的对话消息，包括用户输入、模型回复和工具返回。
# file_route：保存文件路由信息，具体含义需要结合 nodes.py 和 graph.py 中对它的赋值与读取来确定。

class AgentState(TypedDict, total=False):
    # 用于描述一个字典应该具备怎样的结构
    # total=False允许你不提供某些键。
    messages: Annotated[
        list,
        # 当节点更新 messages 时，应该使用 add_messages 合并，而不是简单覆盖。
        add_messages
    ]
    # file_route 是为了保存文件类型或文件处理工具的路由信息
    file_route: str
