# 主图与子图使用的状态字段声明。
# TypedDict 描述字典结构；字段如何合并由 LangGraph 读取的 reducer 决定，并非普通类型检查自动完成。
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

# TypedDict 只约束字典中允许出现哪些字段，运行时仍然是普通 dict。
# Annotated 可以在类型之外附加元数据；LangGraph 会读取这里的 add_messages，
# 在节点返回新消息时把它们追加到历史，而不是直接覆盖 messages。
# messages：保存 Agent 的对话消息，包括用户输入、模型回复和工具返回。
# file_route：保存文件路由信息，具体含义需要结合 nodes.py 和 graph.py 中对它的赋值与读取来确定。
# active_skill：保存本轮匹配到的 Skill 名称；没有匹配时为空字符串。
# skill_instruction：仅保存已匹配 Skill 的完整说明，不会一次加载全部 Skill 正文。
# memory_context：保存本轮检索出的少量相关长期记忆，只作为模型的辅助数据。
# memory_status：记录本轮检索或自动更新状态，供日志展示，不包含敏感内容。
# memory_direct_response：显式查看、保存、修改和删除记忆时的可信操作结果。

# 各节点只需要返回自己改变的字段，不必重复构造全部状态。
# total=False 表示输入允许缺省字段；消息用追加合并，普通标量字段按最新更新覆盖。
class AgentState(TypedDict, total=False):
    # 用于描述一个字典应该具备怎样的结构
    # total=False允许你不提供某些键，适合节点逐步补充状态。
    messages: Annotated[
        list,
        # 当节点更新 messages 时，应该使用 add_messages 合并，而不是简单覆盖。
        add_messages
    ]
    # file_route 是为了保存文件类型或文件处理工具的路由信息
    file_route: str
    # active_skill 和 skill_instruction 由 skill_detector 写入，供 Agent 节点读取。
    active_skill: str
    skill_instruction: str
    memory_context: str
    memory_status: str
    memory_direct_response: str
    # 本轮委托预算随主会话持久化，在请求入口重置，不使用跨会话全局计数。
    # 计数是本轮请求的状态，不是整个服务的总使用量；file_detector 每轮会重置它。
    delegation_count: int
    # 保存任务内容指纹及显式任务 ID 的使用情况；键是摘要，避免把任务原文当索引保存。
    delegation_seen: dict[str, int]
    # 每次用户请求生成新主运行 ID，专业任务通过 parent_run_id 与这一轮建立关联。
    main_run_id: str
    # 当前节点产生的结构化诊断事件；不使用 add_messages，不进入对话正文。
    agent_events: list[dict]
