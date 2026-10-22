# 长期记忆的数据协议：把业务记录编码到 Memory MCP 实体的 observation 字符串中。
# 这些 dataclass 只表达数据；实际读写由 memory/manager.py 通过 MCP 完成。
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone


# 固定前缀负责标记本项目实体与 observation 协议，读取时用它排除其他应用的数据。
MEMORY_ENTITY_PREFIX = "agent-memory::"
MEMORY_ENTITY_TYPE = "local_personal_agent_memory"
MEMORY_OBSERVATION_PREFIX = "AGENT_MEMORY_V1:"


# 返回带 UTC 时区的 ISO 时间文本，供版本比较和记录序列化使用，不访问存储。
def utc_now():
    """生成带时区的 UTC 时间，便于可靠比较记忆的新旧版本。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# 已经保存的业务记忆；memory_id 区分实体，key 表示可被更新的同一主题。
# frozen=True 使字段不可直接重新赋值，更新会创建新记录而不是原地篡改旧版本。
@dataclass(frozen=True)
class MemoryRecord:
    """Memory MCP 中一条由本项目管理的结构化长期记忆。"""

    memory_id: str
    key: str
    kind: str
    content: str
    source: str
    created_at: str
    updated_at: str
    status: str = "active"

    # 把内部 ID 转为带项目命名空间的 MCP 实体名，避免操作其他应用保存的实体。
    @property
    def entity_name(self):
        """每条记忆使用独立实体，删除时不会影响其他记忆。"""
        return f"{MEMORY_ENTITY_PREFIX}{self.memory_id}"

    # 列出协议字段并返回普通字典；不包含 entity_name 这个由 ID 派生的属性。
    def to_payload(self):
        return {
            "memory_id": self.memory_id,
            "key": self.key,
            "kind": self.kind,
            "content": self.content,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
        }

    # 给 JSON 数据加上固定前缀，作为一个 MCP observation 字符串保存。
    # 读取时可以据此区分本项目结构化记录与其他自由文本观察。
    def to_observation(self):
        """把结构化字段编码进观察文本，同时保持 MCP 原生 JSONL 格式。"""
        return MEMORY_OBSERVATION_PREFIX + json.dumps(
            self.to_payload(), ensure_ascii=False, separators=(",", ":")
        )

    # 返回保留原 ID、主题键和创建时间的新实例，仅更新内容、时间及选定字段。
    def updated(self, *, content, kind=None, source=None):
        """保留创建时间和 ID，生成同一事实的新版本。"""
        return replace(
            self,
            content=content,
            kind=kind or self.kind,
            source=source or self.source,
            updated_at=utc_now(),
            status="active",
        )

    # 输入 MCP 实体字典，过滤非本项目实体，逐条解析带前缀的 observation。
    # 损坏或不符合模型字段的记录跳过；若同一实体保留多个版本，选更新时间最大的版本。
    @classmethod
    def from_entity(cls, entity):
        """解析本项目实体；未知或损坏的旧数据会被安全忽略。"""
        name = str(entity.get("name", ""))
        if not name.startswith(MEMORY_ENTITY_PREFIX):
            return None

        # 同一实体可能在更新中暂时保留多个 observation；全部解析后再按更新时间选最新版本。
        records = []
        for observation in entity.get("observations", []):
            if not str(observation).startswith(MEMORY_OBSERVATION_PREFIX):
                continue
            try:
                payload = json.loads(
                    str(observation)[len(MEMORY_OBSERVATION_PREFIX):]
                )
                records.append(cls(**payload))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        if not records:
            return None
        # 更新采用“先追加新版本，再删除旧版本”，即使第二步异常也能选中最新记录。
        return max(records, key=lambda record: record.updated_at)


# 策略层产出的待保存内容，没有持久化 ID；save_memory 根据 key 决定创建、更新或去重。
@dataclass(frozen=True)
class MemoryCandidate:
    """尚未持久化、已经通过策略筛选的记忆候选。"""

    key: str
    kind: str
    content: str
    source: str


# 显式用户命令的解析结果；query 用于查找主题，content 用于保存，confirmed 标记明确确认。
@dataclass(frozen=True)
class MemoryCommand:
    """从自然语言识别出的显式记忆管理命令。"""

    action: str
    query: str = ""
    content: str = ""
    confirmed: bool = False


# 返回给图节点的可信操作结论；handled 说明是否拦截本轮，changed 说明存储是否改变。
# response 可直接展示给用户，action 供路由和日志分类，避免让模型重新猜测操作是否完成。
@dataclass(frozen=True)
class MemoryOperationResult:
    """Memory 节点向 LangGraph 返回的可验证操作结果。"""

    handled: bool
    response: str = ""
    action: str = ""
    changed: bool = False
