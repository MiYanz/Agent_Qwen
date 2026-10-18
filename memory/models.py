import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone


MEMORY_ENTITY_PREFIX = "agent-memory::"
MEMORY_ENTITY_TYPE = "local_personal_agent_memory"
MEMORY_OBSERVATION_PREFIX = "AGENT_MEMORY_V1:"


def utc_now():
    """生成带时区的 UTC 时间，便于可靠比较记忆的新旧版本。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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

    @property
    def entity_name(self):
        """每条记忆使用独立实体，删除时不会影响其他记忆。"""
        return f"{MEMORY_ENTITY_PREFIX}{self.memory_id}"

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

    def to_observation(self):
        """把结构化字段编码进观察文本，同时保持 MCP 原生 JSONL 格式。"""
        return MEMORY_OBSERVATION_PREFIX + json.dumps(
            self.to_payload(), ensure_ascii=False, separators=(",", ":")
        )

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

    @classmethod
    def from_entity(cls, entity):
        """解析本项目实体；未知或损坏的旧数据会被安全忽略。"""
        name = str(entity.get("name", ""))
        if not name.startswith(MEMORY_ENTITY_PREFIX):
            return None

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


@dataclass(frozen=True)
class MemoryCandidate:
    """尚未持久化、已经通过策略筛选的记忆候选。"""

    key: str
    kind: str
    content: str
    source: str


@dataclass(frozen=True)
class MemoryCommand:
    """从自然语言识别出的显式记忆管理命令。"""

    action: str
    query: str = ""
    content: str = ""
    confirmed: bool = False


@dataclass(frozen=True)
class MemoryOperationResult:
    """Memory 节点向 LangGraph 返回的可验证操作结果。"""

    handled: bool
    response: str = ""
    action: str = ""
    changed: bool = False
