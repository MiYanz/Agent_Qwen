# memory 包的公开导入入口，允许调用方直接导入 MemoryManager。
# __all__ 声明星号导入时公开的名称，不会在这里创建数据库或启动 MCP。
"""本地 Agent 的长期记忆管理与短期会话持久化支持。"""

from memory.manager import MemoryManager

__all__ = ["MemoryManager"]
