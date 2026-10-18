# Memory 目录

本目录包含两类职责不同的本地记忆：

- `memory.jsonl`：Memory MCP 的长期记忆知识图谱，是长期记忆的唯一权威来源。
- `conversations.sqlite3`：LangGraph SQLite Checkpointer 保存的短期会话状态。
- `manager.py`：只通过 MCP 接口检索、保存、更新和删除长期记忆。
- `policy.py`：显式记忆命令、保守自动提取、敏感信息和一次性任务过滤策略。
- `models.py`：长期记忆的结构化字段及 MCP observation 编解码。

`memory.jsonl` 和 `conversations.sqlite3*` 都属于用户私有运行数据，已经由
`.gitignore` 排除。代码不得绕过正在运行的 Memory MCP 直接写入 JSONL。

长期记忆只保存明确且稳定的信息；密码、令牌、实时搜索结果和临时文件正文
默认不会保存。SQLite 会话和长期记忆相互独立，清空长期记忆不会删除会话历史。
