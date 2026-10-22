# 离线长期记忆回归：FakeMCPManager 用内存列表模拟 MCP 的实体和观察记录。
# 所有新增、更新和清空操作只改变测试对象，不读写正式 memory.jsonl。
import asyncio
from types import SimpleNamespace

from memory.manager import MemoryManager
from memory.policy import candidate_from_content


# 模拟 Memory MCP 的实体及 observation 增删接口，让真实 MemoryManager 在内存后端上运行。
class FakeMCPManager:
    """只在单元测试中模拟 Memory MCP 的知识图谱行为。"""

    # entities 是测试存储，fail 用来在任意请求时制造后端不可用。
    def __init__(self, fail=False):
        self.entities = []
        self.fail = fail

    # 按原始 MCP 工具名修改内存图谱，返回可供真实解析器读取的 structuredContent。
    # 未知服务或工具直接报错，以便发现测试未覆盖的意外调用。
    async def call_tool(self, tool_name, arguments, server_name="memory"):
        if self.fail:
            raise RuntimeError("simulated unavailable storage")
        if server_name != "memory":
            raise AssertionError(server_name)
        # 返回浅拷贝列表模拟读取；下面的写分支只操作这个 Fake 的 entities，不会连到真实 Memory MCP。
        if tool_name == "read_graph":
            return SimpleNamespace(
                structuredContent={"entities": list(self.entities), "relations": []}
            )
        if tool_name == "create_entities":
            self.entities.extend(arguments["entities"])
        elif tool_name == "add_observations":
            for update in arguments["observations"]:
                entity = next(
                    item for item in self.entities
                    if item["name"] == update["entityName"]
                )
                entity["observations"].extend(update["contents"])
        elif tool_name == "delete_observations":
            for deletion in arguments["deletions"]:
                entity = next(
                    item for item in self.entities
                    if item["name"] == deletion["entityName"]
                )
                entity["observations"] = [
                    item for item in entity["observations"]
                    if item not in deletion["observations"]
                ]
        elif tool_name == "delete_entities":
            names = set(arguments["entityNames"])
            self.entities = [
                item for item in self.entities if item["name"] not in names
            ]
        else:
            raise AssertionError(tool_name)
        return SimpleNamespace(structuredContent={"success": True})


# 在相同 Fake 后端前增加异步等待，用于验证 MemoryManager 的等待上限。
class SlowMCPManager(FakeMCPManager):
    # 先让出事件循环再调用父类；测试设置的更短 timeout 应在等待中取消请求。
    async def call_tool(self, tool_name, arguments, server_name="memory"):
        await asyncio.sleep(0.05)
        return await super().call_tool(tool_name, arguments, server_name)


# 按保存、自动策略、更新去重、相关性、删除、错误和确认清空顺序执行断言。
# 这些操作只修改 backend.entities，不使用用户正式记忆文件。
async def main():
    backend = FakeMCPManager()
    manager = MemoryManager(backend)

    # 显式保存及跨 Manager 对象读取；两个对象共享的 backend 相当于持久存储。
    operation = await manager.handle_command(
        r"记住，我的 Agent 项目路径是 D:\46-Agent。"
    )
    assert operation.handled and operation.changed
    restarted_manager = MemoryManager(backend)
    found = await restarted_manager.retrieve_memories("我的 Agent 项目路径是什么？")
    assert found and r"D:\46-Agent" in found[0].content
    print("EXPLICIT_SAVE_AND_RETRIEVE=PASS")

    # 稳定偏好自动保存，普通一次性查询不保存。
    status = await manager.process_automatic_memory(
        "以后帮我解释 Python 代码时，尽量逐行解释并使用中文。"
    )
    assert status == "created"
    assert await manager.process_automatic_memory("今天星期几？") == "skipped"
    assert await manager.process_automatic_memory("搜索今天的微博热搜") == "skipped"
    print("CONSERVATIVE_AUTO_MEMORY=PASS")

    # 相同 key 的新模型配置更新旧值，相同内容再次保存只会去重。
    await manager.process_automatic_memory("我使用 Qwen3-8B 作为本地模型。")
    assert await manager.process_automatic_memory("我现在使用 Qwen3-14B。") == "updated"
    assert await manager.process_automatic_memory("我现在使用 Qwen3-14B。") == "duplicate"
    records = await manager.list_memories()
    model_records = [item for item in records if item.key == "project.local_model"]
    assert len(model_records) == 1
    assert "14B" in model_records[0].content and "8B" not in model_records[0].content
    print("UPDATE_AND_DEDUPLICATION=PASS")

    # 无关记忆不会被检索结果全量注入。
    unrelated = await manager.retrieve_memories("明天的天气怎么样？")
    assert not unrelated
    assert not await manager.retrieve_memories("刚才的临时代号叫什么？")
    print("RELEVANCE_FILTER=PASS")

    # 指定删除必须真实删除底层实体，之后无法再次检索。
    deleted = await manager.handle_command("忘记我之前关于 Agent 项目路径的记忆")
    assert deleted.changed
    assert not await manager.retrieve_memories("Agent 项目路径")
    print("DELETE_MEMORY=PASS")

    # 敏感内容默认不保存；Memory MCP 故障时显式操作返回真实失败。
    sensitive = await manager.handle_command("记住，我的密码是 123456")
    assert sensitive.handled and not sensitive.changed
    assert "默认没有保存" in sensitive.response
    failed = await MemoryManager(FakeMCPManager(fail=True)).handle_command(
        "查看长期记忆"
    )
    assert failed.handled and "失败" in failed.response
    print("SENSITIVE_AND_FAILURE_POLICY=PASS")

    # 损坏的旧 observation 被隔离；超时也只让记忆功能失败，不抛到聊天主流程。
    backend.entities.append(
        {
            "name": "agent-memory::broken",
            "entityType": "local_personal_agent_memory",
            "observations": ["AGENT_MEMORY_V1:{broken-json"],
        }
    )
    await manager.list_memories(force_refresh=True)
    timed_out = await MemoryManager(
        SlowMCPManager(), timeout=0.001
    ).handle_command("查看长期记忆")
    assert timed_out.handled and "失败" in timed_out.response
    print("MALFORMED_DATA_AND_TIMEOUT=PASS")

    await manager.handle_command("记住，我喜欢简洁回答。")
    warning = await manager.handle_command("清空所有长期记忆")
    assert warning.handled and not warning.changed
    assert await manager.list_memories()
    cleared = await manager.handle_command("确认清空所有长期记忆")
    assert cleared.changed and not await manager.list_memories()
    print("CLEAR_CONFIRMATION=PASS")


if __name__ == "__main__":
    asyncio.run(main())
