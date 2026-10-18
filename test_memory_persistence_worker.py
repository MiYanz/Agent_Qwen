import argparse
import asyncio
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from mcp_client.manager import MCPManager
from memory.manager import MemoryManager


async def memory_mode(args):
    async with MCPManager(args.config) as mcp_manager:
        memory = MemoryManager(mcp_manager)
        if args.action == "write":
            result = await memory.handle_command(
                "记住，我的持久化测试代号是 MEMORY-E2E-2026。"
            )
            assert result.changed
        elif args.action == "read":
            records = await memory.retrieve_memories("我的持久化测试代号是什么？")
            assert any("MEMORY-E2E-2026" in record.content for record in records)
        elif args.action == "delete":
            count = await memory.delete_memories("持久化测试代号")
            assert count == 1
        print(f"MEMORY_PROCESS_{args.action.upper()}=PASS")


async def checkpoint_mode(args):
    async def answer(state):
        return {"messages": [AIMessage(content=f"turn:{len(state['messages'])}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    config = {"configurable": {"thread_id": args.thread_id}}
    async with AsyncSqliteSaver.from_conn_string(args.database) as saver:
        await saver.setup()
        graph = builder.compile(checkpointer=saver)
        if args.action == "write":
            await graph.ainvoke(
                {"messages": [HumanMessage(content="第一轮上下文")]},
                config=config,
            )
        elif args.action == "read":
            snapshot = await graph.aget_state(config)
            contents = [str(message.content) for message in snapshot.values["messages"]]
            assert "第一轮上下文" in contents
        elif args.action == "isolate":
            snapshot = await graph.aget_state(config)
            assert not snapshot.values
        print(f"CHECKPOINT_PROCESS_{args.action.upper()}=PASS")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("memory", "checkpoint"))
    parser.add_argument("action", choices=("write", "read", "delete", "isolate"))
    parser.add_argument("--config")
    parser.add_argument("--database")
    parser.add_argument("--thread-id", default="persistence-test")
    args = parser.parse_args()
    if args.kind == "memory":
        await memory_mode(args)
    else:
        Path(args.database).parent.mkdir(parents=True, exist_ok=True)
        await checkpoint_mode(args)


if __name__ == "__main__":
    asyncio.run(main())
