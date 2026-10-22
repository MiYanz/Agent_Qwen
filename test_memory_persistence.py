# 跨进程持久化的父进程测试：在临时目录创建专用配置、记忆文件和 SQLite。
# 每次操作启动全新的 worker，验证数据确实来自持久存储，而不是同一进程的缓存。
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MEMORY_SERVER = (
    ROOT
    / "mcp_servers"
    / "node_modules"
    / "@modelcontextprotocol"
    / "server-memory"
    / "dist"
    / "index.js"
)
NODE = ROOT / "mcp_servers" / "node_modules" / "node" / "bin" / "node.exe"
WORKER = ROOT / "test_memory_persistence_worker.py"


# 把操作和临时路径作为参数传给全新 Python 进程，check=True 使子进程失败立即暴露。
# 捕获并展示标准输出，避免误把进程启动成功当成持久化检查通过。
def run_worker(*arguments):
    result = subprocess.run(
        [sys.executable, str(WORKER), *map(str, arguments)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    print(result.stdout.strip())


# 临时创建独立 Memory 配置和数据库，让多个 worker 分别写、读、删及验证会话隔离。
# 临时文件退出后自动清理；正常项目记忆文件路径不传给这些 worker。
def main():
    """使用全新进程验证 Memory MCP 和 Checkpointer 在重启后仍能读取。"""
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        memory_file = temp / "memory.jsonl"
        config_file = temp / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "memory": {
                            "command": str(NODE),
                            "args": [str(MEMORY_SERVER)],
                            "env": {"MEMORY_FILE_PATH": str(memory_file)},
                        }
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        # 三个独立进程依次验证写入、重启后读取和删除，防止测试只命中同一对象缓存。
        run_worker("memory", "write", "--config", config_file)
        run_worker("memory", "read", "--config", config_file)
        run_worker("memory", "delete", "--config", config_file)

        database = temp / "conversations.sqlite3"
        run_worker("checkpoint", "write", "--database", database)
        run_worker("checkpoint", "read", "--database", database)
        run_worker(
            "checkpoint",
            "isolate",
            "--database",
            database,
            "--thread-id",
            "independent-session",
        )
    print("CROSS_PROCESS_PERSISTENCE=PASS")


if __name__ == "__main__":
    main()
