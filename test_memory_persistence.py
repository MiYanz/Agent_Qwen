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


def run_worker(*arguments):
    result = subprocess.run(
        [sys.executable, str(WORKER), *map(str, arguments)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    print(result.stdout.strip())


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
