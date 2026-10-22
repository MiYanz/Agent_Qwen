# 最小真实 MCP 调用示例：直接请求 Filesystem 列出项目目录。
# 本文件末尾没有 __main__ 保护，导入时也会启动事件循环及真实 MCP，自动收集测试时要留意。
import asyncio

from mcp_client.manager import MCPManager
from mcp_client.tools import CONFIG_PATH



# 用默认 filesystem 服务列目录，展示无需经过模型也能直接请求 MCP。
# 返回对象直接打印，和主图中 ToolMessage 回到模型后的最终回答不同。
async def main():

    """直接通过 MCPManager 调用 Filesystem 的目录列表工具。"""


    async with MCPManager(CONFIG_PATH) as manager:
        result = await manager.call_tool(

        "list_directory",

        {
            "path":"D:\\46-Agent"
        }

        )


        print(result)



# asyncio.run 创建顶层事件循环；测试脚本外部没有正在运行的 loop。
asyncio.run(main())
