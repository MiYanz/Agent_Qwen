import asyncio

from mcp_client.manager import MCPManager
from mcp_client.tools import CONFIG_PATH



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
