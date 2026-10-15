import asyncio

from mcp_client.manager import MCPManager
from mcp_client.tools import CONFIG_PATH



async def main():


    async with MCPManager(CONFIG_PATH) as manager:
        result = await manager.call_tool(

        "list_directory",

        {
            "path":"D:\\46-Agent"
        }

        )


        print(result)



asyncio.run(main())
