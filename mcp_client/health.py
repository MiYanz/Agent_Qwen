from datetime import datetime


class MCPHealth:
    """在内存中记录各 MCP Server 最近一次健康状态。"""

    def __init__(self):

        self.status = {}


    def update(
        self,
        server,
        state,
        message=""
    ):
        """更新指定 Server 的状态、说明和记录时间。"""

        self.status[server] = {

            "state": state,

            "message": message,

            "time": datetime.now().isoformat()

        }



    def get(
        self,
        server=None
    ):
        """传入名称时返回单个状态，否则返回全部状态。"""

        if server:

            return self.status.get(
                server,
                {}
            )

        return self.status



    def healthy_servers(self):
        """使用列表推导式返回当前状态为 healthy 的 Server 名称。"""

        return [

            name

            for name, info in self.status.items()

            if info["state"] == "healthy"

        ]
