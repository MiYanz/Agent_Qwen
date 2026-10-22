# 只在当前 Python 进程内保存服务最近状态，供管理器和诊断代码查询。
# 这里没有定时探活线程，状态由 MCPManager 在启动、调用异常和关闭时主动更新。
from datetime import datetime


# 以服务名为键保存最后一次状态；这是诊断快照，不是独立监控服务。
class MCPHealth:
    """在内存中记录各 MCP Server 最近一次健康状态。"""

    # 新实例从空状态表开始，没有从磁盘恢复历史健康记录。
    def __init__(self):

        self.status = {}


    # server 是配置名，state 通常为 healthy/unhealthy/stopped，message 保存诊断原因。
    # 覆盖该服务原来的记录并写入当前时间，而不是累积每一次变化。
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



    # 指定服务时查单条，未记录返回空字典；不指定时返回内部状态映射本身，并非副本。
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



    # 按当前记录筛选健康服务；不发起真实探测，也不自动恢复曾失败的状态。
    def healthy_servers(self):
        """使用列表推导式返回当前状态为 healthy 的 Server 名称。"""

        return [

            name

            for name, info in self.status.items()

            if info["state"] == "healthy"

        ]
