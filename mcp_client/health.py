from datetime import datetime


class MCPHealth:

    def __init__(self):

        self.status = {}


    def update(
        self,
        server,
        state,
        message=""
    ):

        self.status[server] = {

            "state": state,

            "message": message,

            "time": datetime.now().isoformat()

        }



    def get(
        self,
        server=None
    ):

        if server:

            return self.status.get(
                server,
                {}
            )

        return self.status



    def healthy_servers(self):

        return [

            name

            for name, info in self.status.items()

            if info["state"] == "healthy"

        ]