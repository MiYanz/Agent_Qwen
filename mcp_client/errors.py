class MCPError(Exception):
    """
    MCP 统一异常基类，便于上层统一捕获所有 MCP 相关故障。
    """
    pass



class MCPConnectionError(MCPError):
    """
    MCP 连接或整体生命周期管理失败。
    """
    pass



class MCPToolError(MCPError):
    """
    MCP 工具已经发起调用，但执行过程失败。
    """
    pass



class MCPServerUnavailable(MCPError):
    """
    目标 MCP Server 未成功启动或当前没有可用 Session。
    """
    pass
