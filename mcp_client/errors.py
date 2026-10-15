class MCPError(Exception):
    """
    MCP统一异常基类
    """
    pass



class MCPConnectionError(MCPError):
    """
    MCP连接失败
    """
    pass



class MCPToolError(MCPError):
    """
    MCP工具执行失败
    """
    pass



class MCPServerUnavailable(MCPError):
    """
    MCP Server不可用
    """
    pass