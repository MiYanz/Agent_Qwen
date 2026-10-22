# 项目自己的 MCP 异常分类，便于调用方区分连接失败、服务不可用和工具执行失败。
# 这些类型本身不执行重试，也不负责启动或关闭服务。
# 所有项目 MCP 异常的公共父类，调用方需要统一处理时可捕获这一类型。
class MCPError(Exception):
    """
    MCP 统一异常基类，便于上层统一捕获所有 MCP 相关故障。
    """
    pass



# 表示连接或生命周期约束失败，例如跨事件循环使用 Session 或所有服务启动失败。
class MCPConnectionError(MCPError):
    """
    MCP 连接或整体生命周期管理失败。
    """
    pass



# 表示已选中服务后工具请求执行失败；原始异常通常通过 raise ... from ... 保留。
class MCPToolError(MCPError):
    """
    MCP 工具已经发起调用，但执行过程失败。
    """
    pass



# 表示配置服务没有可用 Session；与已经发出请求后的执行失败区分。
class MCPServerUnavailable(MCPError):
    """
    目标 MCP Server 未成功启动或当前没有可用 Session。
    """
    pass
