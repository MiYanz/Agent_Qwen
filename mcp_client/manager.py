# MCP 资源所有者：启动子进程、完成握手、缓存工具，并在关闭时释放会话。
# Session 是事件循环相关的异步资源；启动、调用和退出必须在同一事件循环中完成。
import asyncio
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from mcp_client.client import MCPClient
from mcp_client.errors import (
    MCPConnectionError,
    MCPServerUnavailable,
    MCPToolError,
)
from mcp_client.health import MCPHealth


# 每个实例统一持有一组服务进程、Session、工具缓存和每服务调用锁。
# 应用 lifespan 持有此实例，主 Agent 与专业 Agent 因而能复用同一套 MCP。
class MCPManager:
    """Own long-lived MCP stdio processes and client sessions."""

    # Manager 与 MCPClient 的区别是：Manager 会让子进程和 Session 在应用期间持续存活。

    # 初始化配置和空的资源容器；锁及所属事件循环在第一次异步启动时确定。
    def __init__(self, config_path="mcp_client/config.json"):
        self.client = MCPClient(config_path)    # 创建 MCP 客户端
        self.servers = self.client.servers  # 保存服务列表
        self.health = MCPHealth()   # 创建健康状态管理器，记录每个服务是否正常、发生了什么错误。
        self.tools = [] # 初始化工具列表
        self.sessions = {}  # 保存各个 MCP Server 对应的客户端会话
        self.started = False    # 使用布尔值记录整个 MCP 管理器是否已经完成启动。

        # 每个 AsyncExitStack 记录对应 Server 的异步清理回调。
        self._server_stacks = {}
        # 每个 Server 使用独立锁，避免同一 Session 被并发 call_tool 交叉写入。
        self._server_locks = {}
        # 生命周期锁保证并发请求不会重复启动整套 MCP Server。
        self._lifecycle_lock = None
        # MCP 的 stdio Session 必须始终由创建它的同一个事件循环使用。
        self._loop = None

    # 返回当前 loop；实例已归属其他 loop 时立即报错，避免对旧 Session 跨循环读写。
    def _assert_owner_loop(self):
        """检查事件循环是否一致，防止在错误的异步运行环境中操作 MCP 会话。"""
        current_loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not current_loop:
            raise MCPConnectionError(
                "MCPManager must be started, used, and stopped on the same "
                "asyncio event loop."
            )
        return current_loop

    # 幂等启动：已启动时直接返回，否则顺序建立服务并缓存发现结果。
    # 部分服务故障会记录健康状态；全部失败或任务被取消时清理已创建资源。
    async def start(self):
        """启动配置中的 MCP 服务，建立通信会话，并获取每个服务提供的工具。"""
        current_loop = self._assert_owner_loop()
        if self._lifecycle_lock is None:
            self._lifecycle_lock = asyncio.Lock()

        # async with 获取异步锁；等待锁时不会阻塞其他协程运行。
        async with self._lifecycle_lock:
            if self.started:
                return

            self._loop = current_loop
            self.tools = []
            failures = []
            print("[MCP Manager] Starting...", flush=True)

            for server_name in self.servers:
                # 每个 Server 单独使用一个栈，启动失败时只回滚该 Server 的资源。
                stack = AsyncExitStack()
                try:
                    server_params = self.client.get_server_params(server_name)
                    # enter_async_context 把连接的退出逻辑登记到栈中，关闭时逆序执行。
                    read, write = await stack.enter_async_context(
                        stdio_client(server_params)
                    )
                    session = await stack.enter_async_context(
                        ClientSession(read, write)
                    )
                    # MCP 握手完成后才能安全查询 Server 的工具列表。
                    await session.initialize()
                    server_tools = (await session.list_tools()).tools

                    self.sessions[server_name] = session
                    self._server_stacks[server_name] = stack
                    self._server_locks[server_name] = asyncio.Lock()
                    # 保留 server_name，后续转换 StructuredTool 时才能路由回正确 Session。
                    self.tools.extend(
                        (server_name, tool) for tool in server_tools
                    )
                    self.health.update(server_name, "healthy")
                    print(
                        f"[MCP Manager] {server_name}: healthy "
                        f"({len(server_tools)} tools)",
                        flush=True,
                    )
                except Exception as exc:
                    # 普通 Server 故障被隔离，其他已配置 Server 仍继续启动。
                    await stack.aclose()
                    failures.append((server_name, exc))
                    self.health.update(server_name, "unhealthy", str(exc))
                    print(
                        f"[MCP Manager] {server_name}: unhealthy - "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                # 取消与退出不属于普通服务故障，必须释放已打开的所有资源后继续向外传播。
                except BaseException:
                    # Cancellation and process shutdown are BaseException in
                    # modern asyncio. Roll back every server already opened.
                    await stack.aclose()
                    await self._close_resources()
                    self._reset_runtime_state()
                    raise

            # 只有至少一个服务成功才认为管理器已启动；部分不可用的服务稍后调用时会明确报错。
            if not self.sessions:
                await self._close_resources()
                self._loop = None
                details = "; ".join(
                    f"{name}: {error}" for name, error in failures
                )
                raise MCPConnectionError(
                    f"No MCP server could be started. {details}"
                )

            self.started = True
            print(
                f"[MCP Manager] Started {len(self.sessions)}/"
                f"{len(self.servers)} servers",
                flush=True,
            )

    # 确保实例已启动后返回缓存列表的浅拷贝，调用方增删列表不改变管理器缓存。
    async def get_tools(self):
        """获取可用工具列表"""
        await self.start()
        return list(self.tools)

    # 根据 server_name 找 Session，在该服务锁内等待真实 MCP 请求完成。
    # 服务缺失和执行异常分别包装为项目异常；调用成功时返回 MCP 原始对象。
    async def call_tool(self, tool_name, arguments, server_name="filesystem"):
        """将工具调用请求发送给指定 MCP Server，并返回执行结果。"""
        await self.start()
        self._assert_owner_loop()

        # 通过服务名路由而非只凭工具名，避免不同服务的同名工具误调用。
        session = self.sessions.get(server_name)
        if session is None:
            health = self.health.get(server_name)
            detail = health.get("message", "server was not started")
            raise MCPServerUnavailable(f"{server_name}: {detail}")

        try:
            # 锁只串行化同一个 Server，不同 Server 仍可并发执行。
            async with self._server_locks[server_name]:
                return await session.call_tool(tool_name, arguments)
        except Exception as exc:
            self.health.update(server_name, "unhealthy", str(exc))
            raise MCPToolError(
                f"MCP tool {server_name}.{tool_name} failed: {exc}"
            ) from exc

    # 释放全部已登记资源并重置运行状态；若关闭失败，清理尝试完后再汇总报错。
    async def stop(self):
        """停止 MCP 服务，关闭已经启动的 MCP 会话和相关进程，释放资源。"""
        if not self.started and not self._server_stacks:
            return

        self._assert_owner_loop()
        print("[MCP Manager] Stopping...", flush=True)
        errors = await self._close_resources()
        self._reset_runtime_state()
        print("[MCP Manager] Stopped", flush=True)

        if errors:
            details = "; ".join(
                f"{name}: {error}" for name, error in errors
            )
            raise MCPConnectionError(
                f"One or more MCP servers did not stop cleanly: {details}"
            )

    # 按登记顺序的逆序关闭各服务的 AsyncExitStack，并收集普通关闭异常。
    # 单个栈关闭失败不阻止尝试关闭其他服务，返回值用于 stop 的汇总提示。
    async def _close_resources(self):
        """逐个关闭 MCP Server 对应的异步资源，并记录关闭失败的情况。"""
        errors = []
        # 后建立的资源先关闭；pop 后不再保留该服务栈引用，每个关闭结果分别记录。
        for server_name in reversed(list(self._server_stacks)):
            stack = self._server_stacks.pop(server_name)
            try:
                await stack.aclose()
                self.health.update(server_name, "stopped")
            except Exception as exc:
                errors.append((server_name, exc))
                self.health.update(server_name, "stop_error", str(exc))
                print(
                    f"[MCP Manager] Failed to stop {server_name}: {exc}",
                    flush=True,
                )
        return errors

    # 只清理进程内引用与启动标志，不删除配置、数据库或用户记忆文件。
    def _reset_runtime_state(self):
        """清空已经失效的运行期数据，为下一次启动做准备。"""
        self.sessions.clear()
        self._server_locks.clear()
        self.tools = []
        self.started = False
        self._loop = None
        self._lifecycle_lock = None

    # 使 async with MCPManager(...) 自动等待启动，并把当前实例交给 as 变量。
    async def __aenter__(self):
        # 支持 async with MCPManager(...)，进入时自动启动。
        await self.start()
        return self

    # async with 的主体正常结束或抛异常都会尝试 stop；没有返回 True 来吞掉主体异常。
    async def __aexit__(self, exc_type, exc_value, traceback):
        # 无论代码块正常结束还是抛出异常，离开时都释放 MCP 资源。
        await self.stop()
