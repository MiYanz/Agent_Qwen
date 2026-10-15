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


class MCPManager:
    """Own long-lived MCP stdio processes and client sessions."""

    def __init__(self, config_path="mcp_client/config.json"):
        self.client = MCPClient(config_path)
        self.servers = self.client.servers
        self.health = MCPHealth()
        self.tools = []
        self.sessions = {}
        self.started = False

        self._server_stacks = {}
        self._server_locks = {}
        self._lifecycle_lock = None
        self._loop = None

    def _assert_owner_loop(self):
        current_loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not current_loop:
            raise MCPConnectionError(
                "MCPManager must be started, used, and stopped on the same "
                "asyncio event loop."
            )
        return current_loop

    async def start(self):
        """Start every configured server once and cache sessions and tools."""
        current_loop = self._assert_owner_loop()
        if self._lifecycle_lock is None:
            self._lifecycle_lock = asyncio.Lock()

        async with self._lifecycle_lock:
            if self.started:
                return

            self._loop = current_loop
            self.tools = []
            failures = []
            print("[MCP Manager] Starting...", flush=True)

            for server_name in self.servers:
                stack = AsyncExitStack()
                try:
                    server_params = self.client.get_server_params(server_name)
                    read, write = await stack.enter_async_context(
                        stdio_client(server_params)
                    )
                    session = await stack.enter_async_context(
                        ClientSession(read, write)
                    )
                    await session.initialize()
                    server_tools = (await session.list_tools()).tools

                    self.sessions[server_name] = session
                    self._server_stacks[server_name] = stack
                    self._server_locks[server_name] = asyncio.Lock()
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
                    await stack.aclose()
                    failures.append((server_name, exc))
                    self.health.update(server_name, "unhealthy", str(exc))
                    print(
                        f"[MCP Manager] {server_name}: unhealthy - "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                except BaseException:
                    # Cancellation and process shutdown are BaseException in
                    # modern asyncio. Roll back every server already opened.
                    await stack.aclose()
                    await self._close_resources()
                    self._reset_runtime_state()
                    raise

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

    async def get_tools(self):
        await self.start()
        return list(self.tools)

    async def call_tool(self, tool_name, arguments, server_name="filesystem"):
        await self.start()
        self._assert_owner_loop()

        session = self.sessions.get(server_name)
        if session is None:
            health = self.health.get(server_name)
            detail = health.get("message", "server was not started")
            raise MCPServerUnavailable(f"{server_name}: {detail}")

        try:
            async with self._server_locks[server_name]:
                return await session.call_tool(tool_name, arguments)
        except Exception as exc:
            self.health.update(server_name, "unhealthy", str(exc))
            raise MCPToolError(
                f"MCP tool {server_name}.{tool_name} failed: {exc}"
            ) from exc

    async def stop(self):
        """Close all sessions and stdio child processes in reverse order."""
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

    async def _close_resources(self):
        errors = []
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

    def _reset_runtime_state(self):
        self.sessions.clear()
        self._server_locks.clear()
        self.tools = []
        self.started = False
        self._loop = None
        self._lifecycle_lock = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.stop()
