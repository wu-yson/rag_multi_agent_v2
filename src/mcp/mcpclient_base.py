import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from src.utils.logger import log
from mcp import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools




class MCPUnavailableError(RuntimeError):
    """MCP 服务不可用时抛出"""
    pass



@dataclass
class MCPClientConfig:
    name: str
    timeout: int = 60
    tool_whitelist: list[str] = field(default_factory=list)

    # 本地 stdio MCP 使用
    command: str = ""
    args: list[str] = field(default_factory=list)



class BaseMCPClient:

    def __init__(self, config: MCPClientConfig):
        self.config = config
        self._exit_stack: AsyncExitStack | None = None
        self._tools: list | None = None
        self._session: ClientSession | None = None
        self._tools_lock = asyncio.Lock()

    async def get_tools(self):
        if self._tools is not None:
            return self._tools

        async with self._tools_lock:
            if self._tools is not None:
                return self._tools

            stack = AsyncExitStack()
            try:
                read, write = await self._open_transport(stack)

                session = await stack.enter_async_context(
                    ClientSession(read, write)
                )
                await asyncio.wait_for(
                    session.initialize(),
                    timeout=self.config.timeout
                )
                tools = await asyncio.wait_for(
                    load_mcp_tools(session),
                    timeout=self.config.timeout
                )
                self._tools = self._filter_tools(tools)
                self._session = session
                self._exit_stack = stack
                return self._tools
            except Exception as e:
                log.error(f"MCP 服务连接失败: {e}")
                try:
                    await stack.aclose()
                except Exception as close_err:
                    log.warning(f"MCP 临时资源关闭失败: {close_err}")
                raise

    async def _open_transport(self, stack):
        """ 子类需要实现 """
        raise NotImplementedError

    def _filter_tools(self, tools):
        whitelist = set(self.config.tool_whitelist)
        filtered = [tool for tool in tools if tool.name in whitelist]

        if not filtered:
            raise MCPUnavailableError(
                f"MCP 未提供白名单工具: {sorted(whitelist)}"
            )

        return filtered

    async def close(self):
        if not self._exit_stack:
            return

        try:
            await self._exit_stack.aclose()
        except Exception as e:
            log.warning(f"MCP 连接关闭异常: {e}")
        finally:
            self._exit_stack = None
            self._session = None
            self._tools = None