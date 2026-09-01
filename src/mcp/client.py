"""MCP 客户端连接层：常驻会话 + 加载工具"""
import asyncio
import contextvars
import os
from contextlib import AsyncExitStack

from langchain_core.tools import StructuredTool

from src.utils.logger import log
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from src.config.settings import settings

class MCPUnavailableError(RuntimeError):
    """MCP 服务不可用时抛出"""
    pass


if not settings.mcp_server_path:
    raise MCPUnavailableError("未配置 MCP_SERVER_PATH，请在 .env 中设置 MCP 服务端脚本路径")
SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=[settings.mcp_server_path],
)



_tools_lock = asyncio.Lock()




# 全局状态：会话只建一次，工具只加载一次
_exit_stack: AsyncExitStack | None = None
_tools: list | None = None
_session: ClientSession | None = None

# ===== 窗口路径隔离：session_id -> 工作根 =====
_current_session_id: contextvars.ContextVar[str] = contextvars.ContextVar("current_session_id", default="")
_session_roots: dict[str, str] = {}



async def get_tools():
    global _exit_stack, _tools, _session
    if _tools is not None:
        return _tools

    async with _tools_lock:                       # 锁在最外层
        if _tools is not None:                    # 双检
            return _tools
        try:
            _exit_stack = AsyncExitStack()
            read, write = await _exit_stack.enter_async_context(stdio_client(SERVER_PARAMS))
            session = await _exit_stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(session.initialize(), timeout=60)
            _tools = await asyncio.wait_for(load_mcp_tools(session), timeout=60)
            _session = session
            return _tools
        except Exception as e:
            log.error(f"MCP 服务连接失败: {e}")
            _exit_stack = None  # 新增：清掉，避免 close() 重复关闭已取消的连接
            _tools = None
            raise MCPUnavailableError(
                    "MCP 工具服务未启动或连接失败，请先启动 MCP 服务"
                ) from e

async def close():
    """关闭 MCP 连接（用完调用）"""
    global _exit_stack, _tools, _session
    if _exit_stack is not None:
        try:
            await _exit_stack.aclose()
        except Exception as e:
            log.warning(f"关闭 MCP 连接时忽略异常: {e}")
        _exit_stack = None
        _tools = None
        _session = None



# ===== 白名单：按工具名过滤 =====
DOC_TOOL_NAMES = {
    "read_file", "read_lines", "write_file", "append_file",
    "replace_exact", "replace_regex",
    "read_docx", "write_docx", "read_xlsx", "write_xlsx",
    "list_dir", "make_dir", "glob_files", "search_text",
}
RAG_TOOL_NAMES = {"rag_search", "document_storage"}


async def get_doc_tools():
    """doc_agent 用的工具（白名单过滤）"""
    tools = await get_tools()
    return [_patch_tool_with_root(t) for t in tools if t.name in DOC_TOOL_NAMES]


async def get_rag_tools():
    """rag_agent 用的工具（白名单过滤）"""
    tools = await get_tools()
    return [t for t in tools if t.name in RAG_TOOL_NAMES]


def set_workspace_root(session_id: str, path: str) -> None:
    """记录当前窗口的文件操作根目录（纯本地，不再依赖 MCP 全局状态）"""
    _session_roots[session_id] = path
    log.info(f"[MCP] 窗口 {session_id} 工作根: {path}")



def _patch_tool_with_root(tool):
    """包装 MCP 工具：调用时若 file_path 是相对路径，补全成当前窗口根下的绝对路径"""
    async def _run(**kwargs):
        kwargs = dict(kwargs)
        sid = _current_session_id.get()
        root = _session_roots.get(sid)
        if root:
            fp = kwargs.get("file_path")
            if isinstance(fp, str) and not os.path.isabs(fp):
                kwargs["file_path"] = os.path.join(root, fp)
        return await tool.ainvoke(kwargs)
    return StructuredTool.from_function(
        coroutine=_run,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


def begin_request(session_id: str, workspace_path: str = "") -> contextvars.Token:
    """请求入口：记录当前窗口 session + 路径，返回 token 供 finally 重置"""
    token = _current_session_id.set(session_id)
    if workspace_path:
        _session_roots[session_id] = workspace_path
    return token


def end_request(token: contextvars.Token) -> None:
    """请求结束：重置上下文"""
    _current_session_id.reset(token)










if __name__ == '__main__':
    async def main():
        try:
            res = await get_tools()
            print(len(res))
            for i in res:
                print("-", i.name)
            doc = await get_doc_tools()
            rag = await get_rag_tools()
            print("doc 工具数:", len(doc), "| rag 工具数:", len(rag))
        finally:
            await close()   # 和 get_tools() 在同一个循环里

    asyncio.run(main())     # 只开一个循环，建和关都在里面


