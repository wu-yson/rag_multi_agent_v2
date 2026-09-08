"""MCP 客户端连接层：本地 stdio 客户端 + 工具加载。"""
import asyncio
import os
from contextvars import ContextVar, Token

from langchain_core.tools import StructuredTool
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

from src.config.settings import settings
from src.mcp.mcpclient_base import (
    BaseMCPClient,
    MCPClientConfig,
)
from src.utils.logger import log


class StdioMCPClient(BaseMCPClient):
    async def _open_transport(self, stack):
        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
        )
        return await stack.enter_async_context(stdio_client(params))


# ===== 本地工具白名单：按工具名过滤 =====
DOC_TOOL_NAMES = {
    'read_file', 'read_lines', 'write_file', 'append_file',
    'replace_exact', 'replace_regex',
    'read_docx', 'write_docx', 'read_xlsx', 'write_xlsx',
    'list_dir', 'make_dir', 'glob_files', 'search_text',
}
RAG_TOOL_NAMES = {'rag_search', 'document_storage'}


WEB_TOOL_NAMES = set("web_search")

LOCAL_TOOL_NAMES = DOC_TOOL_NAMES | RAG_TOOL_NAMES | WEB_TOOL_NAMES

local_mcp = StdioMCPClient(
    MCPClientConfig(
        name='local_mcp',
        command='python',
        args=[settings.mcp_server_path],
        tool_whitelist=sorted(LOCAL_TOOL_NAMES),
        timeout=60,
    )
)

# 当前请求的会话上下文，按 session_id 隔离。
_current_session_id: ContextVar[str] = ContextVar('current_session_id', default='')
# 每个会话对应的工作根目录，由 begin_request 写入。
_session_roots: dict[str, str] = {}


async def get_tools():
    """返回本地 MCP 已加载并通过白名单的工具。"""
    return await local_mcp.get_tools()


async def close():
    """关闭本地 MCP 连接。"""
    await local_mcp.close()


async def get_doc_tools():
    """doc_agent 用的工具（白名单过滤）"""
    tools = await get_tools()
    return [_patch_tool_with_root(t) for t in tools if t.name in DOC_TOOL_NAMES]


async def get_rag_tools():
    """rag 检索/入库工具（白名单过滤 + 包装超时）。"""
    tools = await get_tools()
    return [
        _patch_tool_with_root(t, timeout=300 if t.name == 'document_storage' else 90)
        for t in tools if t.name in RAG_TOOL_NAMES
    ]


async def get_web_tools():
    """Web Agent 本地搜索工具接入口。"""

    tools = await get_tools()
    return [_patch_tool_with_root(t) for t in tools if t.name in WEB_TOOL_NAMES]


def set_workspace_root(session_id: str, path: str) -> None:
    """记录当前窗口的文件操作根目录（纯本地，不再依赖 MCP 全局状态）。"""
    # 接受 id 和路径存入字典里，做隔离。
    _session_roots[session_id] = path
    log.info(f"[MCP] 窗口 {session_id} 工作根: {path}")


def _patch_tool_with_root(tool, timeout: float = 90):
    """包装 MCP 工具：调用时若 file_path 是相对路径，补全成当前窗口根下的绝对路径。"""
    async def _run(**kwargs):
        kwargs = dict(kwargs)
        sid = _current_session_id.get()
        root = _session_roots.get(sid)
        if root:
            fp = kwargs.get("file_path")
            if isinstance(fp, str) and not os.path.isabs(fp):
                kwargs["file_path"] = os.path.join(root, fp)
        return await asyncio.wait_for(tool.ainvoke(kwargs), timeout=timeout)

    return StructuredTool.from_function(
        coroutine=_run,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


def begin_request(session_id: str, workspace_path: str = '') -> Token:
    """请求入口：记录当前窗口 session + 路径，返回 token 供 finally 重置。"""
    token = _current_session_id.set(session_id)
    if workspace_path:
        _session_roots[session_id] = workspace_path
    return token


def end_request(token: Token) -> None:
    """请求结束：重置上下文。"""
    _current_session_id.reset(token)
