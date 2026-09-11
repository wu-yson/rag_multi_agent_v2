"""简单注册器：把普通函数 / 工具类方法注册成 MCP 工具"""
import asyncio
import inspect

from mcp.server.fastmcp import FastMCP
from mcp_src.utils.logger import log

def _wrap(fn):
    """包装：同步函数放线程池执行（避免阻塞 asyncio 事件循环），并保留原函数签名"""
    async def _inner(*args, **kwargs):
        fn_name = fn.__name__
        log.info(f"[MCP] 调用工具: {fn_name}")
        try:
            result = await asyncio.to_thread(fn, *args, **kwargs)
            return result
        except Exception as e:
            log.error(f"[MCP] 工具异常: {fn_name}: {e}")
            raise
    _inner.__name__ = fn.__name__
    _inner.__doc__ = fn.__doc__
    _inner.__signature__ = inspect.signature(fn)
    return _inner

def register_functions(mcp: FastMCP, funcs: list) -> int:
    """注册一批普通函数（名字=函数名，描述=docstring，参数=签名）"""
    cnt = 0
    for fn in funcs:
        wrapped = _wrap(fn)
        mcp.add_tool(wrapped, name=fn.__name__, description=inspect.getdoc(fn) or "")
        cnt += 1
    return cnt


def register_toolkit(mcp: FastMCP, toolkit, skip_names=None) -> int:
    """把一个工具类的公开方法注册成 MCP 工具"""
    skip = set(skip_names or [])
    cnt = 0
    for name in dir(toolkit):
        if name.startswith("_") or name in skip:
            continue
        attr = getattr(toolkit, name)
        if not callable(attr):
            continue
        wrapped = _wrap(attr)
        mcp.add_tool(wrapped, name=name, description=inspect.getdoc(attr) or "")
        cnt += 1
    return cnt
