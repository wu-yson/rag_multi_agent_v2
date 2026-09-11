"""MCP 工具服务入口：文件工具 + RAG 工具，直接注册工具类方法"""
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from auto_register import register_functions, register_toolkit

# 工具来源：工具类实例 + 普通函数
from mcp_src.doc_tool.office_file_io import office_tools
from mcp_src.doc_tool.text_file_io import text_tools
from mcp_src.doc_tool.paths import path_tools
from mcp_src.doc_tool.search import search_text_tool
from mcp_src.rag_tool.rag_tool import rag_search, document_storage
from mcp_src.web_tool.web_search_tool import web_search

from mcp_src.doc_tool import _helpers as helpers
from mcp_src.rag_tool.rag.factory import rag_factory, build_rag_default_config
from mcp_src.utils.logger import log


mcp = FastMCP("轻量化工具服务")

try:
    rag_factory.create_store(build_rag_default_config().vector_store_name)
    log.info("向量库预热完成")
except Exception as e:
    log.warning(f"向量库预热失败（首次调用时会重试）: {e}")


def set_workspace_root(path: str) -> str:
    """【内部工具】替换沙箱根占位符：本次任务的根目录由主项目传入。"""
    new_root = Path(path).resolve()
    new_root.mkdir(parents=True, exist_ok=True)
    helpers.sandbox_root = new_root
    for tk in (text_tools, office_tools, path_tools, search_text_tool):
        tk.sandbox_root = new_root
    return f"ok: {new_root}"


# 内部工具：主项目专用，不在任何 agent 白名单里
register_functions(mcp, [set_workspace_root])
# RAG 工具（普通函数）
register_functions(mcp, [rag_search, document_storage])
# 网络搜索工具（普通函数）
register_functions(mcp, [web_search])
# 文件工具（工具类方法，直接注册）
register_toolkit(mcp, office_tools, skip_names=["generate_tool_list"])
register_toolkit(mcp, text_tools, skip_names=["generate_tool_list"])
register_toolkit(mcp, path_tools, skip_names=["generate_tool_list"])
register_toolkit(mcp, search_text_tool, skip_names=["generate_tool_list"])

if __name__ == "__main__":
    mcp.run()
