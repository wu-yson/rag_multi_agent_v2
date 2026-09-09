"""知识库节点：纯执行，不用 LLM。检索 / 入库拆成两个节点，由图的 target_agent 路由。"""
import re
from src.base.agents_base import NodeKeyBase, run_node
from src.mcp.client import get_rag_tools, pick_tool
from src.supervisor_agent.graph_tool.graph import agents_graph
from src.utils.logger import log


def _extract_dir_path(content: str) -> str:
    """从任务描述里提取 Windows 绝对路径；提取不到返回空，工具会用默认目录。"""
    m = re.search(r"[A-Za-z]:[\\/][^\s，。；;]+", content)
    return m.group(0) if m else ""


class RagSearchNode:
    """知识库检索节点：直接调 rag_search，把 content 提纯后返回。"""
    output_key = NodeKeyBase.RAG_SEARCH

    async def ainvoke_wrapper(self, state):
        return await run_node(state, self.output_key, self._rag_search)

    async def _rag_search(self, content):
        tools = await get_rag_tools()
        tool = pick_tool(tools, "rag_search")
        result = await tool.ainvoke({"user_input": content})

        text = str(result).strip()
        log.info(f"[RagSearchNode] 检索完成，返回 {len(text)} 字")
        return text


class RagStorageNode:
    """知识库入库节点：直接调 document_storage。"""
    output_key = NodeKeyBase.RAG_STORAGE

    async def ainvoke_wrapper(self, state):
        return await run_node(state, self.output_key, self._rag_storage)

    async def _rag_storage(self, content):
        tools = await get_rag_tools()
        tool = pick_tool(tools, "document_storage")
        dir_path = _extract_dir_path(content)
        result = await tool.ainvoke({"dir_path": dir_path})

        text = str(result).strip()
        log.info(f"[RagStorageNode] 入库结果: {text[:120]}")
        return text


# 图节点注册
rag_search_node = RagSearchNode()
rag_storage_node = RagStorageNode()
agents_graph.register_sub_agent(rag_search_node.output_key, rag_search_node)
agents_graph.register_sub_agent(rag_storage_node.output_key, rag_storage_node)
