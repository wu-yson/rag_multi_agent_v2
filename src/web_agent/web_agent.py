"""Web 搜索节点：纯执行，不用 LLM。直接调 MCP web_search 工具。"""
from src.base.agents_base import NodeKeyBase, run_node
from src.mcp.client import get_web_tools, pick_tool
from src.supervisor_agent.graph_tool.graph import agents_graph
from src.utils.logger import log


class WebSearchNode:
    """联网搜索节点：把任务内容作为 query 直接调用 web_search。"""
    output_key = NodeKeyBase.WEB_SEARCH

    async def ainvoke_wrapper(self, state):
        return await run_node(state, self.output_key, self._web_search)

    async def _web_search(self, content):
        tools = await get_web_tools()
        tool = pick_tool(tools, "web_search")
        result = await tool.ainvoke({"query": content})

        text = str(result).strip()
        log.info(f"[WebSearchNode] 搜索完成，返回 {len(text)} 字")
        return text


# 图节点注册
web_search_node = WebSearchNode()
agents_graph.register_sub_agent(web_search_node.output_key, web_search_node)
