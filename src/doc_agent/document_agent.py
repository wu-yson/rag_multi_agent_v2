
from typing import Any

from src.mcp.client import get_doc_tools
from src.prompts import get_prompt
from src.base.agents_base import BaseAgentTemplate, BaseAgentConfig, NodeKeyBase
from src.supervisor_agent.graph_tool.graph import agents_graph



class DocumentAgent(BaseAgentTemplate):
    """ 文件操作智能体
        文件操作智能体，可操作目标目录内的文件写入和读取
    """


    def __init__(self):
        cfg = BaseAgentConfig()
        super().__init__(config=cfg)

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = get_prompt("doc_agent_prompt")
        return self._system_prompt

    async def _load_tools(self):
        """doc 工具从 MCP 加载"""
        return await get_doc_tools()


    @property
    def output_key(self) -> str:
        return NodeKeyBase.DOC_AGENT

    def _get_error_tip(self) -> str:
        return "本地文档处理服务临时出错，请重试"

# 唯一实例
doc_agent = DocumentAgent()
# 图节点注册
agents_graph.register_sub_agent(doc_agent.output_key, doc_agent)
