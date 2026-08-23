
from typing import Any
from src.prompts import get_prompt
from src.base.agents_base import BaseAgentTemplate, BaseAgentConfig, NodeKeyBase
from src.rag_agent.rag_tool import document_storage, rag_search
from src.supervisor_agent.graph_tool.graph import agents_graph




class RAGAgent(BaseAgentTemplate):
    """ rag智能体
        向量知识库智能体，支持目录文档向量入库、建立索引，同时可以检索向量库内已存入的文档内容，解答文档相关问题
    """


    def __init__(self):
        cfg = BaseAgentConfig()
        super().__init__(config=cfg)

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = get_prompt("rag_agent_prompt")
        return self._system_prompt

    @property
    def tools(self) -> list[Any]:
        if self._tools is None:
            self._tools = [rag_search, document_storage]
        return self._tools

    @property
    def output_key(self) -> str:
        return NodeKeyBase.RAG_AGENT

    def _get_error_tip(self) -> str:
        return "文档检索服务临时出错，请重试"

# 唯一实例
rag_agent = RAGAgent()
# 图节点注册
agents_graph.register_sub_agent(rag_agent.output_key, rag_agent)
