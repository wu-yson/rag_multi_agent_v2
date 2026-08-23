
from typing import Any
from src.prompts import get_prompt
from src.base.agents_base import BaseAgentTemplate, BaseAgentConfig, NodeKeyBase
from src.doc_agent import doc_tools
from src.doc_agent.doc_tools.registry import tool_registry
from src.doc_agent.doc_agent_skill import get_doc_agent_skill
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

    @property
    def tools(self) -> list[Any]:
        if self._tools is None:
            self._tools = [get_doc_agent_skill] + tool_registry.get_all()
        return self._tools

    @property
    def output_key(self) -> str:
        return NodeKeyBase.DOC_AGENT

    def _get_error_tip(self) -> str:
        return "本地文档处理服务临时出错，请重试"

# 唯一实例
doc_agent = DocumentAgent()
# 图节点注册
agents_graph.register_sub_agent(doc_agent.output_key, doc_agent)
