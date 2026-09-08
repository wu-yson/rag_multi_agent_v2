from src.mcp.client import get_web_tools
from src.prompts import get_prompt
from src.base.agents_base import BaseAgentTemplate, BaseAgentConfig, NodeKeyBase
from src.supervisor_agent.graph_tool.graph import agents_graph


class WebAgent(BaseAgentTemplate):
    """联网搜索智能体。"""

    def __init__(self):
        cfg = BaseAgentConfig()
        super().__init__(config=cfg)

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = get_prompt('web_agent_prompt')
        return self._system_prompt

    async def _load_tools(self):
        """web 工具从 MCP 加载"""
        return await get_web_tools()

    @property
    def output_key(self) -> str:
        return NodeKeyBase.WEB_AGENT

    def _get_error_tip(self) -> str:
        return '联网搜索服务暂时不可用，请稍后重试'


web_agent = WebAgent()
agents_graph.register_sub_agent(web_agent.output_key, web_agent)
