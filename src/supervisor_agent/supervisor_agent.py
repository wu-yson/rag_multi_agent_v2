import json

from src.llm.resilience import build_agent_middleware
from src.config.settings import settings
from src.utils.logger import log
from typing import Optional, Any

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessageChunk, AIMessage

from src.base.agents_base import BaseAgentTemplate, BaseAgentConfig, GraphState
from src.llm.factory import llm_factory
from src.memory.memory import CommonMemory
from src.prompts import get_prompt
from src.supervisor_agent.graph_tool.graph import graph_invoke






class SupervisorAgent(BaseAgentTemplate):
    """ 主智能体 负责协调调度 记忆存档 """

    def __init__(self, memory: Optional[CommonMemory] = None):
        cfg = BaseAgentConfig()
        super().__init__(config=cfg)
        self._memory = memory

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = get_prompt("supervisor_agent_prompt")
        return self._system_prompt

    @property
    def tools(self) -> list[Any]:
        if self._tools is None:
            self._tools = [graph_invoke]
        return self._tools

    async def _get_agent(
        self,
        tmp_model: Optional[str],
        tmp_tools: Optional[list[Any]],
    ) -> Any:
        """ 获取自定义的Agent实例 """
        if not any((tmp_model, tmp_tools)):
            return await self.get_agent()
        use_llm = llm_factory.get_client(tmp_model) if tmp_model else self._llm
        use_tools = tmp_tools if tmp_tools is not None else self.tools
        use_prompt = self.system_prompt

        log.info(f" [TopSupervisor] 初始化自定义Agent")
        return create_agent(
            model=use_llm,
            system_prompt=use_prompt,
            tools=use_tools,
            middleware=build_agent_middleware(),
        )

    def _build_messages(
        self,
        user_input: str,
    ):
        """ 构建消息列表 """
        msg_list = [SystemMessage(content=self.system_prompt)]

        if self._memory:
            try:
                for item in self._memory.get_recent():  # 直接取
                    if item["role"] == "human":
                        msg_list.append(HumanMessage(content=item["content"]))
                    elif item["role"] == "ai":
                        msg_list.append(AIMessage(content=item["content"]))

            except Exception as e:
                log.error(f" [TopSupervisor] 获取历史会话失败: {e}")

        msg_list.append(HumanMessage(content=user_input))
        return msg_list

    async def _security_detect(self, user_input: str) -> bool:
        """ 提示词攻击安全检测"""
        detect_prompt = get_prompt("prompt_injection_detect_prompt")
        messages = [
            SystemMessage(content=detect_prompt),
            HumanMessage(content=user_input)
        ]
        # 调用LLM获取判定结果
        resp = await self.llm.ainvoke(messages)
        raw_content = resp.content.strip()

        try:
            # 解析json字符串为字典
            json_data = json.loads(raw_content)
            # 获取布尔标记
            is_attack = json_data.get("Safety_inspection", False)
            if is_attack:
                log.warning(f" [TopSupervisor] 提示词注入攻击已拦截，原始输入：{user_input}")
            return is_attack
        except json.JSONDecodeError:
            # json解析失败，默认放行，避免正常请求卡死
            log.error(f" [TopSupervisor] 安全检测返回内容JSON解析失败，内容：{raw_content}")
            return False


    def _save_memory(self, user_input: str, full_text: list[str], all_msgs: list[BaseMessage]):
        """保存本轮 human / ai / tool 到记忆"""
        if self._memory:
            self._memory.add(role="human", content=user_input)
            self._memory.add(role="ai", content="".join(full_text))
            for msg in all_msgs:
                if isinstance(msg, ToolMessage):
                    tool_content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    self._memory.add(role="tool", content=tool_content)

    async def astream(
        self,
        user_input: str,
        tmp_model: Optional[str] = None,
        tmp_tools: Optional[list[Any]] = None,
        tmp_prompt: Optional[str] = None,
    ):
        """流式调用智能体：逐块产出回答增量（异步生成器）"""
        try:
            # 前置安全检测（开关控制，默认关闭）
            if settings.agent_security_check:
                is_attack = await self._security_detect(user_input)
                if is_attack:
                    yield "此为用户输入被拦截（攻击行为）, 结束本次会话"
                    return

            messages = self._build_messages(user_input)
            agent = await self._get_agent(tmp_model, tmp_tools)
            log.info(f"[TopSupervisor] 开始构建主层Agent")

            full_text: list[str] = []
            all_msgs: list[BaseMessage] = []          # 收集所有消息（含工具消息）
            async for message, _meta in agent.astream(
                {"messages": messages},
                stream_mode="messages",
                config={"recursion_limit": 8},
            ):
                all_msgs.append(message)

                if isinstance(message, AIMessageChunk) and message.content:
                    chunk = str(message.content)
                    full_text.append(chunk)
                    yield chunk

            log.info(f"[TopSupervisor] 顶层Agent推理完成")
            self._save_memory(user_input, full_text, all_msgs)
        except Exception as e:
            log.error(f" [TopSupervisor] 流式调用失败: {e}", exc_info=True)
            if self.config.debug_mode:
                raise RuntimeError(e) from e
            yield f"智能体调用失败: {e}"


    def ainvoke_wrapper(self, state: GraphState) -> GraphState:
        raise RuntimeError("顶层主Agent不作为LangGraph节点调用，该方法禁止执行")

    def _get_error_tip(self) -> str:
        raise RuntimeError("顶层主Agent不作为LangGraph节点调用，该方法禁止执行")
