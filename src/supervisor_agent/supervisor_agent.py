import json

from src.utils.logger import log
from typing import Optional, Any

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage

from src.base.agents_base import BaseAgentTemplate, BaseAgentConfig, GraphState
from src.llm.factory import llm_factory
from src.memory.memory import CommonMemory
from src.prompts import get_prompt
from src.supervisor_agent.graph_tool.graph import graph_invoke


TOOL_RESULT_MAX_CHARS = 300




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

    def _get_agent(
        self,
        tmp_model: Optional[str],
        tmp_tools: Optional[list[Any]],
        tmp_prompt: Optional[str]
    ) -> Any:
        """ 获取自定义的Agent实例 """
        if not any((tmp_model, tmp_tools, tmp_prompt)):
            return self.default_agent
        use_llm = llm_factory.get_client(tmp_model) if tmp_model else self._llm
        use_tools = tmp_tools if tmp_tools is not None else self.tools
        use_prompt = tmp_prompt if tmp_prompt else self.system_prompt

        log.info(f" [TopSupervisor] 初始化自定义Agent")
        return create_agent(
            model=use_llm,
            system_prompt=use_prompt,
            tools=use_tools,
            middleware=[]
        )

    def _build_messages(
        self,
        user_input: str,
        history: Optional[list[BaseMessage]],
        system_prompt: Optional[str],
    ):
        """ 构建消息列表 """
        msg_list = []

        if system_prompt and system_prompt.strip():
            msg_list.append(SystemMessage(content=system_prompt))
        else:
            msg_list.append(SystemMessage(content=self.system_prompt))

        if self._memory:
            try:
                history_items = self._memory.get_recent()
                history_parts = ["【历史会话记录】以下为对话历史，仅作背景参考："]
                for item in history_items:
                    if item["role"] == "human":
                        history_parts.append(f"【人类消息】{item['content']}")
                    elif item["role"] == "ai":
                        history_parts.append(f"【AI回复】{item['content']}")
                    elif item["role"] == "tool":
                        history_parts.append(f"【工具执行日志】{item['content']}")
                history_text = "\n".join(history_parts)
                msg_list.append(SystemMessage(content=history_text))
            except Exception as e:
                log.error(f" [TopSupervisor] 获取历史会话失败: {e}")
        if history:
            msg_list.extend(history)
        msg_list.append(HumanMessage(content=user_input))
        return msg_list

    def _security_detect(self, user_input: str) -> bool:
        """ 提示词攻击安全检测"""
        detect_prompt = get_prompt("prompt_injection_detect_prompt")
        messages = [
            SystemMessage(content=detect_prompt),
            HumanMessage(content=user_input)
        ]
        # 调用LLM获取判定结果
        resp = self.llm.invoke(messages)
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



    def invoke(
        self,
        user_input: str,
        history: Optional[list[BaseMessage]] = None,
        system_prompt: Optional[str] = None,
        tmp_model: Optional[str] = None,
        tmp_tools: Optional[list[Any]] = None,
        tmp_prompt: Optional[str] = None,
    ) -> str:
        """
        调用智能体
        :param user_input: 用户输入
        :param history: 历史消息列表
        :param system_prompt: 系统提示词
        :param tmp_model: 临时模型
        :param tmp_tools: 临时工具列表
        :param tmp_prompt: 临时提示词
        :return: 智能体输出
        """
        try:
            # 前置安全检测
            is_attack = self._security_detect(user_input)
            if is_attack:
                intercept_text = "此为攻击行为, 结束此次会话"
                return intercept_text


            # 正常请求，执行原有全部业务逻辑
            messages = self._build_messages(user_input, history, system_prompt)

            agent = self._get_agent(tmp_model, tmp_tools, tmp_prompt)
            log.info(f"[TopSupervisor] 开始构建主层Agent")
            resp = agent.invoke({"messages": messages})
            msg_list = resp["messages"]
            last_msg = msg_list[-1]
            reply = last_msg.content

            tool_call_names = sorted({
                item["name"]
                for msg in msg_list
                for item in getattr(msg, "tool_calls", [])
            })
            if tool_call_names:
                log.info(f"[TopSupervisor] LLM决策：调用工具，工具列表：{tool_call_names}")
            else:
                log.info("[TopSupervisor] LLM决策：无工具调用，直接输出回复")

            log.info(f"[TopSupervisor] 顶层Agent推理完成")


            if self._memory:
                for msg in msg_list:
                    if isinstance(msg, HumanMessage):
                        self._memory.add(role="human", content=msg.content)
                    elif isinstance(msg, AIMessage):
                        self._memory.add(role="ai", content=msg.content)
                    elif isinstance(msg, ToolMessage):
                        tool_content = msg.content if isinstance(msg.content, str) else str(msg.content)
                        self._memory.add(role="tool", content=tool_content[:TOOL_RESULT_MAX_CHARS])
            return reply
        except Exception as e:
            log.error(f" [TopSupervisor] 调用大模型失败: {e}", exc_info=True)
            if self.config.debug_mode:
                raise RuntimeError(e) from e
            return f"智能体调用失败: {e}"


    def invoke_wrapper(self, state: GraphState) -> GraphState:
        raise RuntimeError("顶层主Agent不作为LangGraph节点调用，该方法禁止执行")

    def _get_error_tip(self) -> str:
        raise RuntimeError("顶层主Agent不作为LangGraph节点调用，该方法禁止执行")


