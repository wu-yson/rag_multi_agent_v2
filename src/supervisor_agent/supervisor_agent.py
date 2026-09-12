import json

from src.llm.resilience import build_agent_middleware
from src.config.settings import settings
from src.utils.logger import log
from typing import Optional, Any
from langchain_community.callbacks import get_openai_callback
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessageChunk, AIMessage
from langchain_core.tools import StructuredTool

from src.base.agents_base import BaseAgentTemplate, BaseAgentConfig, GraphState
from src.llm.factory import llm_factory
from src.memory.memory import CommonMemory
from src.prompts import get_prompt
from src.supervisor_agent.graph_tool.graph import graph_invoke


# 单次返回给主Agent的历史任务结果最大字符数
TASK_RESULT_MAX_CHARS = 5000






class SupervisorAgent(BaseAgentTemplate):
    """ 主智能体 负责协调调度 记忆存档 """

    def __init__(self, memory: Optional[CommonMemory] = None):
        cfg = BaseAgentConfig()
        super().__init__(config=cfg)
        self._memory = memory
        self.model = settings.supervisor_agent_model

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = get_prompt("supervisor_agent_prompt")
        return self._system_prompt

    @property
    def tools(self) -> list[Any]:
        if self._tools is None:
            self._tools = [graph_invoke]
            if self._memory:
                self._tools.append(self._build_task_result_tool())
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

    def _get_cached_tokens(self, messages_response):
        """统计缓存命中的输入 token（Prompt Caching）。

        兼容两种来源：
        1) langchain 标准 usage_metadata.input_token_details.cache_read（流式 chunk 常走这条）
        2) OpenAI 兼容格式 response_metadata.token_usage.prompt_tokens_details.cached_tokens
        同一条消息只取其一，避免重复计数。
        """
        total = 0
        for msg in messages_response:
            cached = 0
            um = getattr(msg, "usage_metadata", None) or {}
            cached = (um.get("input_token_details") or {}).get("cache_read", 0) or 0
            if not cached:
                rmeta = getattr(msg, "response_metadata", None) or {}
                usage = rmeta.get("token_usage", {}) or {}
                details = usage.get("prompt_tokens_details", {}) or {}
                cached = details.get("cached_tokens", 0) or 0
            total += cached
        return total

    def _save_memory(self, user_input: str, full_text: list[str], all_msgs: list[BaseMessage]):
        """保存本轮 human / ai / tool 到记忆"""
        if self._memory:
            self._memory.add(role="human", content=user_input)
            self._memory.add(role="ai", content="".join(full_text))
            for msg in all_msgs:
                if isinstance(msg, ToolMessage):
                    tool_content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    self._memory.add(role="tool", content=tool_content)
                    if msg.artifact is not None:
                        self._memory.add(
                            role="tool",
                            content=json.dumps(
                                {"__kind__": "graph_result", "tasks": msg.artifact},
                                ensure_ascii=False,
                            ),
                        )

    def _build_task_result_tool(self) -> StructuredTool:
        """按task_id查询历史子任务完整结果的主Agent工具"""
        async def _read_task_result(task_id: str) -> str:
            task_id = str(task_id).strip()
            if not task_id:
                return "task_id不能为空"
            for content in self._memory.get_tool_records():
                try:
                    record = json.loads(content)
                except (TypeError, ValueError):
                    continue
                if not isinstance(record, dict) or record.get("__kind__") != "graph_result":
                    continue
                task = (record.get("tasks") or {}).get(task_id)
                if not task:
                    continue
                agent_name = task.get("target_agent", "")
                text = task.get("error") or task.get("result") or "无输出"
                if len(text) > TASK_RESULT_MAX_CHARS:
                    text = text[:TASK_RESULT_MAX_CHARS] + "\n...（内容过长，已截断）"
                return f"task {task_id}（{agent_name}）完整结果：\n{text}"
            return f"未找到 task {task_id} 的历史执行结果"

        return StructuredTool.from_function(
            coroutine=_read_task_result,
            name="read_task_result",
            description=(
                "按 task_id 查询此前多子Agent工作流中某个任务的完整执行结果。"
                "仅当用户追问历史任务（尤其是中间节点）的详细内容时调用，不要用它重新执行任务。"
            ),
        )

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
                    yield 'content', "此为用户输入被拦截（攻击行为）, 结束本次会话"
                    return

            messages = self._build_messages(user_input)
            agent = await self._get_agent(tmp_model, tmp_tools)
            log.info(f"[TopSupervisor] 开始构建主层Agent")

            full_text: list[str] = []               # 消息块
            all_msgs: list[BaseMessage] = []          # 收集所有消息（含工具消息）

            with get_openai_callback() as cb:
                async for message, _meta in agent.astream(
                        {"messages": messages},
                        stream_mode="messages",
                        config={"recursion_limit": 8},
                ):
                    all_msgs.append(message)

                    if isinstance(message, AIMessageChunk):
                        reasoning = message.additional_kwargs.get(
                            'reasoning_content') if message.additional_kwargs else None
                        if reasoning:
                            yield 'thinking', str(reasoning)
                        if message.content:
                            chunk = str(message.content)
                            full_text.append(chunk)
                            yield 'content', chunk

            if not full_text:
                log.warning(" [TopSupervisor] 模型未返回正文内容（输出可能被思考占用），已回退为提示语")
                fallback_text = "模型本次未返回有效内容（输出可能被内部思考占用），请重试或换一种问法。"
                full_text.append(fallback_text)
                yield 'content', fallback_text

            log.info(f"[TopSupervisor] 顶层Agent推理完成")

            # 计算缓存
            cached = self._get_cached_tokens(all_msgs)
            log.info(f"[Token] prompt={cb.prompt_tokens}, completion={cb.completion_tokens}, cached={cached}, total={cb.total_tokens}")
            self._save_memory(user_input, full_text, all_msgs)

            # callback 自动统计了主 Agent + 所有子 Agent 的 LLM 调用
            yield ('usage', {
                'prompt_tokens': cb.prompt_tokens,
                'completion_tokens': cb.completion_tokens,
                'cached_tokens': cached
            })


        except Exception as e:
            log.error(f" [TopSupervisor] 流式调用失败: {e}", exc_info=True)
            if self.config.debug_mode:
                raise RuntimeError(e) from e
            yield 'content', f"智能体调用失败: {e}"


    def ainvoke_wrapper(self, state: GraphState) -> GraphState:
        raise RuntimeError("顶层主Agent不作为LangGraph节点调用，该方法禁止执行")

    def _get_error_tip(self) -> str:
        raise RuntimeError("顶层主Agent不作为LangGraph节点调用，该方法禁止执行")
