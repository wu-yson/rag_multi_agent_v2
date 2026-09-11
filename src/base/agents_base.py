

from src.llm.resilience import build_agent_middleware
from src.utils.logger import log
from dataclasses import dataclass
from typing import Optional, List, Any, Dict, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from src.config.settings import settings
from src.llm.factory import llm_factory


class NodeKeyBase:
    # 统一存放所有Graph子节点标识
    RAG_SEARCH = "rag_search"  # 知识库检索节点(普通函数节点)
    RAG_STORAGE = "rag_storage"  # 文档入库节点(普通函数节点)
    DOC_AGENT = "doc_agent"
    WEB_SEARCH = "web_search"  # 联网搜索节点(普通函数节点)



class GraphState(TypedDict):
    """
    多智能体流程图全局共享状态
    Fields:
        next_node: 下一跳执行节点名称
        task_messages: 待执行任务字典，key 为任务ID
        current_task_id: 当前运行任务id
        agent_outputs: {任务id: {target_agent, result, error}}
        runtime_task_inputs: 当前轮联动时临时注入的前置结果
    """
    next_node: str
    task_messages: Dict[str, Dict[str, Any]]
    current_task_id: Optional[str]
    agent_outputs: Dict[str, Dict[str, str]]
    runtime_task_inputs: List[str]


@dataclass
class BaseAgentConfig:
    """智能体配置基类"""
    default_model: str = settings.agent_default_model
    debug_mode: bool = settings.agent_debug_mode



class BaseAgentTemplate:
    """智能体模板基类"""
    def __init__(self, config: BaseAgentConfig):
        self.config = config
        self.model = self.config.default_model

        self._llm: Optional[Any] = None
        self._system_prompt: Optional[str] = None
        self._default_agent: Optional[Any] = None
        self._tools: Optional[list[Any]] = None

    @property
    def llm(self):
        """ 懒加载大模型客户端 """
        if self._llm is None:
            self._llm = llm_factory.get_client(self.model)
        return self._llm

    @property
    def system_prompt(self) -> str:
        """
        【子类必须重写】
        每个Agent拥有独立提示词，基类不绑定任意提示词工厂函数
        子类内部自行实现懒加载，调用自身对应的提示词获取方法
        """
        raise NotImplementedError("当前Agent需要重写system_prompt，加载专属提示词模板")

    @property
    def tools(self) -> list[Any]:
        """
        加载工具
        """
        raise NotImplementedError("子类必须重写，实现自身工具实例的加载与组装")

    async def _load_tools(self):
        """加载mcp工具：默认用本地 tools 属性；子类可覆写为 MCP 加载"""
        return self.tools

    async def get_agent(self):
        """异步获取 agent：先加载工具（可能来自 MCP），再创建"""
        if self._default_agent is None:
            tools = await self._load_tools()
            self._default_agent = create_agent(
                model=self.llm,
                system_prompt=self.system_prompt,
                tools=tools,
                middleware=build_agent_middleware(),
            )
        return self._default_agent



    @property
    def output_key(self) -> str:
        """子类必须重写：当前Agent在图中的节点名/输出标识"""
        raise NotImplementedError("每个子Agent需要定义自己的输出key，例如 rag_search、file_writer")

    def _get_error_tip(self) -> str:
        """【必须子类重写】返回当前Agent业务专属错误提示文本"""
        raise NotImplementedError("子类需要实现 _get_error_tip 方法")


    async def _invoke_core(self, messages: List[Any]) -> str:
        """
        通用推理执行外壳，统一异常捕获
        :param messages: 消息列表，由子类自行组装传入
        :return: agent输出文本
        """
        log.info(f"[SubAgentInner][{self.output_key}] 开始调用Agent推理")
        try:
            agent = await self.get_agent()
            resp = await agent.ainvoke({"messages": messages})

            msg_list = resp["messages"]
            last_msg = msg_list[-1]

            # 新增：识别子Agent内部工具调用
            tool_call_names = sorted({
                item["name"]
                for msg in msg_list
                for item in getattr(msg, "tool_calls", [])
            })
            if tool_call_names:
                log.info(f"[SubAgentInner][{self.output_key}] LLM决策：调用工具，工具列表：{tool_call_names}")
            else:
                log.info(f"[SubAgentInner][{self.output_key}] LLM决策：无工具调用，直接输出回复")

            result_content = last_msg.content
            if not result_content:  # 既没调工具、又没输出 = 异常
                log.warning(f"[SubAgentInner][{self.output_key}] 模型无有效输出，任务将按失败处理")
                result_content = ""
            log.info(f"[SubAgentInner][{self.output_key}] Agent推理完成")
            return result_content

        except Exception as e:
            log.error(f"[SubAgentInner][{self.output_key}] Agent执行异常: {e}", exc_info=True)
            if self.config.debug_mode:
                raise RuntimeError(f"Agent执行异常: {str(e)}")
            return self._get_error_tip()


    async def ainvoke_wrapper(self, state: GraphState) -> GraphState:
        """ 子Agent任务入口：从图状态取当前任务、执行并回写 agent_outputs。 """
        log.info(f"[SubAgentInner][{self.output_key}] 进入子Agent任务执行流程")
        return await run_node(
            state,
            self.output_key,
            self._run_agent_task,
            empty_error="子Agent未生成有效结果（未调用工具也未输出内容）",
        )

    async def _run_agent_task(self, content: str) -> str:
        """ Agent 型节点的实际执行：拼 system_prompt + 任务内容，走 LLM 推理。 """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=content),
        ]
        result_text = await self._invoke_core(messages)
        log.info(f"[SubAgentInner][{self.output_key}] 子Agent处理完成")
        return result_text


# ===== 图节点公共函数：Agent 子节点与普通函数节点共用，统一回写状态 =====


def _resolve_node_task(state: GraphState):
    """ 取出当前要执行的任务，返回 (任务ID, 拼接好前置结果的完整内容)。

    :param state: 图节点共享状态
    :return: (task_id, content)；若状态里没有当前任务，task_id 为 None
    """
    task_messages = state.get("task_messages") or {}
    current_task_id = state.get("current_task_id")
    task_id = str(current_task_id) if current_task_id is not None else None
    task = task_messages.get(task_id) if task_id is not None else None
    if task is None:
        return None, ""

    content = task.get("task_content", "")
    extra_inputs = state.get("runtime_task_inputs") or []
    if extra_inputs:
        content = content + "\n\n" + "\n\n".join(extra_inputs)
    return task_id, content


def _save_node_output(
    state: GraphState,
    task_id: str,
    output_key: str,
    result: str,
    error: str = "",
) -> GraphState:
    """ 节点处理完成后，统一把结果写回 agent_outputs 并复位本次任务字段。

    :param state: 图节点共享状态（原地修改并返回）
    :param task_id: 当前任务ID
    :param output_key: 节点在图中的路由名（target_agent）
    :param result: 节点处理结果文本
    :param error: 失败原因，为空表示成功
    """
    state["agent_outputs"] = {
        **state.get("agent_outputs", {}),
        str(task_id): {
            "target_agent": output_key,
            "result": result,
            "error": error,
        },
    }
    state["current_task_id"] = None
    state["runtime_task_inputs"] = []
    return state


async def run_node(
    state: GraphState,
    output_key: str,
    runner: Any,
    empty_error: str = "节点执行未返回有效结果",
) -> GraphState:
    """ 统一节点执行：取任务 → runner(content) 执行 → 结果/异常兜底写回。

    供 Agent 子节点与普通函数节点共用：正常结果写 result；
    执行抛异常或返回空时，统一把提示写进 error，保证各节点写回的消息字段格式一致。
    """
    task_id, content = _resolve_node_task(state)
    if task_id is None:
        log.warning(f"[GraphNode][{output_key}] 未找到当前任务，回写状态")
        state["current_task_id"] = None
        state["runtime_task_inputs"] = []
        return state

    try:
        text = str(await runner(content) or "").strip()
        error = "" if text else empty_error
    except Exception as e:
        log.error(f"[GraphNode][{output_key}] 节点执行异常：{e}", exc_info=True)
        text, error = "", str(e)
    return _save_node_output(state, task_id, output_key, text, error)
