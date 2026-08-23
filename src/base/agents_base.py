from src.utils.logger import log
from dataclasses import dataclass
from typing import Optional, List, Any, Dict, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from src.config.settings import settings
from src.llm.factory import llm_factory


class NodeKeyBase:
    # 统一存放所有Graph子节点标识
    RAG_AGENT = "rag_agent"
    DOC_AGENT = "doc_agent"



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
        父类仅做接口约束：返回当前Agent可用的工具实例列表
        加载逻辑、来源、筛选全部由子类自主实现
        """
        raise NotImplementedError("子类必须重写，实现自身工具实例的加载与组装")

    @property
    def default_agent(self):
        """ 懒加载拼装langchain原生Agent执行实例 """
        if self._default_agent is None:
            self._default_agent = create_agent(
                model=self.llm,
                system_prompt=self.system_prompt,
                tools=self.tools,
                middleware=[]
            )
        return self._default_agent

    @property
    def output_key(self) -> str:
        """子类必须重写：当前Agent在图中的节点名/输出标识"""
        raise NotImplementedError("每个子Agent需要定义自己的输出key，例如 rag_search、file_writer")

    def _get_error_tip(self) -> str:
        """【必须子类重写】返回当前Agent业务专属错误提示文本"""
        raise NotImplementedError("子类需要实现 _get_error_tip 方法")


    def _invoke_core(self, messages: List[Any]) -> str:
        """
        通用推理执行外壳，统一异常捕获
        :param messages: 消息列表，由子类自行组装传入
        :return: agent输出文本
        """
        log.info(f"[SubAgentInner][{self.output_key}] 开始调用Agent推理")
        try:
            agent = self.default_agent
            resp = agent.invoke({"messages": messages})
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
            log.info(f"[SubAgentInner][{self.output_key}] Agent推理完成")
            return result_content

        except Exception as e:
            log.error(f"[SubAgentInner][{self.output_key}] Agent执行异常: {e}", exc_info=True)
            if self.config.debug_mode:
                raise RuntimeError(f"Agent执行异常: {str(e)}")
            return self._get_error_tip()

    def invoke_wrapper(self, state: GraphState) -> GraphState:
        """ 子Agent任务入口：从图状态取当前任务、执行并回写 agent_outputs。 """
        log.info(f"[SubAgentInner][{self.output_key}] 进入子Agent任务执行流程")
        task_messages = state.get("task_messages") or {}
        current_task_id = state.get("current_task_id")
        current_task = (
            task_messages.get(str(current_task_id))
            if current_task_id is not None
            else None
        )
        if current_task is None:
            log.warning(f"[SubAgentInner][{self.output_key}] 未找到当前任务，回写状态")
            state["current_task_id"] = None
            state["runtime_task_inputs"] = []
            return state

        task_content = current_task.get("task_content", "")
        extra_inputs = state.get("runtime_task_inputs") or []
        if extra_inputs:
            task_content = task_content + "\n\n" + "\n\n".join(extra_inputs)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=task_content),
        ]
        try:
            result_text = self._invoke_core(messages)
        except Exception as e:
            log.error(f"[SubAgentInner][{self.output_key}] 子Agent执行异常：{e}", exc_info=True)
            state["agent_outputs"] = {
                **state.get("agent_outputs", {}),
                str(current_task_id): {
                    "target_agent": self.output_key,
                    "result": "",
                    "error": str(e),
                },
            }
            state["current_task_id"] = None
            state["runtime_task_inputs"] = []
            return state

        state["agent_outputs"] = {
            **state.get("agent_outputs", {}),
            str(current_task_id): {
                "target_agent": self.output_key,
                "result": result_text,
                "error": "",
            },
        }
        state["current_task_id"] = None
        state["runtime_task_inputs"] = []
        log.info(f"[SubAgentInner][{self.output_key}] 子Agent处理完成")
        return state
