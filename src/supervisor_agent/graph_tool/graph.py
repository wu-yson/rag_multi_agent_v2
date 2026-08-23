import ast
import json

from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from src.base.agents_base import GraphState, BaseAgentConfig
from src.llm.factory import llm_factory
from src.prompts import get_prompt
from src.utils.logger import log


class MultiAgentWorkflow:
    """
    纯内层多智能体协同工作流
    职责：仅负责内部子Agent调度、循环执行、分支流转
    由顶层全局主Agent按需调用
    """

    def __init__(self) -> None:
        self.sub_agents: Dict[str, Any] = {}
        self._graph = None
        self._compiled = False
        self._llm = llm_factory.get_client(BaseAgentConfig().default_model)


    def register_sub_agent(self, node_name: str, agent_ins: Any) -> None:
        """
        子智能体注册
        :param node_name: 节点名称
        :param agent_ins: 智能体实例
        """
        if self._compiled:
            raise RuntimeError("图谱编译完成后无法注册子智能体")
        if not hasattr(agent_ins, "invoke_wrapper"):
            raise AttributeError(f"agent{node_name} 缺少 invoke_wrapper 方法")

        self.sub_agents[node_name] = agent_ins


    def _ensure_compiled(self):
        """ 懒加载编译图 """
        _ensure_sub_agents_registered()
        if not self._compiled:
            self._graph = self._build_graph()
            self._compiled = True


    def _build_graph(self) -> CompiledStateGraph:
        """ 图流程构建 """
        workflow = StateGraph(GraphState)
        workflow.add_node("supervisor", self._make_supervisor_node())
        log.info("[GraphInit]已注册主节点: supervisor ")
        for node_name, agent_ins in self.sub_agents.items():
            workflow.add_node(node_name, self._wrap_sub_agent_node(agent_ins, node_name))
            log.info(f"[GraphInit]已注册子节点: {node_name} ")
        workflow.set_entry_point("supervisor")

        def route_func(state: GraphState) -> str:
            """ 条件判断 是否继续执行 或者结束 """
            target: str = state.get("next_node", "END")
            valid_agent_names = [k.lower() for k in self.sub_agents.keys()]
            target_low = target.lower()
            log.info(f"[Route] LLM输出目标原始节点值：{target}，小写处理：{target_low}")
            if target_low in valid_agent_names:
                return target_low
            elif target.upper() == "END":
                return "END"
            log.warning(f"[Route] 未知节点名称[{target}]，兜底执行结束流程")
            return "END"

        workflow.add_conditional_edges(
            source="supervisor",
            path=route_func,
            path_map={**{k:k for k in self.sub_agents.keys()}, "END": END}
        )
        for node_name in self.sub_agents.keys():
            workflow.add_edge(node_name, "supervisor")
        compiled = workflow.compile()
        log.info("[GraphInit] 工作流构建编译完成 ")
        return compiled

    def _make_supervisor_node(self):
        """ 主节点流程：由图内 LLM 从剩余任务中选择下一步。 """

        def supervisor_core(state: GraphState) -> Dict[str, Any]:
            task_messages = state.get("task_messages") or {}
            if not task_messages:
                log.info("[Supervisor] 没有任务，结束流程")
                return {"next_node": "END"}

            selected = self._select_next_task(state, task_messages)
            if selected is None:
                return {"next_node": "END"}

            task_id, task = selected
            agent_outputs = state.get("agent_outputs", {})
            runtime_task_inputs = [
                f"[前置任务 {dep} 结果]\n{agent_outputs.get(str(dep), {}).get('result', '')}"
                for dep in (task.get("depends_on") or [])
            ]
            return {
                "next_node": task["target_agent"],
                "current_task_id": str(task_id),
                "runtime_task_inputs": runtime_task_inputs,
            }

        return supervisor_core

    @staticmethod
    def _format_task_list(tasks: list[tuple[str, dict[str, Any]]]) -> str:
        """统一格式化待执行任务文本，抽离渲染逻辑"""
        block = []
        for task_id, task in tasks:
            target_agent = task.get("target_agent")
            content = task.get("task_content")
            depends_on = task.get("depends_on", [])
            block.append(f"task {task_id} | target_agent={target_agent} | depends_on={depends_on} | 任务内容：{content}")
        return "\n".join(block)

    @staticmethod
    def _format_result_dict(agent_outputs: Dict[str, Dict[str, str]]) -> str:
        """统一格式化已完成任务结果"""
        if not agent_outputs:
            return "暂无任何已完成任务输出"
        block = []
        for task_id in sorted(agent_outputs.keys(), key=lambda x: int(x) if x.isdigit() else x):
            item = agent_outputs[task_id]
            if item.get("error"):
                block.append(f"task {task_id} 执行失败：{item['error']}")
            else:
                block.append(f"task {task_id} 执行结果：{item.get('result', '')}")
        return "\n".join(block)

    def _select_next_task(
            self,
            state: GraphState,
            task_messages: Dict[str, Dict[str, Any]],
    ) -> tuple[str, Dict[str, Any]] | None:
        """ 图内 LLM 按主Agent计划输出下一个待执行 task_id。 """
        agent_outputs = state.get("agent_outputs", {})
        if any(item.get("error") for item in agent_outputs.values()):
            log.warning("[Supervisor] 存在子Agent执行失败结果，终止调度")
            return None

        ready_tasks = [
            (str(task_id), task)
            for task_id, task in task_messages.items()
            if str(task_id) not in agent_outputs
            and all(
                str(dep) in agent_outputs
                for dep in (task.get("depends_on") or [])
            )
        ]
        if not ready_tasks:
            return None

        if len(ready_tasks) == 1:
            return ready_tasks[0]

        remaining_tasks = self._format_task_list(ready_tasks)
        agent_outputs_text = self._format_result_dict(agent_outputs)

        messages = ChatPromptTemplate.from_messages([
            ("system", get_prompt("graph_prompt")),
        ]).format_messages(
            remaining_tasks=remaining_tasks,
            agent_outputs=agent_outputs_text,
        )

        resp = self._llm.invoke(messages)
        choice = str(resp.content).strip()
        log.info(f"[Supervisor] 图内LLM下一步任务ID：{choice}")

        if choice.upper() == "END":
            return None

        selected = next(
            (
                item for item in ready_tasks
                if item[0] == choice
            ),
            None,
        )
        # LLM输出task_id不在ready任务中，直接终止流程
        if selected is None:
            log.warning(f"[Supervisor] LLM输出[{choice}] 不在ready任务ID中，终止调度")
            return None
        return selected

    def _wrap_sub_agent_node(self, agent_ins: Any, node_name: str):
        """ 子智能体节点流程 """
        def sub_node(state: GraphState) -> GraphState:
            log.info(f"[SubAgent] 开始执行子节点：{node_name}")
            return agent_ins.invoke_wrapper(state)
        return sub_node

    def invoke(self, **kwargs) -> GraphState:
        """
        图对外调用唯一接口
        :param kwargs: 其他自定义状态字段
        :return: GraphState 完整运行后的状态
        """
        self._ensure_compiled()

        init_state = {
            "next_node": "",
            "task_messages": {},
            "current_task_id": None,
            "agent_outputs": {},
            "runtime_task_inputs": [],
            **kwargs
        }
        return self._graph.invoke(init_state)

    def build(self) -> None:
        """ 获取流程图 """
        return self._ensure_compiled()

# 创建图 唯一 实例对象
agents_graph = MultiAgentWorkflow()


def _ensure_sub_agents_registered():
    """ 懒加载子 Agent，避免包初始化循环导入。 """
    if "rag_agent" not in agents_graph.sub_agents:
        from src.rag_agent.rag_agent import rag_agent  # noqa: F401
    if "doc_agent" not in agents_graph.sub_agents:
        from src.doc_agent.document_agent import doc_agent  # noqa: F401


class GraphInvokeTool(BaseTool):
    """ 图工具接口 """
    name: str = "graph_invoke"
    description: str = (
        "调用内部多子Agent协同工作流；调用时传入包含 task_messages 的 JSON 字符串。"
        "文档入库、知识库检索必须用 target_agent=rag_agent；本地文件读写、生成Word/Excel/Txt必须用 target_agent=doc_agent。"
    )

    def _run(self, workflow_json: str) -> str:
        try:
            if isinstance(workflow_json, str):
                try:
                    plan = json.loads(workflow_json)
                except json.JSONDecodeError:
                    plan = ast.literal_eval(workflow_json)
            else:
                plan = workflow_json
            if (
                not isinstance(plan, dict)
                or not isinstance(plan.get("task_messages"), dict)
                or not plan["task_messages"]
            ):
                raise ValueError("缺少 task_messages 字典")
            normalized_tasks = {}
            for task_id, task in plan["task_messages"].items():
                if not isinstance(task, dict) or not task.get("target_agent"):
                    raise ValueError(f"任务 {task_id} 缺少 target_agent")
                if not isinstance(task.get("depends_on", []), list):
                    raise ValueError(f"任务 {task_id} 的 depends_on 格式错误")
                normalized_tasks[str(task_id)] = task
        except (ValueError, TypeError, SyntaxError) as e:
            log.warning(f"[GraphTool] task_messages 格式错误：{e}")
            return json.dumps({
                "error": f"FORMAT_ERROR: {e}",
                "hint": "请重新输出 task_messages 字典，key 为任务ID，字段为 target_agent、task_content、depends_on",
            }, ensure_ascii=False)

        try:
            log.info(f"[GraphTool] 主Agent拆解任务：{json.dumps(plan, ensure_ascii=False)}")
            graph_state = agents_graph.invoke(task_messages=normalized_tasks)

            summary_parts = [
                "========== 多子Agent工作流执行结果汇总 ==========",
                "各步骤输出：",
            ]

            agent_outputs = graph_state.get("agent_outputs", {})
            for task_id in sorted(agent_outputs, key=lambda x: int(x) if x.isdigit() else x):
                item = agent_outputs[task_id]
                output = item.get("error") or item.get("result") or "无输出"
                summary_parts.append(f"task {task_id} 输出：{output}")
            return "\n".join(summary_parts)
        except Exception as e:
            log.error(f"[GraphTool] 多Agent工作流执行异常，错误信息：{str(e)}", exc_info=True)
            raise

# 注册工具实例
graph_invoke = GraphInvokeTool()
