import ast
import asyncio
import json

from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from watchfiles import awatch

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
        if not hasattr(agent_ins, "ainvoke_wrapper"):
            raise AttributeError(f"agent{node_name} 缺少 ainvoke_wrapper 方法")

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
        workflow.set_entry_point("supervisor")
        workflow.add_edge("supervisor", END)  # 调度都在 supervisor 内部完成
        compiled = workflow.compile()
        log.info("[GraphInit] 依赖驱动调度图构建完成")
        return compiled

    async def _execute_all(self, state):
        """依赖驱动调度：谁的前置完成谁立刻跑"""
        task_messages = state.get("task_messages", {})
        outputs = state.get("agent_outputs", {})
        started = set()

        def deps_satisfied(tid):
            task = task_messages[tid]
            return all(str(d) in outputs for d in (task.get("depends_on") or []))

        async def run_one(tid):
            task = task_messages[tid]
            sub_state = dict(state)
            sub_state["current_task_id"] = str(tid)
            sub_state["runtime_task_inputs"] = [
                f"[前置任务 {dep} 结果]\n{outputs.get(str(dep), {}).get('result', '')}"
                for dep in (task.get("depends_on") or [])
            ]
            result_state = await self.sub_agents[task["target_agent"]].ainvoke_wrapper(sub_state)
            outputs[str(tid)] = result_state["agent_outputs"][str(tid)]

        running = set()
        while True:
            for tid in task_messages:
                if tid not in started and deps_satisfied(tid):
                    started.add(tid)
                    running.add(asyncio.create_task(run_one(tid)))
            if not running:
                break
            done, running = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
        return outputs



    def _make_supervisor_node(self):
        """ 主节点流程 """
        async def supervisor_core(state: GraphState) -> Dict[str, Any]:
            task_messages = state.get("task_messages") or {}
            if not task_messages:
                log.info("[Supervisor] 没有任务，结束流程")
                return {"next_node": "END"}
            outputs = await self._execute_all(state)
            return {"agent_outputs": outputs, "next_node": "END"}
        return supervisor_core





    async def ainvoke(self, **kwargs) -> GraphState:
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
        return await self._graph.ainvoke(init_state)

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

    async def _arun(self, workflow_json: str) -> str:
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
            graph_state = await agents_graph.ainvoke(task_messages=normalized_tasks)

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

    def _run(self, workflow_json: str) -> str:
        raise NotImplementedError("graph_invoke 仅支持异步调用")

# 注册工具实例
graph_invoke = GraphInvokeTool()
