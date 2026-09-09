
import asyncio
import json
from typing import Any, Dict
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from src.base.agents_base import GraphState
from src.utils.logger import log


SUB_AGENT_TIMEOUT = 300


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
            deps = task_messages[tid].get("depends_on") or []
            return all(str(d) in outputs for d in deps)

        async def run_one(tid):
            task = task_messages[tid]
            sub_state = dict(state)
            sub_state["current_task_id"] = str(tid)
            sub_state["runtime_task_inputs"] = [
                f"[前置任务 {dep} 结果]\n{outputs.get(str(dep), {}).get('result', '')}"
                for dep in (task.get("depends_on") or [])
            ]
            try:
                result_state = await asyncio.wait_for(
                    self.sub_agents[task["target_agent"]].ainvoke_wrapper(sub_state),
                    timeout=SUB_AGENT_TIMEOUT,
                )
                outputs[str(tid)] = result_state["agent_outputs"][str(tid)]
            except asyncio.TimeoutError:
                log.error(f"[Graph] 子agent {task['target_agent']} 任务{tid} 超时")
                outputs[str(tid)] = {
                    "target_agent": task['target_agent'],
                    "result": "",
                    "error": f"子Agent执行超时（>{SUB_AGENT_TIMEOUT}秒）"
                }
        running = set()
        while len(started) < len(task_messages):
            for tid in task_messages:
                if tid not in started and deps_satisfied(tid):
                    started.add(tid)
                    running.add(asyncio.create_task(run_one(tid)))
            if not running:
                raise RuntimeError(f"任务依赖无法满足: {[t for t in task_messages if t not in started]}")
            await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)  # ③ 等一个完成
            running = {t for t in running if not t.done()}  # 清掉已完成的，留还在跑的
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
    if "rag_search" not in agents_graph.sub_agents:
        from src.rag_agent.rag_agent import rag_search_node  # noqa: F401
    if "rag_storage" not in agents_graph.sub_agents:
        from src.rag_agent.rag_agent import rag_storage_node  # noqa: F401  # noqa: F401
    if "doc_agent" not in agents_graph.sub_agents:
        from src.doc_agent.document_agent import doc_agent  # noqa: F401
    if "web_search" not in agents_graph.sub_agents:
        from src.web_agent.web_agent import web_search_node  # noqa: F401


class TaskItem(BaseModel):
    target_agent: str = Field(description="子节点名：rag_search / rag_storage / doc_agent / web_search")
    task_content: str = Field(description="该任务的具体内容")
    depends_on: list[str] = Field(default_factory=list, description="依赖的任务ID列表")

class WorkflowPlan(BaseModel):
    task_messages: dict[str, TaskItem] = Field(description="任务字典，key 为任务ID字符串")


class GraphInvokeTool(BaseTool):
    """ 图工具接口 """
    name: str = "graph_invoke"
    args_schema: type[WorkflowPlan] = WorkflowPlan
    description: str = (
        "调用内部多子Agent协同工作流；传入 task_messages 任务字典。"
        "知识库检索必须用 target_agent=rag_search；文档入库必须用 target_agent=rag_storage；"
        "本地文件读写、生成Word/Excel/Txt必须用 target_agent=doc_agent。"
        "联网搜索、网页资料查询、最新信息检索必须用 target_agent=web_search。"
    )

    def _build_summary(self, graph_state, normalized_tasks) -> str:
        """按依赖区分中间/最终任务，生成结果汇总"""
        summary_parts = [
            "========== 多子Agent工作流执行结果汇总 ==========",
            "各步骤输出：",
        ]
        agent_outputs = graph_state.get('agent_outputs', {})
        depended = set()
        for t in normalized_tasks.values():
            for dep in (t.get('depends_on') or []):
                depended.add(str(dep))
        for task_id in sorted(agent_outputs, key=lambda x: int(x) if x.isdigit() else x):
            item = agent_outputs[task_id]
            full = item.get("error") or item.get("result") or "无输出"
            log.info(f"[GraphTool] task {task_id} 完整结果: {full}")
            if str(task_id) in depended:
                status = "失败" if item.get("error") else "成功"
                summary_parts.append(f"task {task_id}：{status}")
            else:
                summary_parts.append(f"task {task_id} 输出：{full}")
        summary_text = "\n".join(summary_parts)
        log.info(f"[GraphTool] 返回给主Agent的汇总:\n{summary_text}")
        return summary_text

    async def _arun(self, task_messages: dict) -> str:
        normalized_tasks = {
            str(k): (v.model_dump() if isinstance(v, TaskItem) else v)
            for k, v in task_messages.items()
        }
        try:
            log.info(f"[GraphTool] 主Agent拆解任务：{json.dumps(normalized_tasks, ensure_ascii=False)}")
            graph_state = await agents_graph.ainvoke(task_messages=normalized_tasks)
            return self._build_summary(graph_state, normalized_tasks)
        except Exception as e:
            log.error(f"[GraphTool] 多Agent工作流执行异常，错误信息：{str(e)}", exc_info=True)
            raise

    def _run(self, task_messages: dict) -> str:
        raise NotImplementedError("graph_invoke 仅支持异步调用")

graph_invoke = GraphInvokeTool()
