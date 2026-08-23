# 轻量化多智能体工程调度系统
基于单智能体底座工程迭代升级，采用 Supervisor 主管调度架构，依托 LangGraph 实现任务自动拆解、子 Agent 编排、任务依赖管控、会话记忆隔离；
完善异常处理、接口封装、边界流程测试，补齐整套工程化配套能力，支持复杂多步骤业务任务闭环执行。
启用 LangGraph 的多智能体应用骨架，主 Agent 拆解任务，图内 LLM 按任务 ID 调度，子 Agent 读取状态并执行工具，完成后回写结果。

## 设计原则

- 主 Agent 只负责任务拆解，不直接执行业务。
- 图内 LLM 只负责按计划选择下一步 `task_id`。
- 子 Agent 只读取当前任务和临时前置上下文，执行后回写结果。
- 核心调度链路保持稳定，工程化能力在外层叠加。
- 业务失败或空结果不自动重试；任务格式错误允许主 Agent 修正后重试。


## 核心功能

- 多智能体调度
  - 主 Agent 输出 `task_messages`
  - 图内 LLM 输出下一步 `task_id`
  - supervisor 将 `task_id` 翻译为 `target_agent` 路由
  - 子 Agent 通过 `current_task_id` 取任务
  - `depends_on` 控制依赖顺序，`runtime_task_inputs` 传递前置结果
- RAG
  - txt 文档加载与分割
  - Chroma / FAISS 向量库
  - 文档入库、相似度检索
- 文档 Agent
  - txt / docx / xlsx 读写
  - 工具注册与 skill 说明
- 工程化能力
  - FastAPI 接口
  - `.env` 配置管理
  - SQLite / MySQL 会话记忆
  - 统一日志
  - 任务格式错误返回 `FORMAT_ERROR`，由主 Agent 修正重试
- 安全
  - 提示词注入检测
  - 文件路径沙箱限制

## 核心调度流程

1. 主 Agent 接收用户请求，判断是否需要子 Agent 协作。
2. 需要协作时输出 `task_messages` 并调用 `graph_invoke`。
3. `graph_invoke` 校验任务格式；格式错误返回 `FORMAT_ERROR`，由主 Agent 修正后重试。
4. 图内 LLM 根据 `task_messages` 和 `agent_outputs` 输出下一个 `task_id`。
5. supervisor 将 `task_id` 翻译为 `target_agent`，并设置 `current_task_id`。
6. 如果任务有 `depends_on`，从 `agent_outputs` 取前置结果写入 `runtime_task_inputs`。
7. `runtime_task_inputs` 是当前轮临时字段，每次选择新任务时重新生成，子 Agent 执行完成后清空。
8. 子 Agent 通过 `current_task_id` 读取任务，执行工具后写回 `agent_outputs`。
9. 图回到 supervisor 继续选择下一个任务，直到输出 `END`。
10. 主 Agent 汇总子 Agent 结果，生成最终回复。


## 技术栈

Python / LangChain / LangGraph / FastAPI / SQLModel / Chroma / FAISS / Pydantic-Settings

## 目录结构

```text
src/
├── api/                 # FastAPI 接口
├── base/                # GraphState、Agent 基类
├── config/              # 配置中心
├── doc_agent/           # 文档 Agent 与文件工具
├── llm/                 # LLM 工厂与供应商
├── memory/              # 会话记忆
├── prompts/             # 主 Agent、子 Agent、图调度提示词
├── rag_agent/           # RAG_Agent RAG工厂、加载、分割、向量库
├── supervisor_agent/    # 主 Agent 与 graph_tool
└── utils/               # 日志等通用模块
src/scripts/             # 手动测试
```

## 配置说明

主要配置位于 `.env`，由 `src/config/settings.py` 统一加载：

```env
QWEN_API_KEY=your_api_key
QWEN_BASE_URL=https://your-qwen-endpoint
QWEN_TIMEOUT=60
QWEN_TEMPERATURE=0.7
OLLAMA_BASE_URL=http://localhost:11434
```

## 快速开始

```bash
pip install -r requirements.txt
```

在 `.env` 中配置：

```env
QWEN_API_KEY=your_api_key
```

启动服务：

```bash
python main.py
```

接口：

```text
POST /api/chat
POST /api/memory/clear
```

聊天请求示例：

```json
{
    "query": "总结知识库中机器人使用手册内容并写入 Word",
    "session_id": "test_session_001"
}
```

清空记忆请求示例：

```json
{
    "session_id": "test_session_001"
}
```

测试：

```bash
python src/scripts/run_llm_simple.py
python src/scripts/run_full_flow.py
```





