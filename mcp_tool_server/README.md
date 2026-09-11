# MCP 工具服务（轻量化工具组件）

为上层多智能体项目提供 **文档处理 / RAG 检索 / 联网搜索** 三类工具能力的 MCP Server。
通过 **stdio 方式**被主项目拉起（主项目 `MCP_SERVER_PATH` 指向本目录的 `server.py`），也可独立运行 `python server.py`。

## 能力清单

| 分类 | 工具 | 说明 |
|---|---|---|
| 文件读写（doc） | `read_file` / `read_lines` / `write_file` / `append_file` / `replace_exact` / `replace_regex` | 文本文件操作 |
| 办公文档 | `read_docx` / `write_docx` / `read_xlsx` / `write_xlsx` | Word / Excel 读写 |
| 路径与检索 | `list_dir` / `make_dir` / `glob_files` / `search_text` | 目录、文件查找 |
| 知识库 RAG | `rag_search` / `document_storage` | 相似度检索 / 文档入库（向量化） |
| 联网搜索 | `web_search` | 直连 Bing 搜索页并解析结果（国内网络可直连，无需 API Key） |

## 架构

```text
主项目（多智能体调度）
   └─ MCP 客户端（stdio）
        └─ server.py（FastMCP）
             ├─ doc_tool      文件 / Word / Excel 工具（路径沙箱）
             ├─ rag_tool      向量化入库 + 相似度检索（Chroma / FAISS / Milvus-Lite）
             └─ web_tool      联网搜索（requests + lxml 解析 Bing）
```

- **路径沙箱**：每次任务的读写根目录由主项目通过 `set_workspace_root` 注入，隔离不同会话的工作目录。
- **向量库**：默认使用 **Milvus-Lite**（`RAG_VECTOR_STORE_NAME` 可切换 chroma / faiss），索引持久化在本目录 `mcp_src/db/` 下。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 .env（复制 .env 模板并填写）
#    QWEN_API_KEY=xxxx            # 可选：OpenAI 兼容模型
#    OLLAMA_BASE_URL=http://localhost:11434
#    RAG_DATA_PATH=D:/path/to/your/docs   # 入库的原始文档目录

# 3. 启动（stdio 模式，通常由主项目拉起）
python server.py
```

## 配置说明（.env）

| 键 | 说明 |
|---|---|
| `QWEN_API_KEY` | 可选，OpenAI 兼容模型 key |
| `OLLAMA_BASE_URL` | 本地 Ollama 地址（嵌入模型/多模态可选） |
| `RAG_DATA_PATH` | 入库检索的原始文档目录（绝对路径） |

更多可调项（向量库类型、各库路径、检索 top_k、模型名等）见 `mcp_src/config/settings.py`。

## 目录结构

```text
server.py                 # MCP Server 入口（FastMCP）
auto_register.py          # 函数 / 工具类注册器
mcp_src/
├── config/               # 配置中心（.env 加载）
├── doc_tool/             # 文件 / Word / Excel 工具
├── rag_tool/             # RAG：加载、切分、向量库、检索/入库
├── web_tool/             # 联网搜索（Bing）
├── llm_tool/             # 服务端模型调用（可选）
├── utils/                # 日志等
└── db/                   # 向量库持久化（不入库）
logs/                     # 运行日志（不入库）
workspace/                # 默认工作目录（不入库）
```

## 说明

- `.env`、`mcp_src/db/`、`logs/`、`workspace/` 均不纳入版本管理。
- 本服务为独立组件：主项目通过 MCP 接入，便于复用与隔离；向量数据与文档工具边界清晰。
