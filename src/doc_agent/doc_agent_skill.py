from langchain_core.tools import tool

from src.doc_agent import doc_tools  # noqa: F401  导入以触发文件工具注册
from src.doc_agent.doc_tools.registry import tool_registry


DOC_AGENT_SKILL_TEXT = """
【doc_agent 文件工具使用规则】

1. txt 文本文件
   - read_file：读取整个文本文件内容
   - read_lines：按行区间读取，适合只取部分内容
   - write_file：覆盖写入或新建 txt
   - append_file：追加内容到 txt
   - replace_exact：精确字符串替换
   - replace_regex：正则表达式替换

2. Word / Excel
   - read_docx：读取 docx 正文文本
   - write_docx：新建或覆盖 docx
   - read_xlsx：读取 xlsx 为表格文本
   - write_xlsx：按二维列表写入 xlsx

3. 目录 / 搜索
   - list_dir：列出目录内容
   - make_dir：创建目录
   - glob_files：按文件名模式查找文件
   - search_text：在多个文件中搜索文本内容

【决策建议】
- 需要读取 txt 全文 -> read_file
- 需要读取指定行 -> read_lines
- 需要生成 docx -> write_docx
- 需要生成 xlsx 表格 -> write_xlsx
- 需要查找文件名 -> glob_files
- 需要搜索文本内容 -> search_text
- 需要查看目录 -> list_dir
- 需要创建目录 -> make_dir

【执行约束】
- 多步骤文件任务必须先读取或搜索得到真实内容，再执行写入。
- 不允许把没有读取到的内容编造后写入文件。
- 不确定当前任务该用哪个工具时，先调用本工具获取说明。
"""


@tool
def get_doc_agent_skill() -> str:
    """获取 doc_agent 本地文件工具的使用说明；当不确定使用哪个文件工具、工具参数含义或需要文件操作流程建议时调用。"""
    tool_lines = [
        f"- {tool_obj.name}: {tool_obj.description}"
        for tool_obj in tool_registry.get_all()
    ]
    return DOC_AGENT_SKILL_TEXT + "\n\n【当前可用工具】\n" + "\n".join(tool_lines)
