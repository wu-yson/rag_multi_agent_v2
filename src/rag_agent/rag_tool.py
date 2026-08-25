from pathlib import Path

from langchain_core.tools import tool

from src.config.settings import settings
from src.rag_agent.rag.factory import rag_factory


def _get_data_root() -> Path:
    """从配置读取默认数据目录，未配置时给出明确提示。"""
    if not settings.rag_data_path:
        raise RuntimeError("未配置数据目录，请在 .env 中设置 RAG_DATA_PATH")
    return Path(settings.rag_data_path).resolve()


@tool(description="向量库相似度检索，输入用户问题，召回匹配的参考文档片段")
def rag_search(user_input: str) -> list[dict]:
    """仅执行向量库检索，返回拼接后的参考文档文本，不直接生成回答"""
    try:
        return rag_factory.query(query=user_input)
    except Exception as e:
        return [{"text": f"向量库检索异常：{str(e)}"}]


@tool(description="向量库文档存储：用户提供完整路径时入库该路径，未提供路径时使用默认数据目录；路径合法性由系统自动校验，无需向用户确认")
def document_storage(dir_path: str = "") -> str:
    """
    文档批量入库
    :param dir_path: 可选，待入库目录；为空时使用默认数据目录
    :return: 执行结果文本
    """
    try:
        data_root = _get_data_root()
        raw = dir_path.strip()
        if not raw:
            target = data_root
        else:
            p = Path(raw)
            if not p.is_absolute():
                return "操作失败：请传入完整绝对路径。"
            target = p.resolve()
            if not target.is_relative_to(data_root):
                return f"安全限制：「{raw}」不在允许的数据目录范围内，禁止执行入库操作。"

        if not target.exists():
            return f"操作失败：路径「{raw or data_root}」不存在。"
        if not target.is_dir():
            return f"操作失败：「{raw or data_root}」不是有效目录，当前仅支持传入目录路径批量入库。"

        rag_factory.index_directory(dir_path=str(target))
        return "文档入库成功"
    except Exception as e:
        return f"文档入库失败，错误信息：{str(e)}"
