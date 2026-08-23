from pathlib import Path

from langchain_core.tools import tool
from src.rag_agent.rag.factory import rag_factory


# 加载沙箱根目录，统一解析为绝对路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SANDBOX_ROOT = (BASE_DIR / "src" / "data").resolve()

@tool(description = "向量库相似度检索，输入用户问题，召回匹配的参考文档片段")
def rag_search(user_input: str) -> list[dict]:
    """ 仅执行向量库检索，返回拼接后的参考文档文本，不直接生成回答 """
    try:
        return rag_factory.query(query=user_input)
    except Exception as e:
        return [{"text": f"向量库检索异常：{str(e)}"}]

@tool(description="向量库文档存储，输入目录路径，将目录内文档入库向量库；仅允许沙箱目录内存在的目录路径")
def document_storage(dir_path: str) -> str:
    """
    文档批量入库
    :param dir_path: 待入库文档所在目录（必须位于沙箱目录内）
    :return: 执行结果文本
    """
    try:
        p = Path(dir_path)
        if not p.is_absolute():
            target = (BASE_DIR / p).resolve()
        else:
            target = p.resolve()
        # 校验1：路径是否存在
        if not target.exists():
            return f"操作失败：路径【{dir_path}】不存在。"
        # 校验2：路径是否在沙箱范围内，防止路径穿越
        if not target.is_relative_to(SANDBOX_ROOT):
            return f"安全限制：【{dir_path}】不在允许的沙箱目录范围内，禁止执行入库操作。"
        # 校验3：当前接口仅支持目录，不支持单个文件
        if not target.is_dir():
            return f"操作失败：【{dir_path}】不是有效目录，当前仅支持传入目录路径批量入库。"

        rag_factory.index_directory(dir_path=str(target))
        return "文档入库成功"
    except Exception as e:
        return f"文档入库失败，错误信息：{str(e)}"

