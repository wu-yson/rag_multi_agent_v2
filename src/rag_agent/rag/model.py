
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from src.utils.logger import log


# ======================================================================
# 异常
# ======================================================================


class StoreNotFoundException(Exception):
    """请求的向量存储后端未注册。"""

    def __init__(self, name: str, supported: list[str]) ->  None:
        self.name = name
        self.supported = supported
        # 调用父类的构造函数, 传入异常信息,抛出去, 细分异常精准找到错误
        super().__init__(f"请求的向量存储后端: {name}, 未注册. 已注册的向量存储: {supported}")



class LoaderNotFoundException(Exception):
    """请求的文件类型没有对应的加载器。"""

    # extension为传进来的错误文件后缀, supported为支持的后缀列表
    def __init__(self, extension: str, supported: list[str]):
        self.extension = extension
        self.supported = supported
        # 调用父类的构造函数, 传入异常信息,抛出去, 细分异常精准找到错误
        super().__init__(f"文件扩展名: {extension}, 无对应加载器. 已注册的加载器: {supported}")




# ======================================================================
# 配置数据类
# ======================================================================

@dataclass
class ChromaVectorConfig:
    """Chroma向量存储配置。"""
    persist_path: Optional[str]
    embedding_model: Optional[str]
    search_top_k: int


@dataclass
class FaissVectorConfig:
    """Faiss向量存储配置。"""
    persist_path: str
    embedding_model: str
    search_top_k: int
    meta_path: str


@dataclass
class SplitConfig:
    """分割器配置。"""
    chunk_size: int = 500  # 文本分块单块最大字符数
    chunk_overlap: int = 50  # 文本分块单块重叠字符数


@dataclass
class RAGConfig:
    """rag配置。"""
    vector_store_name :str
    chat_model: Optional[str]



# ======================================================================
# 文档加载器策略接口
# ======================================================================


class DocumentLoader(ABC):
    """文档加载器策略接口。"""

    SUPPORTED_EXTENSIONS: list[str] = []

    @abstractmethod
    def load(self, file_path: str) -> list[dict]:
        """加载文件，返回 [{"content": ..., "metadata": {"source": ...}}]。"""




# ======================================================================
# 向量存储策略接口
# ======================================================================

class VectorStoreProvider(ABC):
    """向量存储供应商接口。"""
    @abstractmethod
    def add_texts(self, texts: list[str], metadatas: Optional[list[dict]] = None, ids: Optional[List[str]] = None) -> None:
        """添加文本到向量存储。"""

    @abstractmethod
    def similarity_search(self, query: str, k: int = None) -> list[dict]:
        """相似度检索，返回 [{"content": ..., "metadata": ...}]。"""


    def save(self, path: Optional[str] = None) -> None:
        """持久化到磁盘。"""

    @abstractmethod
    def load(self, path: str) -> None:
        """从磁盘加载。"""

    # 基类新增抽象方法
    def get_all_stored_file_md5(self) -> set[str]:
        """获取库中所有已持久化的完整文件md5集合，用于前置整文件校验"""
        raise NotImplementedError("Chroma/FAISS子类单独实现")

    # ===================== 新增通用工具方法，子类直接调用，不用重复写 =====================
    def _clean_text_batch(self, texts: list[str], metadatas: Optional[list[dict]]):
        """
        公共：文本清洗、剔除空字符串
        return (clean_texts, clean_metas, file_record_map)
        file_record_map: {file_md5: file_path}
        """
        if not texts:
            log.warning(f"传入文本共{len(texts)}条，无数据，放弃入库，请检查上游分片输出")
            return [], [], {}
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]

        texts_list = []
        metadatas_list = []
        record_map: Dict[str, str] = {}

        for t, m in zip(texts, metadatas):
            clean_text = t.strip()
            if clean_text:
                texts_list.append(clean_text)
                metadatas_list.append(m)
                f_md5 = m.get("file_md5")
                f_path = m.get("file_path")
                if f_md5 and f_path:
                    record_map[f_md5] = f_path
        return texts_list, metadatas_list, record_map

    def _wrap_search_result(self, docs) -> list[dict[str, Any]]:
        """公共：统一封装检索结果格式"""
        res = []
        for doc in docs:
            if not doc.metadata.get("_tmp"):
                res.append({"content": doc.page_content, "metadata": doc.metadata})
        return res


# ======================================================================
# 文本分割器策略接口
# ======================================================================


class TextSplitter(ABC):
    """文本分割器策略接口。"""
    @abstractmethod
    def split(self, documents: list[dict]) -> list[dict]:
        """分割文档列表，返回 [{"content": ..., "metadata": ...}]。"""


