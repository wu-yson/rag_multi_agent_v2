from pathlib import Path
from typing import Optional, List, Any
from langchain_chroma import Chroma
import logging
from src.llm.factory import llm_factory
from src.rag_agent.rag.factory import register_vectorstore
from src.rag_agent.rag.model import VectorStoreProvider, ChromaVectorConfig
from src.config.settings import settings
logger = logging.getLogger(__name__)



def build_chroma_default_config() -> ChromaVectorConfig:
    return ChromaVectorConfig(
        persist_path=settings.chroma_persist_path,
        embedding_model=settings.chroma_embedding_model,
        search_top_k = settings.chroma_search_top_k,

    )


@register_vectorstore("chroma",lambda: build_chroma_default_config())
class ChromaStore(VectorStoreProvider):
    """
    Chroma 向量存储，对齐顶层抽象接口规范
    """
    def __init__(self, config: ChromaVectorConfig) -> None:
        self._config = config
        self._embedding = self._config.embedding_model
        self.k = self._config.search_top_k

        if self._config.persist_path is None or not self._config.persist_path.strip():
            raise ValueError("向量库持久化路径不能为空")

        self.persist_path = Path(self._config.persist_path)
        self._store: Optional[Chroma] = None
        if self.persist_path and self.persist_path.exists():
            try:
                self._store = Chroma(
                    persist_directory=str(self.persist_path),
                    embedding_function=llm_factory.get_client(self._embedding),
                )
                logging.info(f"从{self.persist_path} 载 Chroma 向量库")
                return
            except Exception as e:
                logging.error(f"加载失败，将创建全新Chroma库: {e}")
        self._store = Chroma(

            persist_directory = str(self.persist_path),
            embedding_function=llm_factory.get_client(self._embedding),
        )
        logger.info(f"创建新 Chroma 向量库: {self.persist_path}")

    def get_all_stored_file_md5(self) -> set[str]:
        """
        获取向量库文档md5元数据
        :return: md5
        """
        data = self.store.get(include=["metadatas"])
        metadatas = data["metadatas"]
        md5_set = set()
        for meta in metadatas:
            if meta and "file_md5" in meta:
                md5_set.add(meta["file_md5"])
        return md5_set

    def add_texts(self, texts: list[str], metadatas: Optional[list[dict]] = None,
                  ids: Optional[List[str]] = None) -> None:
        """
        添加文本
        :param texts: 文本列表
        :param metadatas: 元数据列表
        :param ids: 文档ID 可不写
        :return:
        """
        texts_list, metadatas_list, _ = self._clean_text_batch(texts, metadatas)
        if not texts_list:
            logger.warning("清洗后无有效内容，放弃入库")
            return

        self._store.add_texts(
            texts=texts_list,
            metadatas=metadatas_list,
            ids=ids
        )
        logger.info(f"向量库添加 {len(texts_list)} 条数据")

    def similarity_search(self, query: str, k: int = None) -> list[dict[str, Any]]:
        use_k = k if k is not None else self.k
        docs = self._store.similarity_search(query, k=use_k)
        return self._wrap_search_result(docs)

    def save(self, path: Optional[str] = None) -> None:
        """
        保存索引到磁盘
        Chroma向量库开启持久化目录之后会自动落盘，无需手动执行保存操作。
        """
        pass

    def delete_texts(self, ids: List[str]) -> int:
        """根据ID批量删除向量"""
        if  not ids:
            return 0
        self._store.delete(ids=ids)
        logger.info(f"根据ID删除 {len(ids)} 条数据")
        return len(ids)

    @property
    def store(self) -> Chroma:
        """直接暴露原生 Chroma 对象"""
        return self._store

    def load(self, path: str) -> None:
        """
        切换加载指定路径的存量向量库，运行中动态切换数据源(分别做有2个已存在的向量库数据)
        执行完毕需要使用别的库, 需要手动切库
        """
        new_path = Path(path)
        if not new_path.exists():
            logger.error(f"向量库路径不存在：{path}")
            return
        self._store = Chroma(
            persist_directory=str(new_path),
            embedding_function=llm_factory.get_client(self._embedding),
        )
        logger.info(f"临时加载向量库:{path}")

if __name__ == "__main__":
    # 创建向量库实例
    chroma_store = ChromaStore(ChromaVectorConfig())
    # 添加文本
    chroma_store.add_texts(["hello world", "hello rag"])

