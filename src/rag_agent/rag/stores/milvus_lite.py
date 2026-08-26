"""Milvus Lite 向量存储：走 LangChain 封装，实现稠密向量 + BM25 混合检索。"""

from pathlib import Path
from typing import Optional, List, Any
import logging

from langchain_milvus import Milvus, BM25BuiltInFunction

from src.llm.factory import llm_factory
from src.rag_agent.rag.factory import register_vectorstore
from src.rag_agent.rag.model import VectorStoreProvider, MilvusLiteVectorConfig
from src.config.settings import settings

logger = logging.getLogger(__name__)


def build_milvus_default_config() -> MilvusLiteVectorConfig:
    """读取 Milvus Lite 专属配置。"""
    return MilvusLiteVectorConfig(
        persist_path=settings.milvus_persist_path,
        embedding_model=settings.milvus_embedding_model,
        search_top_k=settings.milvus_search_top_k,
    )


@register_vectorstore("milvus", lambda: build_milvus_default_config())
class MilvusLiteStore(VectorStoreProvider):
    """Milvus Lite 向量存储，基于 LangChain Milvus 封装实现稠密 + BM25 混合检索。"""

    def __init__(self, config: MilvusLiteVectorConfig) -> None:
        self._config = config
        self._embedding = self._config.embedding_model
        self.k = self._config.search_top_k

        if self._config.persist_path is None or not self._config.persist_path.strip():
            raise ValueError("向量库持久化路径不能为空")

        self.persist_path = Path(self._config.persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self._uri = str(self.persist_path / "milvus_lite.db")
        self._store: Optional[Milvus] = None
        self._init_store()
        logger.info(f"加载 Milvus 向量库: {self._uri}")

    def _init_store(self) -> None:
        """创建或加载 LangChain Milvus 客户端，sparse 由 BM25 内置函数自动生成。"""
        self._store = Milvus(
            embedding_function=llm_factory.get_client(self._embedding),
            builtin_function=BM25BuiltInFunction(),  # BM25检索
            vector_field=["dense", "sparse"],   # 双路检索 稠密 + 稀疏双路
            # BRUTE_FORCE 避免 Windows 中文路径下 FAISS 落盘失败。
            index_params=[
                {"index_type": "BRUTE_FORCE", "metric_type": "COSINE", "params": {}},
                {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25", "params": {}},
            ],
            search_params=[
                {"metric_type": "COSINE", "params": {}},
                {"metric_type": "BM25", "params": {}},
            ],
            collection_name="documents",
            connection_args={
                "uri": self._uri,
                # 降低 keepalive 频率，避免 Milvus Lite 报 too_many_pings。
                "grpc_options": {
                    "grpc.keepalive_time_ms": 300000,
                    "grpc.keepalive_timeout_ms": 20000,
                    "grpc.keepalive_permit_without_calls": False,
                },
            },
            auto_id=True,
            enable_dynamic_field=True,
        )

    def get_all_stored_file_md5(self) -> set[str]:
        if not self._store.client.has_collection("documents"):
            return set()
        rows = self._store.client.query(
            collection_name="documents",
            filter="pk >= 0",
            output_fields=["file_md5"],
        )
        return {item["file_md5"] for item in rows if item.get("file_md5")}

    def add_texts(self, texts: list[str], metadatas: Optional[list[dict]] = None,
                  ids: Optional[List[str]] = None) -> None:
        texts_list, metadatas_list, _ = self._clean_text_batch(texts, metadatas)
        if not texts_list:
            logger.warning("清洗后无有效内容，放弃入库")
            return
        self._store.add_texts(texts=texts_list, metadatas=metadatas_list, ids=ids)
        logger.info(f"Milvus 添加 {len(texts_list)} 条数据")

    def similarity_search(self, query: str, k: int = None) -> list[dict[str, Any]]:
        use_k = k if k is not None else self.k
        docs = self._store.similarity_search(query, k=use_k, ranker_type="rrf")  # ranker_type="rrf" 上面两路结果合并排序  RRF 融合
        results = []
        for doc in docs:
            metadata = dict(doc.metadata)
            metadata.pop("pk", None)
            results.append({"content": doc.page_content, "metadata": metadata})
        return results

    def save(self, path: Optional[str] = None) -> None:
        """Milvus Lite 自动持久化，无需手动保存。"""

    def delete_texts(self, ids: List[str]) -> int:
        if not ids:
            return 0
        deleted = self._store.delete(ids=ids)
        return len(ids) if deleted else 0

    @property
    def store(self) -> Milvus:
        """直接暴露 LangChain Milvus 对象。"""
        return self._store

    def load(self, path: str) -> None:
        """切换加载指定路径的 Milvus Lite 向量库。"""
        new_path = Path(path)
        new_path.mkdir(parents=True, exist_ok=True)
        self._uri = str(new_path / "milvus_lite.db")
        self._init_store()
        logger.info(f"切换加载 Milvus 向量库: {self._uri}")


if __name__ == "__main__":
    milvus_store = MilvusLiteStore(build_milvus_default_config())
    milvus_store.add_texts(["hello world", "hello rag"])