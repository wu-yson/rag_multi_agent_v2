import json
from pathlib import Path
from typing import Optional, List, Any

from langchain_community.vectorstores import FAISS
import logging

from src.config.settings import settings
from src.llm.factory import llm_factory
from src.rag_agent.rag.factory import register_vectorstore
from src.rag_agent.rag.model import VectorStoreProvider, FaissVectorConfig

logger = logging.getLogger(__name__)

def build_faiss_default_config() -> FaissVectorConfig:
    return FaissVectorConfig(
        persist_path=settings.faiss_persist_path,
        embedding_model=settings.faiss_embedding_model,
        search_top_k=settings.faiss_search_top_k,
        meta_path=settings.faiss_meta_path
    )




@register_vectorstore("faiss",lambda: build_faiss_default_config())
class FAISSStore(VectorStoreProvider):
    """
    FAISS 向量存储后端，对齐顶层抽象接口规范
    """
    def __init__(self, config: FaissVectorConfig) -> None:
        self._config = config

        self.k = self._config.search_top_k
        self._embedding = self._config.embedding_model
        if self._config.persist_path is None or not self._config.persist_path.strip():
            raise ValueError("向量库持久化路径不能为空")

        self.persist_path = Path(self._config.persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self._store: Optional[FAISS] = None
        self._meta_path = self._config.meta_path

        if self.persist_path and self.persist_path.exists():
            try:
                self._store = FAISS.load_local(
                    folder_path=str(self.persist_path),  # 持久化路径
                    embeddings=llm_factory.get_client(self._embedding),   # 嵌入模型
                    allow_dangerous_deserialization=True,  # 安全配置参数
                )
                logging.info(f"从{self.persist_path} 载 FAISS 向量库")
                return
            except Exception as e:
                logging.error(f"加载失败，将创建全新FAISS库: {e}")

        self._store = FAISS.from_texts(
            texts=["__init_placeholder__"],
            embedding=llm_factory.get_client(self._embedding),
            metadatas=[{"_tmp": True}],
        )
        all_doc_ids = list(self._store.index_to_docstore_id.values())
        if all_doc_ids:
            self._store.delete(all_doc_ids)
        logger.info(f"创建新 FAISS 向量库: {self.persist_path}")

    def _load_meta_mapping(self) -> dict[str, str]:
        """读取文件md5映射表"""
        if not self._meta_path.exists():
            return {}
        try:
            with open(self._meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_meta_mapping(self, mapping: dict[str, str]):
        """写入文件md5映射表"""
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

    def add_texts(self, texts: list[str], metadatas: Optional[list[dict]] = None,
                  ids: Optional[List[str]] = None) -> None:
        texts_list, metadatas_list, new_file_records = self._clean_text_batch(texts, metadatas)
        if not texts_list:
            logger.warning("清洗后无有效内容，放弃入库")
            return

        self._store.add_texts(
            texts=texts_list,
            metadatas=metadatas_list,
        )

        # FAISS特有：同步更新file_meta.json
        if new_file_records:
            meta_map = self._load_meta_mapping()
            meta_map.update(new_file_records)
            self._save_meta_mapping(meta_map)

        logger.info(f"向量库添加 {len(texts_list)} 条数据")

    def similarity_search(self, query: str, k: int = None) -> list[dict[str, Any]]:
        use_k = k if k is not None else self.k
        docs = self._store.similarity_search(query, k=use_k)
        return self._wrap_search_result(docs)



    def get_all_stored_file_md5(self) -> set[str]:
        """读取所有已经入库的完整文件md5集合（上层index_directory调用）"""
        meta_map = self._load_meta_mapping()
        return set(meta_map.keys())


    def save(self, path: Optional[str] = None) -> None:
        """
        保存索引到磁盘
        """
        if path is None:
            save_path = self.persist_path
        else:
            save_path = Path(path)
        self._store.save_local(str(save_path))
        logger.info(f"保存 FAISS 索引到 {save_path}")


    def delete_texts(self, ids: List[str]) -> int:
        """FAISS 不支持按 ID 删除，抛出明确异常"""
        raise NotImplementedError("FAISS 底层不支持按 ID 删除，请重建索引或使用其他向量库")

    @property
    def store(self) -> FAISS:
        """直接暴露原生 FAISS 对象"""
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

        self._store = FAISS.load_local(
            folder_path= str(new_path),
            embeddings=self._embedding,
            allow_dangerous_deserialization=True
        )
        logger.info(f"临时加载向量库:{path}")

if __name__ == "__main__":
    # 创建向量库实例
    chroma_store = FAISSStore(build_faiss_default_config())
    # 添加文本
    chroma_store.add_texts(["hello world", "hello rag"])

