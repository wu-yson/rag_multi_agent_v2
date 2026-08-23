"""递归字符分割器 — 按字符数递归分割，优先在段落/句子边界切分。"""
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag_agent.rag.factory import register_splitter
from src.rag_agent.rag.model import TextSplitter, SplitConfig


@register_splitter("recursive", default=True)
class RecursiveCharacterSplitter(TextSplitter):
    """递归字符分割器，优先在段落、换行、句号等边界切分。"""

    def __init__(self, config: SplitConfig = None, **kwargs: Any) -> None:
        self._config = config or SplitConfig()
        self._chunk_size = self._config.chunk_size
        self._chunk_overlap = self._config.chunk_overlap


    def split(self, documents: list[dict]) -> list[dict]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,  # ->  允许最大分隔的字符数
            chunk_overlap=self._chunk_overlap,  # ->  允许重叠的字符数

        )
        chunks = []

        for doc in documents:
            texts = splitter.split_text(doc["content"])
            for text in texts:
                clean = text.strip()
                if clean:
                    chunks.append({"content": text, "metadata": doc["metadata"]})
        return  chunks