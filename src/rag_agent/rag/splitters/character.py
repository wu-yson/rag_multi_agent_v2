"""按固定字符数分割器 — 简单按字符数切割，不关心语义边界。"""

from langchain_text_splitters import CharacterTextSplitter
from src.rag_agent.rag.factory import register_splitter
from src.rag_agent.rag.model import TextSplitter, SplitConfig




@register_splitter("character")
class CharacterSplitter(TextSplitter):
    """按固定字符数分割，可选分隔符。"""

    def __init__(self, config: SplitConfig, separator: str = "\n\n", **kwargs) -> None:
        self._config = config or SplitConfig()
        self._chunk_size = self._config.chunk_size
        self._chunk_overlap = self._config.chunk_overlap
        self._separator = separator

    def split(self, documents: list[dict]) -> list[dict]:
        splitter = CharacterTextSplitter(
            chunk_size=self._chunk_size,  # ->  允许最大分隔的字符数
            chunk_overlap=self._chunk_overlap,  # ->  允许重叠的字符数
            separator=self._separator,
        )
        chunks = []

        for doc in documents:
            texts = splitter.split_text(doc["content"])
            for text in texts:
                clean = text.strip()
                if not clean:
                    chunks.append({"content": text, "metadata": doc["metadata"]})
        return  chunks