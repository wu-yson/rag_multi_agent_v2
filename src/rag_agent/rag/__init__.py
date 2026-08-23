"""RAG 包——导入各组件以触发 @register_loader/@register_splitter/@register_vectorstore 注册。"""

from src.rag_agent.rag.stores import faiss, chroma
from src.rag_agent.rag.loaders import text_loader
from src.rag_agent.rag.splitters import character, recursive
