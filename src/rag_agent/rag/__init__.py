"""RAG 包：入口各组件以 @register_loader/@register_splitter/@register_vectorstore 注册。"""

from src.rag_agent.rag.stores import faiss, chroma, milvus_lite
from src.rag_agent.rag.loaders import text_loader, pdf_loader
from src.rag_agent.rag.splitters import character, recursive
