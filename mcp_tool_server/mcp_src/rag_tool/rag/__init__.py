"""RAG 包：入口各组件以 @register_loader/@register_splitter/@register_vectorstore 注册。"""

from mcp_src.rag_tool.rag.stores import faiss, chroma, milvus_lite
from mcp_src.rag_tool.rag.loaders import text_loader, pdf_loader
from mcp_src.rag_tool.rag.splitters import recursive
