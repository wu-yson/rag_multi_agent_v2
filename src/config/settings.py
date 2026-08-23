from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
# 拿到项目根目录：src/config 往上两层
BASE_DIR = Path(__file__).parent.parent.parent

class AppSettings(BaseSettings):
    # 读取规则
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8"
    )


    #  qwen 通义千问 __
    qwen_api_key: SecretStr = SecretStr("")
    qwen_base_url: str = "https://ws-30uu5ov858l45bjs.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    qwen_timeout: int = 15
    qwen_temperature: float = 0.7

    # ollama本地 __
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 15
    ollama_temperature: float = 0.7

    # 记忆模块配置 __
    # 存储数据库文件路径
    memory_sqlite_file: str = str(BASE_DIR / "src/db/agent_memory_db")

    # 日志模块 __
    log_save_dir: str = str(BASE_DIR / "logs")  # 存储路径

    # chroma向量库 __
    chroma_persist_path: Optional[str] = str(BASE_DIR /"src/db/cloud_chroma_db")  # 持久化路径
    chroma_embedding_model: Optional[str] = "qwen3.7-text-embedding"  # 嵌入模型
    chroma_search_top_k: int = 4  # 搜索相似度检索返回条数

    # faiss向量库 __
    faiss_persist_path: str = str(BASE_DIR /"src/db/cloud_faiss_db")  # 持久化路径
    faiss_embedding_model: str = "qwen3.7-text-embedding"  # 嵌入模型
    faiss_search_top_k: int = 4  # 搜索相似度检索返回条数
    faiss_meta_path: str = str(BASE_DIR / "src/db/cloud_faiss_db/file_meta.json")   # 元数据存储路径

    # rag链路 __
    rag_vector_store_name: str = "chroma"   # 当前使用向量库
    rag_chat_model: Optional[str] = "qwen3.7-flash"   # 对话模型

    # 通用agent __
    agent_default_model: Optional[str] = "qwen3.7-flash"  # 所有agent使用模型
    agent_debug_mode: bool = True  # 调试模式


settings = AppSettings()