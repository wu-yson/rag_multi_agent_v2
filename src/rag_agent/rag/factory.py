from datetime import datetime
import hashlib,logging
import json
from pathlib import Path
from typing import Optional, Callable, Any, Dict

from src.config.settings import settings
from src.rag_agent.rag.model import (DocumentLoader,
                 TextSplitter,
                 VectorStoreProvider,
                 SplitConfig,
                 RAGConfig,
                 StoreNotFoundException,
                 LoaderNotFoundException,
                 )

logger = logging.getLogger(__name__)


# 台账持久化文件路径，你可以统一放到你的配置
BASE_DIR = Path(__file__).parent
KB_FILE_REGISTRY_PATH = BASE_DIR / "kb_file_registry.json"

def load_kb_file_registry() -> list[Dict]:
    """加载文件入库台账，不存在返回空列表"""
    if not KB_FILE_REGISTRY_PATH.exists():
        return []
    with open(KB_FILE_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_kb_file_registry(registry: list[Dict]):
    """保存台账，同名file_md5会覆盖旧条目"""
    logger.info(f"保存台账，总记录数：{len(registry)}, 文件列表: {registry}")
    with open(KB_FILE_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)





def build_rag_default_config() -> RAGConfig:
    return RAGConfig(
        vector_store_name=settings.rag_vector_store_name,
        chat_model=settings.rag_chat_model
    )



_BATCH_SIZE = 500

class RAGFactory:
    """RAG 抽象工厂，整合向量存储、文档加载、文本分割。"""
    def __init__(self) -> None:
        self._loader_registry: dict[str, type[DocumentLoader]] = {}
        self._splitter_registry: dict[str, type[TextSplitter]] = {}
        self._splitter_configs: dict[str, Callable[[], SplitConfig]] = {}
        self._default_splitter: Optional[str] = None
        self._store_configs: dict[str, Callable[[], Any]] = {}
        self._store_registry: dict[str, type[VectorStoreProvider]] = {}

    # ── 注册 ────────────────────────────────────────────
    def register_store(
        self,
        name: str,
        store_cls: type[VectorStoreProvider],
        config_fn: Callable[[], Any]
    ) -> None:
        """注册向量库类与对应默认配置生成函数"""
        self._store_registry[name] = store_cls
        self._store_configs[name] = config_fn

    def register_loader(self, loader_cls: type[DocumentLoader]) -> None:
        """注册文本加载器。"""
        for ext in getattr(loader_cls, "SUPPORTED_EXTENSIONS", []):
            self._loader_registry[ext] = loader_cls

    def register_splitter(
        self,
        name: str,
        splitter_cls: type[TextSplitter],
        config_fn: Callable[[], SplitConfig],
        default: bool = False
    ) -> None:
        """注册文本分割器。default=True 设为默认分割器。"""
        self._splitter_registry[name] = splitter_cls
        self._splitter_configs[name] = config_fn
        if default or self._default_splitter is None:
            self._default_splitter = name

    def create_store(self, name: str, config: Optional[Any] = None) -> VectorStoreProvider:
        """创建向量库。"""
        if name not in self._store_registry:
            raise StoreNotFoundException(name, list(self._store_registry))
        if config is None:
            config = self._store_configs[name]()
        return self._store_registry[name](config)

    def create_loader(self, file_path: str) -> DocumentLoader:
        """创建文档加载器实例。"""
        text = Path(file_path).suffix.lower()
        if text not in self._loader_registry:
            raise LoaderNotFoundException(text, list(self._loader_registry))
        return self._loader_registry[text]()

    def create_splitter(self, name: Optional[str] = None, **kwargs: Any) -> TextSplitter:
        """创建文本分割器实例。不传 name 则使用默认"""
        splitter_name = name or self._default_splitter
        if splitter_name not in self._splitter_registry:
            raise ValueError(
                f"splitter [{splitter_name}] not registered, available: {list(self._splitter_registry)}")
        return self._splitter_registry[splitter_name](**kwargs)

    # ── 去重 ────────────────────────────────────────────

    def _calc_file_md5(self, file_path: Path) -> str:
        """计算完整文件二进制MD5，用于全局整文件去重幂等"""
        md5_obj = hashlib.md5()
        with open(file_path, "rb") as f:
            while buffer := f.read(8192):
                md5_obj.update(buffer)
        return md5_obj.hexdigest()


    def _calc_chunk_md5(self, chunk_text: str) -> str:
        """计算文本块MD5，用于文件重复入库校验 """
        md5_obj = hashlib.md5(chunk_text.encode("utf-8"))
        return md5_obj.hexdigest()





    # ─── 轻量化入库核心方法（保留在类内，精简冗余重型逻辑）──────────────────────

    def index_directory(
        self,
        dir_path: str,
        store_name: str = None,
        store_config: Optional[Any] = None,
        splitter_name: Optional[str] = None,
        splitter_config: Optional[SplitConfig] = None
    ) -> VectorStoreProvider:
        """扫描目录 → 加载文档 → 分割 → 入库（分批写入 ）。"""
        vs_name = store_name if store_name is not None else build_rag_default_config().vector_store_name
        store = self.create_store(vs_name, store_config)
        real_name = splitter_name or self._default_splitter
        cfg = splitter_config or self._splitter_configs[real_name]()
        splitter = self.create_splitter(splitter_name, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
        directory = Path(dir_path)

        # ── 预扫描：入库前先检查所有文件后缀是否受支持 ──
        unsupported_files = []
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext not in self._loader_registry:
                    unsupported_files.append(f"{file_path.name}（{ext}）")

        if unsupported_files:
            raise RuntimeError(
                "以下文件格式不受支持，无法入库：" + "；".join(unsupported_files)
            )


        all_chunks: list[str] = []
        all_metadata: list[dict] = []
        stored_file_md5 = store.get_all_stored_file_md5()
        exist_chunk_md5_set: set[str] = set()

        old_registry = load_kb_file_registry()
        new_entries = []  # 本轮扫描新增的台账，和旧台账分开

        for file_path in directory.rglob("*"):
            if file_path.is_file():
                f_md5 = self._calc_file_md5(file_path)
                try:

                    if f_md5 in stored_file_md5:
                        logger.info(f"完整文件副本已存在，跳过 {file_path.name}")
                        continue
                    stored_file_md5.add(f_md5)
                    loader = self.create_loader(str(file_path))
                    documents = loader.load(str(file_path))
                    chunks = splitter.split(documents)
                    for chunk in chunks:
                        chunk_content = chunk["content"]
                        chunk_md5 = self._calc_chunk_md5(chunk_content)
                        if chunk_md5 in exist_chunk_md5_set:
                            continue

                        meta = chunk["metadata"].copy()
                        meta["file_md5"] = f_md5
                        exist_chunk_md5_set.add(chunk_md5)
                        all_chunks.append(chunk_content)
                        all_metadata.append(meta)

                    new_entries.append({
                        "file_md5": f_md5,
                        "file_name": file_path.name,
                        "file_path": str(file_path.resolve()),
                        "index_at": datetime.now().isoformat()
                    })

                except Exception as e:
                    logger.warning(f" 加载文件失败： {file_path}: {e}")
            if len(all_chunks) >= _BATCH_SIZE:
                store.add_texts(all_chunks, all_metadata)
                all_chunks, all_metadata = [], []
        if all_chunks:
            store.add_texts(all_chunks, all_metadata)
            store.save()

        merged = old_registry.copy()
        md5_map = {item["file_md5"]: item for item in merged}
        for entry in new_entries:
            md5_map[entry["file_md5"]] = entry
        final_registry = list(md5_map.values())

        save_kb_file_registry(final_registry)

        logger.info(f" 文本分块入库完成，目录： {dir_path} ")
        return store

    # ─── 检索 ────────────────────────────────────────────────────
    def query(self, query: str, store: VectorStoreProvider = None,
              config: Any = None, k: int = None) -> list[dict]:
        """向量库检索。通过用户输入内容检索相关内容, 不经过大模型判断"""
        if store is None:
            store = self.create_store(build_rag_default_config().vector_store_name, config)
        return store.similarity_search(query, k=k)




# 创建工厂实例
rag_factory = RAGFactory()


# ======================================================================
# 装饰器
# ======================================================================


def register_loader():
    """类装饰器：自动注册文档加载器。"""
    def decorator(cls: type[DocumentLoader]) -> type[DocumentLoader]:
        rag_factory.register_loader(cls)
        return cls
    return decorator

def register_splitter(name: str, default: bool = False):
    """类装饰器：自动注册文本分割器。"""
    def decorator(cls: type[TextSplitter]) -> type[TextSplitter]:
        # 调用注册器类方法
        rag_factory.register_splitter(name, cls, lambda: SplitConfig(), default=default)
        return cls
    return decorator

def register_vectorstore(name: str, config_fn: Callable[[], Any]):
    """类装饰器：自动注册向量库。"""
    def decorator(cls: type[VectorStoreProvider]) -> type[VectorStoreProvider]:
        # 调用注册器类方法
        rag_factory.register_store(name, cls, config_fn)
        return cls
    return decorator

