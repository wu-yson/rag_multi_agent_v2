"""纯文本和 Markdown 文件加载器。"""

from pathlib import Path

from src.rag_agent.rag.factory import register_loader
from src.rag_agent.rag.model import DocumentLoader, LoaderNotFoundException


@register_loader()
class TextLoader(DocumentLoader):
    """纯文本和 Markdown 文件加载器。"""
    SUPPORTED_EXTENSIONS = [".txt", ".md"]

    def load(self, file_path: str):
        """ 支持.txt 和 .md的文件加载器"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在:{file_path}")
        file_suffix = path.suffix.lower()
        if file_suffix not in self.SUPPORTED_EXTENSIONS:
            raise LoaderNotFoundException(file_suffix, self.SUPPORTED_EXTENSIONS) from None
        try:
            raw_text = path.read_text(encoding="utf-8")
            content = raw_text.strip()

        except UnicodeDecodeError as e:
            raise ValueError(f"文件编码错误:{file_path}: {e}") from e
        if not content:
            raise ValueError(f"文件内容为空:{file_path}")
        return [{"content": content,
                 # 元数据: {文件路径, 文件后缀}
                 "metadata": {"source": str(path), "suffix": path.suffix},
            }
        ]