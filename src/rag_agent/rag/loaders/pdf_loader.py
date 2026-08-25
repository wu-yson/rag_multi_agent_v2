"""PDF 图文加载器：文本直抽，图片裁图后走多模态描述。"""

import base64
import logging
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.messages import HumanMessage

from src.config.settings import settings
from src.llm.factory import llm_factory
from src.rag_agent.rag.factory import register_loader
from src.rag_agent.rag.model import DocumentLoader, LoaderNotFoundException

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
ASSET_ROOT = BASE_DIR / "src" / "data" / "_pdf_assets"
MIME_MAP = {"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg"}


def _describe_image(image_bytes: bytes, ext: str, page_text: str = "") -> str:
    """调用多模态模型生成图片描述，page_text 提供页面文字上下文。"""
    mime = MIME_MAP.get(ext, "image/png")
    prompt = (
        "你是文档解析助手。请分析这张图片并输出结构化描述，用于知识库检索。\n"
        "必须包含以下四部分：\n"
        "1. 图片类型：图表 / 流程图 / 截图 / 表格 / 照片 / 其他\n"
        "2. 标题或主题\n"
        "3. 关键文字：逐字转录图中出现的文字\n"
        "4. 数据与结论：关键数字、趋势、流程或结论\n"
    )
    if page_text:
        prompt = f"当前页面文字内容：\n---\n{page_text[:500]}\n---\n\n" + prompt
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url",
         "image_url": {"url": f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"}},
    ])
    try:
        client = llm_factory.get_client(settings.rag_vision_model)
        return str(client.invoke([message]).content)
    except Exception as e:
        return f"[图片描述生成失败] {e}"


@register_loader()
class PDFLoader(DocumentLoader):
    """PDF 加载器：每页返回文本块 + 图片描述块。"""

    SUPPORTED_EXTENSIONS = [".pdf"]

    def load(self, file_path: str) -> list[dict]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在:{file_path}")
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise LoaderNotFoundException(path.suffix, self.SUPPORTED_EXTENSIONS)

        documents = []
        try:
            with fitz.open(str(path)) as pdf:
                for page in pdf:
                    pno = page.number + 1

                    # 通道1：文本层直抽，零模型调用
                    text = page.get_text("text").strip()
                    if text:
                        documents.append({
                            "content": text,
                            "metadata": {
                                "source": str(path), "suffix": ".pdf",
                                "content_type": "text", "page": pno,
                            },
                        })

                    # 通道2：内嵌图片 -> 落盘 -> 多模态描述
                    for idx, info in enumerate(page.get_images(full=True), start=1):
                        try:
                            image = pdf.extract_image(info[0])
                            image_bytes = image["image"]
                            ext = image.get("ext", "png")

                            image_path = ASSET_ROOT / path.stem / f"page{pno:03d}_img{idx:03d}.{ext}"
                            image_path.parent.mkdir(parents=True, exist_ok=True)
                            image_path.write_bytes(image_bytes)

                            documents.append({
                                "content": _describe_image(image_bytes, ext, page_text=text),
                                "metadata": {
                                    "source": str(path), "suffix": ".pdf",
                                    "content_type": "image", "page": pno,
                                    "image_path": str(image_path),
                                },
                            })
                        except Exception as e:
                            logger.warning(f"[PDF] 第{pno}页第{idx}张图处理失败: {e}")
        except Exception as e:
            raise ValueError(f"PDF 解析失败:{file_path}: {e}") from e

        if not documents:
            raise ValueError(f"PDF 未解析出任何可入库内容:{file_path}")
        return documents
