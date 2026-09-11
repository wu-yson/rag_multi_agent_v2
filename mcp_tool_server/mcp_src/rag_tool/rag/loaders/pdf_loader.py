"""PDF 图文加载器：文本直抽，图片走多模态描述后以文本形式入库。"""

import base64

from pathlib import Path
from mcp_src.utils.logger import log
import pymupdf  # PyMuPDF

from mcp_src.config.settings import settings
from mcp_src.llm_tool.factory import llm_factory
from mcp_src.rag_tool.rag.factory import register_loader
from mcp_src.rag_tool.rag.model import DocumentLoader, LoaderNotFoundException




def _describe_image(image_bytes: bytes, page_text: str = "", image_ext: str = "png") -> str:
    """调用视觉模型生成图片描述，page_text 提供页面文字上下文。"""
    prompt = (
        "你是文档解析助手。请分析这张图片并输出结构化描述，用于知识库检索。\n"
        "必须包含以下四部分：\n"
        "1. 图片类型：图表 / 流程图 / 截图 / 表格 / 照片 / 其他\n"
        "2. 标题或主题\n"
        "3. 关键文字：逐字转录图中出现的文字\n"
        "4. 数据与结论：关键数字、趋势、流程或结论\n"
    )
    if page_text:
        prompt = f"当前页面文字内容：\n---\n{page_text[:200]}\n---\n\n" + prompt

    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    mime = "image/jpeg" if image_ext.lower() in ("jpg", "jpeg") else f"image/{image_ext.lower()}"
    client = llm_factory.get_client(settings.rag_vision_model)
    response = client.chat.completions.create(
        model=settings.rag_vision_model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:{mime};base64,{image_b64}"},
        ]}],
        max_tokens=1024,
    )
    return _extract_content(response.choices[0].message)


def _extract_content(message) -> str:
    """content 为空时回退到 reasoning/thinking，兼容会思考的视觉模型。"""
    content = getattr(message, "content", "") or ""
    if content:
        return content if isinstance(content, str) else str(content)
    raw = ""
    for attr in ("reasoning", "thinking", "reasoning_content"):
        value = getattr(message, attr, None)
        if value:
            raw = str(value)
            break
    if not raw:
        for meta in (getattr(message, "response_metadata", {}) or {},
                     getattr(message, "additional_kwargs", {}) or {}):
            if not isinstance(meta, dict):
                continue
            for key in ("reasoning", "thinking", "reasoning_content"):
                value = meta.get(key)
                if value:
                    raw = str(value)
                    break
            if raw:
                break
    return raw.replace("<think>", "").replace("</think>", "").strip()


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
            with pymupdf.open(str(path)) as pdf:
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

                    # 通道2：内嵌图片 -> 多模态描述（不落盘，描述文本直接入库）
                    for idx, info in enumerate(page.get_images(full=True), start=1):
                        try:
                            image = pdf.extract_image(info[0])
                            image_bytes = image["image"]

                            documents.append({
                                "content": _describe_image(image_bytes, page_text=text, image_ext=image.get("ext", "png")),
                                "metadata": {
                                    "source": str(path), "suffix": ".pdf",
                                    "content_type": "image", "page": pno,
                                },
                            })
                        except Exception as e:
                            log.warning(f"[PDF] 第{pno}页第{idx}张图处理失败: {e}")
        except Exception as e:
            raise ValueError(f"PDF 解析失败:{file_path}: {e}") from e

        if not documents:
            raise ValueError(f"PDF 未解析出任何可入库内容:{file_path}")
        return documents
