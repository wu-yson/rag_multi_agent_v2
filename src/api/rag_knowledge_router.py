# -*- coding: utf-8 -*-
import json
from pathlib import Path
from fastapi import APIRouter
from src.utils.logger import log

router = APIRouter()

REGISTRY_PATH = Path("D:/python-xuexi/\u9762\u8bd5\u6587\u4ef6/MCP\u5de5\u5177\u7ec4\u4ef6/mcp_tool_server/mcp_src/rag_tool/rag/kb_file_registry.json")


@router.get("/knowledge/documents")
async def list_knowledge_documents():
    """ \u5217\u51fa\u5df2\u5165\u5e93\u7684\u77e5\u8bc6\u5e93\u6587\u6863 """
    try:
        if not REGISTRY_PATH.exists():
            return {"code": 200, "data": []}

        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = json.load(f)

        documents = [
            {
                "name": item["file_name"],
                "type": item["file_name"].rsplit(".", 1)[-1],
                "indexed_at": item["index_at"],
            }
            for item in registry
        ]
        return {"code": 200, "data": documents}
    except Exception as e:
        log.error(f"[RagKnowledge] \u83b7\u53d6\u77e5\u8bc6\u5e93\u6587\u6863\u5931\u8d25: {e}")
        return {"code": 500, "msg": str(e), "data": []}
