import json
from src.mcp.client import begin_request, end_request
from fastapi import APIRouter
from starlette.responses import StreamingResponse
from src.config.settings import settings
from src.api.schemas import ChatBody
from src.llm.factory import llm_factory
from src.memory.memory import CommonMemory
from src.supervisor_agent.supervisor_agent import SupervisorAgent
from src.utils.logger import log

# 整个文件只实例化一次
router = APIRouter()

@router.get("/models")
async def list_models():
    """ 列出所有支持的对话模型（给前端下拉框用，chat + vision，排除 embed） """
    chat_models = llm_factory.get_supported_models(model_types=["chat", "vision"])
    return {
        "code": 200,
        "data": chat_models,
        "default_model": settings.agent_default_model,  # 新增：默认模型名
    }



@router.post("/chat/stream")
async def chat_api(body: ChatBody):
    """ 对话接口 """

    async def event_generator():
        token = None
        try:
            token = begin_request(body.session_id, body.workspace_path or "")
            memory = CommonMemory(session_id=body.session_id)
            agent = SupervisorAgent(memory=memory)

            query = body.query
            if body.workspace_path:
                query = f"[当前工作目录：{body.workspace_path}]\n\n{query}"

            async for item in agent.astream(query, tmp_model=body.model):
                if isinstance(item, tuple):
                    chunk_type, chunk_data = item
                    yield f"data: {json.dumps({chunk_type: chunk_data}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'content': item}, ensure_ascii=False)}\n\n"
        except Exception as e:
            log.exception("流式接口异常")
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            if token is not None:
                end_request(token)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

