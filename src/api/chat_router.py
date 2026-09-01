import json

from fastapi import APIRouter
from starlette.responses import StreamingResponse

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
    return {"code": 200, "data": chat_models}



@router.post("/chat/stream")
async def chat_api(body: ChatBody):
    """ 对话接口 """

    async def event_generator():
        try:
            memory = CommonMemory(session_id=body.session_id)
            agent = SupervisorAgent(memory=memory)
            async for chunk in agent.astream(body.query, tmp_model=body.model):
                yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
        except Exception as e:
            log.exception("流式接口异常")
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

