from fastapi import APIRouter

from src.api.schemas import ChatBody
from src.memory.memory import CommonMemory
from src.supervisor_agent.supervisor_agent import SupervisorAgent
from src.utils.logger import log

# 整个文件只实例化一次
router = APIRouter()

@router.post("/chat")
def chat_api(body: ChatBody):
    """ 对话接口 """
    question = body.query
    sid = body.session_id
    try:
        # 每次请求新建内存、agent，会话隔离
        memory = CommonMemory(session_id=sid)
        agent = SupervisorAgent(memory=memory)
        resp = agent.invoke(question)

        return {
            "code": 200,
            "msg": "ok",
            "data": resp
        }
    except Exception as e:
        log.exception("接口执行异常")
        return {
            "code": 500,
            "msg": str(e),
            "data": {}
        }
