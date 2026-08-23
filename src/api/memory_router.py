from fastapi import APIRouter

from src.api.schemas import MemoryClearBody
from src.memory.memory import CommonMemory
from src.utils.logger import log

router = APIRouter()



@router.post("/memory/clear")
def memory_clear_api(body: MemoryClearBody):
    """
    清空指定会话记忆
    完整地址：http://127.0.0.1:8000/api/memory/clear
    """
    sid = body.session_id
    try:
        memory = CommonMemory(session_id=sid)
        memory.delete_session()
        return {
            "code": 200,
            "msg": "会话记忆清空成功",
            "data": None
        }
    except Exception as e:
        log.exception("记忆清空接口执行异常")
        return {
            "code": 500,
            "msg": str(e),
            "data": {}
        }