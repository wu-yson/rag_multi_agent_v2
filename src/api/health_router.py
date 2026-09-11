from fastapi import APIRouter

from src.llm.resilience import get_all_breaker_states

router = APIRouter()


@router.get("/health/breakers")
async def health_breakers():
    """ 返回每个模型的熔断状态：{模型名: closed/open/half_open} """
    return {"code": 200, "data": get_all_breaker_states()}