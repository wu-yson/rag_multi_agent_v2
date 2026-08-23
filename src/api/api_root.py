from fastapi import APIRouter
from src.api.chat_router import router as chat_router
from src.api.memory_router import router as memory_router

# 全局总路由
api_router = APIRouter()

# 注册所有子模块路由
api_router.include_router(chat_router)
api_router.include_router(memory_router)