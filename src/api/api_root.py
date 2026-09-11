from fastapi import APIRouter
from src.api.file_router import router as file_router
from src.api.conversation_router import router as conversation_router
from src.api.chat_router import router as chat_router
from src.api.memory_router import router as memory_router
from src.api.rag_knowledge_router import router as rag_knowledge_router
from src.api.health_router import router as health_router
# 全局总路由
api_router = APIRouter()

# 注册所有子模块路由
api_router.include_router(chat_router)
api_router.include_router(memory_router)
api_router.include_router(rag_knowledge_router)
api_router.include_router(health_router)
api_router.include_router(conversation_router)

api_router.include_router(file_router)