# 前端POST接口需要传给后端的字段
from pydantic import BaseModel

# 聊天接口专属请求体
class ChatBody(BaseModel):
    query: str
    session_id: str

# 当前记忆接口专属请求体
class MemoryClearBody(BaseModel):
    session_id: str