import re
from fastapi import APIRouter
from sqlmodel import Session, select, func
from src.memory.memory import global_engine, ChatRecord, CommonMemory
from src.utils.logger import log

router = APIRouter()


def _get_title(session_id: str) -> str:
    """会话标题：取第一条用户消息，优先文件名，否则截前20字"""
    with Session(global_engine) as session:
        first = session.exec(
            select(ChatRecord)
            .where(ChatRecord.session_id == session_id, ChatRecord.role == "human")
            .order_by(ChatRecord.create_time.asc())
            .limit(1)
        ).first()
        if first is None:
            return "新对话"
        text = first.content.strip()
        m = re.search(r"([\w\u4e00-\u9fa5]+\.\w+)", text)
        if m:
            return m.group(1)
        return text[:20]


@router.get("/conversations")
async def list_conversations():
    """会话列表：按 session_id 聚合，最新会话在前"""
    try:
        with Session(global_engine) as session:
            rows = session.exec(
                select(
                    ChatRecord.session_id,
                    func.count(ChatRecord.id),
                    func.min(ChatRecord.create_time),
                    func.max(ChatRecord.create_time),
                )
                .where(ChatRecord.deleted == False)
                .group_by(ChatRecord.session_id)
                .order_by(func.max(ChatRecord.create_time).desc())
            ).all()
        data = [
            {
                "session_id": sid,
                "title": _get_title(sid),
                "message_count": cnt,
                "created_at": str(created_at),
                "updated_at": str(updated_at),
            }
            for sid, cnt, created_at, updated_at in rows
        ]
        return {"code": 200, "data": data}
    except Exception as e:
        log.exception("会话列表接口异常")
        return {"code": 500, "msg": str(e), "data": []}


@router.get("/conversations/{session_id}/messages")
async def get_conversation_messages(session_id: str):
    """某会话完整消息（时间正序）"""
    try:
        with Session(global_engine) as session:
            rows = session.exec(
                select(ChatRecord)
                .where(
                    ChatRecord.session_id == session_id,
                    ChatRecord.deleted == False,
                )
                .order_by(ChatRecord.create_time.asc())
            ).all()
        data = [
            {"role": r.role, "content": r.content, "create_time": str(r.create_time)}
            for r in rows
        ]
        return {"code": 200, "data": data}
    except Exception as e:
        log.exception("会话消息接口异常")
        return {"code": 500, "msg": str(e), "data": []}


@router.delete("/conversations/{session_id}")
async def delete_conversation(session_id: str):
    """删除会话（软删：打标记不删数据，可恢复）"""
    try:
        from sqlmodel import update as sql_update
        with Session(global_engine) as session:
            session.exec(
                sql_update(ChatRecord)
                .where(ChatRecord.session_id == session_id)
                .values(deleted=True)
            )
            session.commit()
        return {"code": 200, "msg": "会话删除成功", "data": None}
    except Exception as e:
        log.exception("会话删除接口异常")
        return {"code": 500, "msg": str(e), "data": None}