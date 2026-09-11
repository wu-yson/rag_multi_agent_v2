from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from src.memory.memory import ChatRecord, CommonMemory


def test_get_recent_excludes_tools_from_context_budget():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    session_id = "test-context-budget"

    with Session(engine) as session:
        session.add_all([
            ChatRecord(session_id=session_id, role="human", content="question", single_token=3),
            ChatRecord(session_id=session_id, role="ai", content="answer", single_token=4),
            ChatRecord(session_id=session_id, role="tool", content="tool-summary", single_token=8),
            ChatRecord(session_id=session_id, role="tool", content="tool-artifact", single_token=9),
        ])
        session.commit()

    memory = CommonMemory(
        session_id=session_id,
        max_context_tokens=10,
        engine=engine,
    )

    assert memory.get_recent() == [
        {"role": "human", "content": "question"},
        {"role": "ai", "content": "answer"},
    ]
