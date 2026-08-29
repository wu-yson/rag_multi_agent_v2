"""验证官方 with_retry：假模型第一次失败、第二次成功"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import asyncio

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typing import ClassVar

from src.llm.resilience import ResilientChatModel





class FakeFlakyModel(BaseChatModel):
    _count: ClassVar[int] = 0

    @property
    def _llm_type(self) -> str:
        return "fake-flaky"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        FakeFlakyModel._count += 1
        if FakeFlakyModel._count == 1:
            raise ConnectionError("第一次抖动")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="重试成功"))])


async def main():
    wrapped = ResilientChatModel(inner=FakeFlakyModel(), model_name="flaky-model")
    wrapped = wrapped.with_retry(
        stop_after_attempt=2,
        retry_if_exception_type=(ConnectionError, TimeoutError),
    )
    resp = await wrapped.ainvoke("你好")
    print("重试后结果:", resp.content)            # 预期：重试成功
    print("实际调用次数:", FakeFlakyModel._count)  # 预期：2


if __name__ == "__main__":
    asyncio.run(main())