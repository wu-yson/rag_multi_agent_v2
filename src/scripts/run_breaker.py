"""单独测试熔断：通过 ResilientChatModel 包装，closed → open → 拦截"""
import asyncio

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult

from src.llm.resilience import ResilientChatModel, get_breaker


class FakeFailModel(BaseChatModel):
    """假模型：每次调用都抛错，用来触发熔断"""

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise ConnectionError("模拟模型挂了")


async def main():
    wrapped = ResilientChatModel(inner=FakeFailModel(), model_name="fake-model")

    for i in range(3):
        try:
            await wrapped.ainvoke("你好")
        except Exception as e:
            print(f"第{i+1}次调用失败: {type(e).__name__}: {e}")

    print("熔断状态:", get_breaker("fake-model").state)   # 预期 open

    try:
        await wrapped.ainvoke("你好")
    except Exception as e:
        print(f"打开后被拦截: {type(e).__name__}: {e}")   # 预期 CircuitOpenError


if __name__ == "__main__":
    asyncio.run(main())