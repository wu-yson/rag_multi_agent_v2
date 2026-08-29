"""模型调用弹性层：熔断 + 重试 + 运行时兜底"""
from typing import Any, Optional

import pybreaker

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from src.utils.logger import log




class CircuitOpenError(RuntimeError):
    """熔断打开时抛出"""
    pass


class AsyncCircuitBreaker:
    """pybreaker 的封装：用它的 call_async / call，熔断打开时翻译成 CircuitOpenError"""

    def __init__(self, name: str, fail_max: int = 3, reset_timeout: int = 30):
        self.name = name
        self._breaker = pybreaker.CircuitBreaker(
            fail_max=fail_max,
            reset_timeout=reset_timeout,
        )

    @property
    def state(self) -> str:
        return self._breaker.current_state  # closed / open / half_open

    async def call(self, coro_factory):
        """执行异步调用；熔断打开时抛 CircuitOpenError"""
        try:
            return await self._breaker.call_async(coro_factory)
        except pybreaker.CircuitBreakerError as e:
            raise CircuitOpenError(f"熔断已打开: {self.name}") from e

    def call_sync(self, func):
        """执行同步调用；熔断打开时抛 CircuitOpenError"""
        try:
            return self._breaker.call(func)
        except pybreaker.CircuitBreakerError as e:
            raise CircuitOpenError(f"熔断已打开: {self.name}") from e


# 每个模型一个熔断器，互不影响
_breakers: dict[str, AsyncCircuitBreaker] = {}


def get_breaker(model_name: str) -> AsyncCircuitBreaker:
    """按模型名取熔断器，不存在则创建"""
    if model_name not in _breakers:
        _breakers[model_name] = AsyncCircuitBreaker(name=model_name)
    return _breakers[model_name]


def get_all_breaker_states() -> dict[str, str]:
    """给健康检查接口用：{模型名: 熔断状态}"""
    return {name: br.state for name, br in _breakers.items()}


class ResilientChatModel(BaseChatModel):
    """带熔断的模型包装：继承 BaseChatModel，所有 agent 调用自动受保护"""

    inner: BaseChatModel            # 真正的客户端（ChatOpenAI / ChatOllama）
    model_name: str = ""            # 当前模型名（熔断器按它隔离）
    fallback_models: list[str] = [] # 备用模型列表（兜底用，任务4再加）

    @property
    def _llm_type(self) -> str:
        return "resilient"



    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        breaker = get_breaker(self.model_name)
        return breaker.call_sync(
            lambda: self.inner._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        )