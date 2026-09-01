"""模型调用弹性层：熔断（手写极简版）+ 兜底 + 重试（agent 中间件）"""
import time
import httpx
import openai
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRetryMiddleware,
    ModelFallbackMiddleware,
)

from src.config.settings import settings
from src.utils.logger import log


# 重试白名单
TRANSIENT_ERRORS = (
    TimeoutError,
    ConnectionError,
    httpx.ConnectError,
    httpx.TimeoutException,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class CircuitOpenError(RuntimeError):
    """熔断打开时抛出"""
    pass


class AsyncCircuitBreaker:
    """极简熔断器：closed / open / half_open 三态 + 失败计数"""

    def __init__(self, name: str, fail_max: int = 3, reset_timeout: int = 30):
        self.name = name
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self._fail_count = 0
        self._state = "closed"
        self._opened_at = 0.0

    @property
    def state(self) -> str:
        """closed / open / half_open"""
        return self._state

    def allow_request(self) -> bool:
        """熔断是否允许放行请求"""
        if self._state == "open":
            # 冷却时间到了 → 半开，放一个试探请求
            if time.time() - self._opened_at >= self.reset_timeout:
                self._state = "half_open"
                log.warning(f"[resilience] 熔断器 [{self.name}] 状态变化 -> half_open")
                return True
            return False
        return True

    def call_success(self):
        """报告一次成功调用"""
        self._fail_count = 0
        if self._state == "half_open":
            self._state = "closed"
            log.warning(f"[resilience] 熔断器 [{self.name}] 状态变化 -> closed")

    def call_failure(self):
        """报告一次失败调用；连续失败达到阈值 → 打开"""
        if self._state == "half_open":
            # 半开试探失败 → 重新打开
            self._state = "open"
            self._opened_at = time.time()
            log.warning(f"[resilience] 熔断器 [{self.name}] 状态变化 -> open")
            return
        self._fail_count += 1
        if self._fail_count >= self.fail_max:
            self._state = "open"
            self._opened_at = time.time()
            log.warning(f"[resilience] 熔断器 [{self.name}] 状态变化 -> open")


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


def _model_name(model) -> str:
    """从模型对象提取模型名（ChatOpenAI 用 model_name，ChatOllama 用 model）"""
    return (
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or str(model)
    )


class CircuitBreakerMiddleware(AgentMiddleware):
    """熔断中间件：拦截每次模型调用，熔断打开时快速失败"""

    async def awrap_model_call(self, request, handler):
        name = _model_name(request.model)
        breaker = get_breaker(name)
        if not breaker.allow_request():
            raise CircuitOpenError(f"熔断已打开: {name}")
        try:
            response = await handler(request)
            breaker.call_success()
            return response
        except CircuitOpenError:
            raise
        except Exception as e:
            breaker.call_failure()
            log.warning(f"[resilience] 模型 {name} 调用失败: {e}")
            raise


def build_agent_middleware(primary_model, client_loader):
    """组装 agent 弹性中间件：熔断 → 兜底 → 重试
    :param primary_model: 主模型客户端
    :param client_loader: 取客户端的函数（传 llm_factory.get_client）
    """
    middleware = [CircuitBreakerMiddleware()]

    # 兜底：备用模型（用真实客户端对象）
    fallback_clients = []
    for m in settings.model_fallback_chain:
        try:
            fallback_clients.append(client_loader(m))
        except Exception as e:
            log.warning(f"[resilience] 备用模型 {m} 获取失败: {e}")
    if fallback_clients:
        middleware.append(ModelFallbackMiddleware(primary_model, *fallback_clients))

    # 重试：临时抖动重试 1 次
    middleware.append(ModelRetryMiddleware(max_retries=1, retry_on=TRANSIENT_ERRORS, on_failure="error"))
    return middleware
