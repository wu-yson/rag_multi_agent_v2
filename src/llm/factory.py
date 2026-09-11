from typing import Any

from src.config.settings import settings
from src.utils.logger import log
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from src.llm.models import ProviderInitializationError, ProviderConfig, ModelNotSupportedError


# ========== 加载供应商配置 ==========

def load_glm_config() -> ProviderConfig:
    """ 加载 云端供应商 glm 配置。"""
    return ProviderConfig(
        api_key=settings.glm_api_key.get_secret_value(),
        base_url=settings.glm_base_url,
        timeout=settings.glm_timeout,
        temperature = settings.glm_temperature
    )



def load_qwen_config() -> ProviderConfig:
    """ 加载 云端供应商 Qwen 配置。"""
    return ProviderConfig(
        api_key=settings.qwen_api_key.get_secret_value(),
        base_url=settings.qwen_base_url,
        timeout=settings.qwen_timeout,
        temperature = settings.qwen_temperature
    )

def load_ollama_config() -> ProviderConfig:
    """ 加载 本地模型Ollama 配置。"""
    return ProviderConfig(
        api_key=None,
        base_url=settings.ollama_base_url,
        timeout=settings.ollama_timeout,
        temperature=settings.ollama_temperature
    )

# ========== 供应商类（负责创建客户端） ==========
class GLMNProvider:
    """ glm供应商类, 创建客户端."""
    supported_models_chat = ["glm-5.3-flash"]
    supported_models_embed = []

    def __init__(self, config: ProviderConfig):
        if not config.api_key:
            raise ProviderInitializationError("qwen API Key 未设置，请检查api")
        self.config =  config

    def get_client(self, model_name: str) -> BaseChatModel | DashScopeEmbeddings:
        """ 获取glm模型客户端。"""
        if model_name not in self.supported_models_chat and model_name not in self.supported_models_embed:
            raise ModelNotSupportedError(f"模型 {model_name} 不是glm模型, 仅支持模型: {self.supported_models_chat}, {self.supported_models_embed}")
        if model_name in self.supported_models_chat:
            log.info(f" [LLM] 使用云端供应商 聊天模型: {model_name}")
            return ChatOpenAI(
                model=model_name,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                temperature=self.config.temperature,
                reasoning_effort=settings.llm_reasoning_effort,
                max_retries=1,
            )
        else:
            log.info(f" [LLM] 使用云端供应商 文本嵌入模型: {model_name}")
            return DashScopeEmbeddings(
                model=model_name,
                dashscope_api_key=self.config.api_key,
                max_retries=1,
            )


class QWENProvider:
    """ QWEN供应商类, 创建客户端."""
    supported_models_chat = ["qwen3.8-flash", "deepseek-v4-flash-0731"]
    supported_models_embed = ["qwen3.7-text-embedding"]

    def __init__(self, config: ProviderConfig):
        if not config.api_key:
            raise ProviderInitializationError("qwen API Key 未设置，请检查api")
        self.config =  config

    def get_client(self, model_name: str) -> BaseChatModel | DashScopeEmbeddings:
        """ 获取模型客户端。"""
        if model_name not in self.supported_models_chat and model_name not in self.supported_models_embed:
            raise ModelNotSupportedError(f"模型 {model_name} 不是通义模型, 仅支持模型: {self.supported_models_chat}, {self.supported_models_embed}")
        if model_name in self.supported_models_chat:
            log.info(f" [LLM] 使用云端供应商 聊天模型: {model_name}")
            return ChatOpenAI(
                model=model_name,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                temperature=self.config.temperature,
                reasoning_effort=settings.llm_reasoning_effort,
                extra_body={"enable_thinking": settings.llm_enable_thinking},
                max_retries=1,
            )
        else:
            log.info(f" [LLM] 使用云端供应商 文本嵌入模型: {model_name}")
            return DashScopeEmbeddings(
                model=model_name,
                dashscope_api_key=self.config.api_key,
                max_retries=1,
            )


class OllamaProvider:
    """ Ollama供应商类, 创建客户端."""
    supported_models_chat = ["qwen3.5:9b"]
    vision_models = ["qwen3-vl:2b", "gemma3:4b"]
    supported_models_embed = ["qwen3-embedding:8b"]

    def __init__(self, config: ProviderConfig):
        self.config = config

    def get_client(self, model_name: str) -> BaseChatModel | OllamaEmbeddings:
        """ 获取模型客户端。"""
        if (model_name not in self.supported_models_chat
                and model_name not in self.supported_models_embed
                and model_name not in self.vision_models):
            raise ModelNotSupportedError(
                f"模型 {model_name} 不是本地模型, 仅支持模型: {self.supported_models_chat}, {self.supported_models_embed}")
        if model_name in self.supported_models_chat:
            log.info(f" [LLM] 创建 Ollama 客户端, 使用本地模型: {model_name}")
            return ChatOllama(
                model=model_name,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                temperature=self.config.temperature,
                reasoning=False,

            )
        elif model_name in self.supported_models_embed:
            log.info(f" [LLM] 创建 Ollama 嵌入模型客户端, 使用本地模型: {model_name}")
            return OllamaEmbeddings(
                model=model_name,
                base_url=self.config.base_url,
                client_kwargs={"timeout": self.config.timeout}
            )
        else:
            log.info(f" [LLM] 创建 Ollama 视觉模型客户端, 使用本地模型: {model_name}")
            return OpenAI(
                api_key="ollama",
                base_url=f"{self.coning.base_url.rstrip('/')}/v1",
                timeout=self.coning.timeout,
            )


# ========== 工厂核心 ================

class LLMFactory:
    """
        轻量级工厂：
        - 根据模型名称自动路由到对应供应商
        - 配置外部化，支持环境变量
        - 无缓存、无重试、无复杂观测（这些由上层或网关处理）
    """
    def __init__(self):
        self._providers: dict[str, tuple] = {
            "qwen":(load_qwen_config, QWENProvider),
            "ollama":(load_ollama_config, OllamaProvider),
            "glm": (load_glm_config, GLMNProvider),
        }
        self._routing: dict[str, str] = {}
        self._client_cache: dict[str, Any] = {}
        self._model_types: dict[str, str] = {}
        self._build_routing()


    def _build_routing(self):
        """从每个供应商的 supported_models 构建路由表"""
        self._routing.clear()
        self._model_types.clear()
        for provider_name, (_, provider_class) in self._providers.items():
            for model_name in getattr(provider_class, "supported_models_chat", []):
                self._routing[model_name] = provider_name
                self._model_types[model_name] = "chat"
            for model_name in getattr(provider_class, "supported_models_embed", []):
                self._routing[model_name] = provider_name
                self._model_types[model_name] = "embed"
            for model_name in getattr(provider_class, "vision_models", []):
                self._routing[model_name] = provider_name
                self._model_types[model_name] = "vision"
        log.info(f" [LLM] 工厂路由表构建完成，共 {len(self._routing)} 个模型")


    def get_client(self, model_name: str) -> BaseChatModel | Embeddings:
        """ 根据模型名称获取对应的提供商和客户端（裸客户端，弹性由 agent 中间件负责）"""
        if model_name in self._client_cache:
            return self._client_cache[model_name]

        provider_name = self._routing.get(model_name)
        if not provider_name:
            raise ModelNotSupportedError(f"模型 {model_name} 不支持，支持列表为:{list(self._routing)}")

        try:
            (config_loader, provider_cls) = self._providers[provider_name]
            config = config_loader()
            provider = provider_cls(config)
            client = provider.get_client(model_name)
            self._client_cache[model_name] = client
            return client
        except ProviderInitializationError as e:
            log.error(f" [LLM] 提供商初始化失败: {e}")
            raise
        except Exception as e:
            log.error(f" [LLM] 创建客户端失败: {e}")
            raise ProviderInitializationError(f"无法创建 {model_name} 客户端: {e}")


    # 工具函数
    def get_supported_models(self, model_types: list[str] | None = None) -> list[str]:
        """ 获取支持的模型列表
        :param model_types: 按类型过滤，如 ["chat"] 或 ["chat","vision"]；不传返回全部
        """
        if model_types is None:
            return list(self._routing.keys())
        return [m for m, t in self._model_types.items() if t in model_types]



# ========== 全局实例对象 ==========
llm_factory = LLMFactory()
