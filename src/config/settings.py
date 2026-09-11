from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

# 拿到项目根目录：src/config 往上两层
BASE_DIR = Path(__file__).parent.parent.parent

class AppSettings(BaseSettings):
    # 读取规则
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8"
    )

    # LLM 思考强度：None=不传该参数（响应最快）；按需可填 "low"/"medium"/"high"
    # 注意：实测该参数会让模型把 token 大量消耗在内部思考上，导致响应显著变慢甚至超时
    llm_reasoning_effort: Optional[str] = None

    # 是否允许模型内部思考：False=关闭（响应快、稳定，避免"只思考不输出"）；True=开启
    # 仅对 dashscope(通义/deepseek) 通道生效，通过 extra_body 的 enable_thinking 传参
    llm_enable_thinking: bool = False

    #  glm 智普 __
    glm_api_key: SecretStr = SecretStr("")
    glm_base_url: str ="https://open.bigmodel.cn/api/paas/v4/"
    glm_timeout: int = 90
    glm_temperature: float = 0.7



    #  qwen 通义千问 __
    qwen_api_key: SecretStr = SecretStr("")
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # 单次模型请求超时（秒）：长文本生成需要 30~60 秒
    qwen_timeout: int = 90
    qwen_temperature: float = 0.7

    # ollama本地 __
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 120
    ollama_temperature: float = 0.7



    # 记忆模块配置 __
    # 存储数据库文件路径
    memory_sqlite_file: str = str(BASE_DIR / "src/db/agent_memory_db")

    # 日志模块 __
    log_save_dir: str = str(BASE_DIR / "logs")  # 存储路径


    # 通用agent __
    agent_default_model: Optional[str] = "qwen3.8-flash"  # 子agent使用模型
    agent_debug_mode: bool = True  # 调试模式
    agent_security_check: bool = False  # 提示词注入安全检测开关（默认关闭False）

    # 子 Agent 单次推理总超时（秒）：防止模型调用卡死时无限等待
    subagent_timeout: int = 180

    supervisor_agent_model: Optional[str] = "qwen3.8-flash"  # 主Agent使用模型：云端

    # 本地 mcp配置__
    mcp_server_path: str = ""   # 不写死, 可设置根据需求设置路径





settings = AppSettings()