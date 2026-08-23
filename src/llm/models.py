from dataclasses import dataclass
from typing import Optional



# ========== 自定义异常定义 ==========
class ModelNotSupportedError(Exception):
    """模型不存在或未配置"""
    pass

class ProviderInitializationError(Exception):
    """供应商初始化失败（如 API Key 缺失）"""
    pass




# ==========  供应商配置（通用） ==========
@dataclass
class ProviderConfig:
    """ 通用配置实体，定义字段默认值（仅外部不传参时生效）。"""
    api_key: Optional[str]
    base_url: Optional[str]
    timeout: int
    temperature: float
