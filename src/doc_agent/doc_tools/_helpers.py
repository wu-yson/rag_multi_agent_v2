"""工具模块内部共享辅助函数。"""


from pathlib import Path
from typing import List
from langchain_core.tools import BaseTool, tool

project_root = Path(r"D:\python-xuexi\测试读写文件")
project_root.mkdir(parents=True, exist_ok=True)
sandbox_root = project_root


def resolve_project_path(raw_path: str, sandbox_root: Path) -> Path:
    """将原始路径转为为绝对路径。"""
    path = Path(raw_path)
    if not path.is_absolute():
        path = sandbox_root / path
    return  path.resolve()

def is_within_project(path: Path,  sandbox_root: Path) -> bool:
    """判断给定路径是否在项目根目录内。"""
    try:
        path.relative_to(sandbox_root)
        return True
    except ValueError:
        return False

def ensure_project_path(raw_path: str,  sandbox_root: Path) -> Path:
    """解析路径并确保其在项目根目录内，否则抛出 ValueError。"""
    path = resolve_project_path(raw_path, sandbox_root)
    if not is_within_project(path, sandbox_root):
        raise ValueError(f"路径超出项目根目录：{raw_path}")
    return path

def to_project_relative(path: Path, sandbox_root: Path = sandbox_root) -> str:
    """将绝对路径转换为相对于项目根目录的路径字符串。"""

    try:
        return str(path.relative_to(sandbox_root))
    except ValueError:
        return f"【外部路径，不在项目目录内】{str(path)}"

# ========== 批量扫描生成标准工具列表核心方法（固定模板不用改） ==========
def generate_agent_tools(toolkit_instance: object, skip_names: List[str] = None) -> List[BaseTool]:
    """
    通用工具扫描生成器，所有IO工具类共用
    :param toolkit_instance: 工具类实例
    :param skip_names: 额外需要跳过的方法名，默认跳过 __init__
    """
    skip_list = ["__init__", "generate_tool_list"]
    if skip_names:
        skip_list.extend(skip_names)

    tool_list: List[BaseTool] = []
    for attr_name in dir(toolkit_instance):
        attr = getattr(toolkit_instance, attr_name)
        if callable(attr) and not attr_name.startswith("_") and attr_name not in skip_list:
            decorator = tool
            tool_obj = decorator(attr)
            tool_obj.name = attr_name
            tool_obj.description = attr.__doc__.strip() if attr.__doc__ else ""
            tool_list.append(tool_obj)
    return tool_list
