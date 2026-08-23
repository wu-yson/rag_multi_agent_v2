from langchain_core.tools import BaseTool, tool
from typing import Optional, Self, List
from dataclasses import dataclass

# 精简：只保留一种工具类型，不需要多分类枚举
@dataclass
class SimpleToolRecord:
    tool: BaseTool


class LiteToolRegistry:
    """
    工具仓库
    """
    _instance: Optional[Self] = None
    _tools: dict[str, SimpleToolRecord]

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    def register(self, tool_obj: BaseTool) -> bool:
        """手动注册单个工具，同名覆盖跳过"""
        name = tool_obj.name
        if name in self._tools:
            return False
        self._tools[name] = SimpleToolRecord(tool=tool_obj)
        return True

    def register_many(self, tool_list: List[BaseTool]) -> int:
        """批量注册多个工具，同名覆盖跳过"""
        cnt = 0
        for t in tool_list:
            if self.register(t):
                cnt += 1
        return cnt

    def get(self, name:  str) -> BaseTool:
        """按名字取工具，不存在抛异常"""
        if name not in self._tools:
            raise KeyError(f"工具 {name} 未注册")
        return self._tools[name].tool

    def get_all(self) -> List[BaseTool]:
        """获取全部已注册工具"""
        return [item.tool for item in self._tools.values()]

    def resolve_by_names(self, name_list: List[str]) -> List[BaseTool]:
        """白名单筛选指定工具，Agent配置tool_names专用"""
        return [self.get(name) for name in name_list]

    def has_tool(self, name: str) -> bool:
        """判断工具是否存在"""
        return name in self._tools

    def __len__(self):
        # 获取已注册工具数量
        return len(self._tools)

# 全局唯一实例
tool_registry = LiteToolRegistry()




