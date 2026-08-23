"""项目内路径与目录工具。
全部路径强制校验，仅允许访问项目内部目录
"""

import json
from pathlib import Path

from src.doc_agent.doc_tools._helpers import sandbox_root, ensure_project_path, to_project_relative, \
    generate_agent_tools
from src.doc_agent.doc_tools.registry import tool_registry


class PathsTool:
    """项目内路径与目录工具。
    全部路径强制校验，仅允许访问项目内部目录
    """

    def __init__(self, custom_sandbox: Path | None = None):
        # 不传参自动读取本地默认路径，兼容原有写法
        if custom_sandbox is not None:
            self.sandbox_root = custom_sandbox.resolve()
            self.sandbox_root.mkdir(parents=True, exist_ok=True)
        else:
            self.sandbox_root = sandbox_root



    def _safe_target(self, raw_path: str) -> Path |str:
        """统一路径安全校验，非法路径直接返回错误字符串"""
        try:
            return ensure_project_path(raw_path, self.sandbox_root)
        except ValueError as e:
            return f"[ERROR] {e}"

    def _dump_json(self, data: dict) -> str:
        """统一json序列化输出，消除重复参数"""
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _walk_with_depth(self, directory: Path, max_depth: int):
        """递归遍历目录，限制最大深度（独有递归逻辑保留）"""

        def _walk(current: Path, depth: int):
            if depth > max_depth:
                return
            try:
                for item in sorted(current.iterdir()):
                    yield item

                    if item.is_dir():
                        yield from _walk(item, depth + 1)
            except PermissionError:
                pass

        yield from _walk(directory, 0)

    # ---------------------- 对外业务工具 ----------------------

    def list_dir(
        self,
        path: str = ".",  # 默认当前目录 "."
        recursive: bool = False,  # 是否递归, 默认不递归
        include_hidden: bool = False,   # 是否包含隐藏文件, 默认不包含
        max_depth: int = 5,  # 递归最大深度, 默认5级
    ) -> str:
        """列出项目内目录内容。"""
        target = self._safe_target(path)
        if isinstance(target, str):
            return target
        if not target.exists():
            return f"[ERROR] 路径不存在: {path}"
        if not target.is_dir():
            return f"[ERROR] 不是目录: {path}"

        iterator = self._walk_with_depth(target, max_depth) if recursive else target.iterdir()

        entries = []
        for item in sorted(iterator) if not recursive else iterator:
            if not include_hidden and item.name.startswith("."):
                continue

            entries.append({
                "name": item.name,
                "type": "dir" if item.is_dir() else "file",
                "path": to_project_relative(item),
                "abs_path": str(item)  # 完整物理绝对路径，永久可溯源，人工排错唯一依据
            })
        return self._dump_json({
            "ok": True,
            "query_rel_dir": to_project_relative(target),
            "query_abs_dir": str(target),
            "entries": entries,
        })

    def make_dir(self, path: str, parents: bool = True) -> str:
        """在项目根目录内创建目录。"""
        target = self._safe_target(path)
        if isinstance(target, str):
            return target

        try:
            target.mkdir(parents=parents, exist_ok=True)
        except OSError as e:
            return f"[ERROR] 创建目录失败：{e}"

        return self._dump_json({
            "ok": True,
            "created": True,
            "path": to_project_relative(target),
            "abs_path": str(target),
        })


    def glob_files(self, pattern: str, root_dir: str = ".") -> str:
        """按 glob 模式在项目内查找文件。"""

        root = self._safe_target(root_dir)
        if isinstance(root, str):
            return root
        if not root.exists():
            return f"[ERROR] 路径不存在：{root_dir}"
        if not root.is_dir():
            return f"[ERROR] 路径不是目录：{root_dir}"
        if not pattern:
            return "[ERROR] pattern 不能为空"

        matches = [
            to_project_relative(path)
            for path in sorted(root.rglob(pattern))
            if path.is_file()
        ]
        return self._dump_json({
            "ok": True,
            "root_dir": to_project_relative(root),
            "root_abs": str(root),
            "pattern": pattern,
            "matches": matches,
        })



# 全局唯一实例
path_tools = PathsTool()
# 注册工具列表
tools_list = generate_agent_tools(path_tools, skip_names=["generate_tool_list"])

# 注册工具
tool_registry.register_many(tools_list)