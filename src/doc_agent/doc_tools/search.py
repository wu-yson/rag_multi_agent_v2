

from pathlib import Path

from src.doc_agent.doc_tools._helpers import sandbox_root, ensure_project_path, generate_agent_tools
from src.doc_agent.doc_tools.registry import tool_registry

# 单个搜索文件的最大大小（10MB），超过则跳过
_MAX_SEARCH_FILE_SIZE = 10485760



class SearchTextTool:
    """文本检索工具 """

    def __init__(self, custom_sandbox: Path | None = None):
        if custom_sandbox is not None:
            self.sandbox_root = custom_sandbox.resolve()
            self.sandbox_root.mkdir(parents=True, exist_ok=True)
        else:
            self.sandbox_root = sandbox_root


    # ---------------------- 公共抽离重复路径校验逻辑（和另外两套工具风格统一） ----------------------

    def _safe_search_root(self, raw_root: str) -> Path | str:
        """统一校验搜索根目录是否合法，非法直接返回错误文本"""
        try:
            root = ensure_project_path(raw_root, self.sandbox_root)
        except ValueError as e:
            return f"[ERROR] {e}"

        if not root.exists():
            return f"[ERROR] 搜索目录不存在: {raw_root}"
        if not root.is_dir():
            return f"[ERROR] 搜索目录不是目录: {raw_root}"
        return root

    # ---------------------- 对外业务工具（仅保留独有检索逻辑） ----------------------

    def search_text(
        self,
        pattern: str,
        root_dir: str = ".",
        file_glob: str = "*",
        max_matches: int = 50,
        encoding: str = "utf-8",
    ) -> str:
        """在目录下递归搜索文本，返回匹配的文件和行号。
            Args:
                pattern: 要搜索的文本片段。
                root_dir: 搜索根目录，默认当前目录。
                file_glob: 文件匹配模式，如 ``*.py``。
                max_matches: 最多返回的匹配条数。
                encoding: 文本读取编码，默认 utf-8。
            """

        root = self._safe_search_root(root_dir)

        if isinstance(root, str):
            return root
        if max_matches <=0:
            return "[ERROR] 最大匹配数必须大于0"
        if not pattern:
            return "[ERROR] 搜索文本不能为空"

        matches: list[str] = []
        for path in sorted(root.rglob(file_glob)):
            if not path.is_file():
                continue

            try:
                file_size = path.stat().st_size
                if file_size > _MAX_SEARCH_FILE_SIZE:
                    continue
                rel_path_str = str(path.relative_to(self.sandbox_root))

                with path.open("r", encoding=encoding) as f:
                    for line_no, line in enumerate(f, start=1):
                        if pattern in line:
                            matches.append(f"{rel_path_str}:{line_no}:{line.rstrip()}")
                            if len(matches) >= max_matches:
                                return "\n".join( matches)
            except UnicodeDecodeError:
                continue
        if not matches:
            return f"[INFO] 未找到匹配内容：{pattern}"
        return "\n".join(matches)



# 全局唯一实例
search_text_tool = SearchTextTool()
# 注册工具列表
tools_list = generate_agent_tools(search_text_tool, skip_names=["generate_tool_list"])

# 注册工具
tool_registry.register_many(tools_list)