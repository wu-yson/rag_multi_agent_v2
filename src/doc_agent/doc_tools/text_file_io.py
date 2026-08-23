
import re
import threading
from pathlib import Path

from src.doc_agent.doc_tools._helpers import sandbox_root, ensure_project_path, generate_agent_tools
from src.doc_agent.doc_tools.registry import tool_registry


class TextFileIOToolkit:
    """文本操作工具集
    覆盖写入/追加/精确替换/正则替换/全文读取/按行区间读取
    强制路径白名单，仅允许项目内文件操作
    """
    # 新增全局文件互斥锁
    _file_op_lock = threading.Lock()

    def __init__(self, custom_sandbox: Path | None = None):
        # 不传参自动读取本地默认路径，兼容原有写法
        if custom_sandbox is not None:
            self.sandbox_root = custom_sandbox.resolve()
            self.sandbox_root.mkdir(parents=True, exist_ok=True)
        else:
            self.sandbox_root = sandbox_root

    # -------------------------- 公共抽离重复逻辑，消除大量冗余 --------------------------

    def _get_safe_path(self, file_path: str) -> Path | str:
        """统一路径校验，越权直接返回错误文本，否则返回合法Path对象"""
        try:
            return ensure_project_path(file_path, self.sandbox_root)
        except ValueError as e:
            return f"文件路径非法: {e}"

    def _read_text_safe(self,path: Path, encoding: str = "utf-8") -> str:
        """安全读取文本，统一捕获所有读异常"""
        try:
            # 读取文件
            return path.read_text(encoding=encoding)
        except FileExistsError:
            return f"[ERROR]文件不存在 "
        except PermissionError:
            return f"[ERROR]无读取权限 "
        except UnicodeDecodeError:
            return f"[ERROR]文件编码错误, 请使用 {encoding} 编码 "
        except Exception as e:
            return f"[ERROR]文件读取异常: {e}"

    def _write_text_safe(self, path: Path, content: str, encoding: str = "utf-8", append: bool = False) -> str:
        """安全写入/追加写入"""
        path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"
        content = content.rstrip()

        with self._file_op_lock:
            # ============ 重复内容拦截逻辑 ============
            if append:
                # 追加模式：只检查文件末尾6行，防止重复追加整块相同文本
                if path.exists():
                    with open(path, "r", encoding=encoding) as f:
                        tail_lines = f.readlines()[-6:]
                    tail_text = "".join(tail_lines).rstrip()
                    if content in tail_text:
                        return f"[INFO]尾部已存在相同段落，跳过追加: {path}"
            else:
                # 覆盖模式：全文对比，无变化直接跳过
                if path.exists():
                    old_content = path.read_text(encoding=encoding).rstrip()
                    if old_content == content:
                        return f"[INFO]文件内容无变更，跳过覆盖写入: {path}"

        try:
            with path.open(mode, encoding=encoding) as f:
                f.write(content)
            op = "追加" if append else "写入"
            return f"成功{op}文件: {path} ({len(content)})"
        except PermissionError:
            return f"[ERROR]无写入权限 "
        except Exception as e:
            return f"[ERROR]文件写入异常: {e}"

    # -------------------------- 业务工具（仅核心逻辑，无重复模板代码） --------------------------

    def write_file(self, file_path: str, content: str, encoding: str = "utf-8") -> str:
        """覆盖写入文件，不存在则自动创建父目录与文件"""
        # 获取到传入路径file_path, 统一校验
        path = self._get_safe_path(file_path)
        if isinstance(path, str):
            return path
        # 调用写入方法
        else:
            return self._write_text_safe(path, content, encoding, append=False)


    def append_file(self, file_path: str, content: str, encoding: str = "utf-8") -> str:
        """追加写入文件，不存在则自动创建父目录与文件"""
        path = self._get_safe_path(file_path)
        if isinstance(path, str):
            return path
        else:
            return self._write_text_safe(path, content, encoding, append=True)


    def replace_exact(self, file_path: str, old: str, new: str, encoding: str = "utf-8") -> str:
        """全文精确字符串批量替换, 传入的旧字符串内容old需要和旧文本内容一致才能完成替换, 多了空格都不算"""

        path = self._get_safe_path(file_path)
        if isinstance(path, str):
            return path
        content = self._read_text_safe(path, encoding)
        if content.startswith("[ERROR]"):
            return f"{content} : {file_path}"
        cnt = content.count(old)

        if cnt == 0:
            return f"未找到匹配项: {file_path}"
        new_txt = content.replace(old, new)
        res = self._write_text_safe(path, new_txt, encoding)
        return f" {res} ({cnt} 处替换) "


    def replace_regex(self, file_path: str, pattern: str, replacement: str, encoding: str = "utf-8") -> str:
        """正则表达式批量替换, 允许模糊,范围匹配修改替换"""
        path = self._get_safe_path(file_path)
        if isinstance(path, str):
            return path
        content = self._read_text_safe(path, encoding)
        if content.startswith("[ERROR]"):
            return f"{content} : {file_path}"

        try:
            reg = re.compile(pattern)
        except re.error as e:
            return f"[ERROR]正则表达式错误: {e}"
        new_txt, cnt = reg.subn(replacement, content)
        if cnt == 0:
            return f"[INFO] 未找到正则匹配，未做修改：{file_path}"
        if new_txt == content:
            return f"[INFO]替换后内容无变化，不执行写入: {file_path}"
        res = self._write_text_safe(path, new_txt, encoding)
        return f"{res} ({cnt} 处替换)"


    def read_file(self, file_path: str, encoding: str = "utf-8") -> str:
        """读取文件全部文本内容"""

        path = self._get_safe_path(file_path)
        if isinstance(path, str):
            return path
        if not path.exists():
            return f"[ERROR]文件不存在: {file_path}"
        if not path.is_file():
            return f"[ERROR]不是文件: {file_path}"
        return self._read_text_safe(path, encoding)


    def read_lines(self, file_path: str, start: int = 1, end: int = 0, encoding: str = "utf-8") -> str:
        """按行号区间读取文件，行号从1开始"""

        path = self._get_safe_path(file_path)
        if isinstance(path, str):
            return path
        if not path.exists():
            return f"[ERROR]文件不存在: {file_path}"
        if not path.is_file():
            return f"[ERROR]不是文件: {file_path}"

        try:
            with path.open("r", encoding=encoding) as f:
                lines = f.readlines()
        except Exception as e:
            return f"[ERROR]文件读取异常: {e}"

        total = len(lines)  # 获取文件行数
        end_idx = total if end <= 0 else min(end, total)
        start_idx = max(1, start)
        if start_idx > end_idx:
            return f"[INFO] 起始行 {start_idx} 超出文件总行数（{total} 行）"
        slice_lines = lines[start_idx - 1:end_idx]
        return "".join(f"{i + start_idx:>4} | {line}" for i, line in enumerate(slice_lines))





# 全局唯一实例
text_tools = TextFileIOToolkit()
# 注册工具列表
tools_list = generate_agent_tools(text_tools, skip_names=["generate_tool_list"])

# 注册工具
tool_registry.register_many(tools_list)

