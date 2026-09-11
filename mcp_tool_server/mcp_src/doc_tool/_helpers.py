"""工具模块内部共享辅助函数。"""


from pathlib import Path

# 默认路径
project_root = Path(__file__).resolve().parents[2] / "workspace"
project_root.mkdir(parents=True, exist_ok=True)
sandbox_root = project_root


# ===== 危险路径黑名单（前缀匹配，命中即拒绝） =====
BLOCKED_PATH_PREFIXES = [
    r"c:\windows", r"c:/windows",
    r"c:\windows.old", r"c:/windows.old",
    r"c:\program files", r"c:/program files",
    r"c:\program files (x86)", r"c:/program files (x86)",
    r"c:\programdata", r"c:/programdata",
    r"c:\$recycle.bin", r"c:/$recycle.bin",
    r"c:\system volume information", r"c:/system volume information",
    r"/etc", r"/usr", r"/bin", r"/sbin",
    r"/proc", r"/dev", r"/boot", r"/sys", r"/root",
]

def is_blocked_path(path: Path) -> bool:
    """判断路径是否命中危险黑名单（不区分大小写）"""
    p = str(path.resolve()).lower()
    return any(p.startswith(prefix.lower()) for prefix in BLOCKED_PATH_PREFIXES)


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
    """解析路径并返回绝对路径：绝对路径直接放行（信任传入路径），相对路径拼到沙箱根下。"""
    path = resolve_project_path(raw_path, sandbox_root)
    if is_blocked_path(path):
        raise ValueError(f"路径被安全策略拦截：{raw_path}")
    return path

def to_project_relative(path: Path, sandbox_root: Path = sandbox_root) -> str:
    """将绝对路径转换为相对于项目根目录的路径字符串。"""

    try:
        return str(path.relative_to(sandbox_root))
    except ValueError:
        return f"【外部路径，不在项目目录内】{str(path)}"


