"""
src/utils/logger.py
全局唯一日志实例，通用 + Agent场景合一，无多实例混乱
调用永远只用 log.xxx()，区分场景靠传入标签字符串
"""
import logging
import os
from datetime import datetime
from typing import Optional

from src.config.settings import settings

# 标准日志格式：时间 | 日志名 | 级别 | 代码文件:行号 | 内容
LOG_FORMAT = logging.Formatter(
    "%(asctime)s | %(name)-12s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s"
)

class GlobalLogger:
    """ 全局日志, 也可用于多日志情况 """
    _logger: logging.Logger
    def __init__(
        self,
        logger_name: Optional[str] = None,
        log_dir: Optional[str] = None
    ):
        # 默认日志名称
        self._logger_name = logger_name if logger_name else "global_agent_log"
        # 默认存储目录
        self._log_dir = log_dir if log_dir else settings.log_save_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._logger = logging.getLogger(self._logger_name)
        self._logger.setLevel(logging.DEBUG)

        # 防止重复挂载handler，多次实例化只初始化一次
        if not self._logger.handlers:
            self._init_console_handler()
            self._init_file_handler()

    def _init_console_handler(self):
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(LOG_FORMAT)
        console_handler.setLevel(logging.INFO)
        self._logger.addHandler(console_handler)

    def _init_file_handler(self):
        # 每次运行生成独立时间戳日志文件，互不覆盖
        time_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self._log_dir, f"{self._logger_name}_{time_suffix}.log")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(LOG_FORMAT)
        file_handler.setLevel(logging.DEBUG)
        self._logger.addHandler(file_handler)

    # 底层原生四级日志，全局统一调用入口
    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

# 全局单例，项目所有文件只导入这一个
log = GlobalLogger()

if __name__ == '__main__':
    log.debug("调试信息，控制台不显示，只写入文件")
    log.info("正常流程日志")
    log.warning("警告信息")
    log.error("错误日志")