"""记忆板块"""

import os, yaml
from datetime import datetime
from typing import List, Dict
from src.utils.logger import log
from src.config.settings import settings
from sqlalchemy import func
from sqlmodel import SQLModel, Field, create_engine, Session, select, delete


# 单条消息最大token数，默认4k
max_tokens = 4000

# ========== 数据库全局配置 ==========

# 当前使用的数据库类型，可选 mysql / sqlite
db_type = "sqlite"
# 存储数据库文件路径
sqlite_file =  settings.memory_sqlite_file
db_dir = os.path.dirname(sqlite_file)
os.makedirs(db_dir, exist_ok=True)




CONFIG_YML_PATH = "mysql_config.yml"
def load_mysql_config():
    """读取mysql_config.yml配置文件"""
    if not os.path.exists(CONFIG_YML_PATH):
        raise FileNotFoundError(f"找不到数据库配置文件：{CONFIG_YML_PATH}")
    with open(CONFIG_YML_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("mysql", {})



# 判断数据库类型, 切换数据库
if db_type == "sqlite":
    global_engine = create_engine(f"sqlite:///{sqlite_file}", echo=False)
elif db_type == "mysql":
    mysql_cfg = load_mysql_config()
    host = mysql_cfg["host"]
    port = mysql_cfg["port"]
    user = mysql_cfg["user"]
    pwd = mysql_cfg["password"]
    db_name = mysql_cfg["database"]
    MYSQL_URL = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db_name}"
    global_engine = create_engine(MYSQL_URL, echo=False)
else:
    raise Exception("数据库类型错误")

# ===================== Token 计数工具 =====================

tiktoken = None
_tokenizer = None
try:
    import tiktoken
    _tokenizer = tiktoken.get_encoding("cl100k_base")
    log.info("[memory] 使用tiktoken分词器")
except ImportError:
    log.warning("[memory] 未安装tiktoken，将使用简易token估算")
def count_tokens(text: str) -> int:
    if tiktoken and _tokenizer:
        return len(_tokenizer.encode(str(text)))
    return len(text) // 4



# ========== 数据库表模型（切换mysql/sqlite无需修改此类） ==========
class ChatRecord(SQLModel, table=True):
    """
    会话记录表, 内置表头设置
    """
    __tablename__ = "agent_chat_memory"
    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(default="default_main")
    role: str
    content: str
    single_token: int = Field(default=0)
    create_time: datetime = Field(default_factory=datetime.now)

# ========== 统一对话内存管理类 =========
class CommonMemory:
    def __init__(
        self,
        session_id: str = "default_main",
        max_context_tokens: int |  None = None,
        engine = None
    ):
        """
        初始化
        :param session_id: 当前会话id，默认通用会话
        :param max_context_tokens: 单条消息最大token数，默认4000
        :param engine: 数据库引擎
        """
        self.session_id = session_id
        self.engine = engine or global_engine
        if max_context_tokens is None:
            self.max_context_tokens = max_tokens
        else:
            self.max_context_tokens = max_context_tokens
        SQLModel.metadata.create_all(global_engine)

    def add(self, role: str, content: str):
        """
        数据入库
        :param role: 角色：human / ai
        :param content: 内容
        """
        allow_roles = {"human", "ai", "tool"}
        if role not in allow_roles:
            log.warning(f"[memory] 非法角色{role}，跳过入库")
            return
        token_num = count_tokens(content)
        try:
            with Session(self.engine) as session:
                session.add(ChatRecord(
                    session_id = self.session_id,  # 当前实例绑定的会话ID
                    role = role,  # 消息角色（user/assistant）
                    content = content,  # 对话文本内容
                    single_token = token_num  # 本条消息的token数量
                ))
                session.commit()
                log.info(f"[memory]写入记忆成功：{content}")
        except Exception as e:
            log.error(f"[memory]写入记忆失败：{e}")


    def get_recent(self) -> List[Dict[str, str]]:
        """获取对话记录"""
        with Session(self.engine) as session:
            try:
                all_rows = session.exec(
                    select(ChatRecord)
                    .where(ChatRecord.session_id == self.session_id)
                    .order_by(ChatRecord.create_time.desc())
                    .limit(50)

                ).all()   # 获取所有查询结果
            except Exception as e:
                log.error(f"[memory]获取记忆失败：{e}")
                return []

            total_tok = 0
            queue = []

            for item in reversed(all_rows):
                if item.single_token > self.max_context_tokens:
                    continue
                queue.append(item)
                total_tok += item.single_token

                while total_tok > self.max_context_tokens and len(queue) > 0:
                    removed_item = queue.pop(0)
                    total_tok -= removed_item.single_token
            result = [{"role": i.role, "content": i.content} for i in queue]
            log.info(f"[memory]获取记忆成功：{result}")
            return result

    def delete_session(self):
        """清空当前会话, 谨慎使用"""
        try:
            with Session(self.engine) as session:
                exist_row = session.exec(
                    select(ChatRecord)
                    .where(ChatRecord.session_id == self.session_id)
                ).first()

                if exist_row is None:
                    msg = f"会话{self.session_id}无记录，无需执行删除"
                    log.info(f"[memory]:{msg}")
                    return msg

                session.exec(
                    delete(ChatRecord)
                    .where(ChatRecord.session_id == self.session_id)
                )
                session.commit()
                msg = f"会话{self.session_id}全部记忆删除成功"
                log.info(f"[memory]:{msg}")
                return msg
        except Exception as e:
            err_msg = f"删除会话异常失败：{str(e)}"
            log.error(f"[memory]:{err_msg}")
            return err_msg


    def trim_redundant(self, reserve_num: int = 100):
        """手动执行：当前会话只保留最新reserve_num条，老旧数据全部删除, """
        try:
            with Session(self.engine) as session:
                keep_id_list = session.exec(
                    select(ChatRecord.id)
                    .where(ChatRecord.session_id == self.session_id)
                    .order_by(ChatRecord.create_time.desc())
                    .limit(reserve_num)
                ).scalars().all()

                if not keep_id_list:
                    log.info(f"[memory] 当前会话无聊天记录，无需清理")
                    return

                session.exec(
                    delete(ChatRecord)
                    .where(ChatRecord.session_id == self.session_id)
                    .where(ChatRecord.id.notin_(keep_id_list))
                )
                session.commit()
                log.info(f"[memory]冗余记录清理完成，保留最新{reserve_num}条")

        except Exception as e:
            log.error(f"[memory]删除多余数据失败：{e}")

    def get_session_record_count(self) -> int:
        """获取当前会话的聊天记录条数"""
        try:
            with Session(self.engine) as session:
                stmt = session.exec(
                    select(func.count(ChatRecord.id))
                    .where(ChatRecord.session_id == self.session_id)
                ).scalar()
                return stmt or 0
        except Exception as e:
            log.error(f"[memory]获取会话记录条数失败：{e}")
            return 0



if __name__ == '__main__':
    # 测试删除id为0001的会话
    memory = CommonMemory("0001")
    res = memory.delete_session()

    print(res)
