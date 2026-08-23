"""仅单独测试大模型连通性，不启动agent、不启动graph"""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.llm.factory import llm_factory

if __name__ == "__main__":
    llm = llm_factory.get_client("qwen3.7-flash")
    resp = llm.invoke("ping，请只回复ok")
    print(f"llm返回：{resp.content}")
