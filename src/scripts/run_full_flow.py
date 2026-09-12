""" 多智能体流程测试 """

import sys
from pathlib import Path
import asyncio

from src.doc_agent.document_agent import doc_agent
from src.mcp.client import close

BASE_DIR = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(BASE_DIR))

from src.memory.memory import CommonMemory
from src.supervisor_agent.supervisor_agent import SupervisorAgent



test_memory = CommonMemory(session_id="test_session_id_001")
si_agent = SupervisorAgent(memory=test_memory)


async def run_test():

    while True:
        question = input("请输出内容")
        if question == "end" :
            break
        result_parts = []
        async for item in si_agent.astream(question):
            if isinstance(item, tuple) and item[0] == "content":
                result_parts.append(item[1])
        result = "".join(result_parts)
        print("===== 完整链路输出结果 =====")
        print(result)

async def main():
    await doc_agent.get_agent()
    print("doc agent 创建成功")
    await close()  # 关闭mcp进程




if __name__ == "__main__":
    asyncio.run( run_test() )
    # asyncio.run(main())