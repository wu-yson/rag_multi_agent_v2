""" 多智能体流程测试 """

import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(BASE_DIR))

from src.memory.memory import CommonMemory
from src.supervisor_agent.supervisor_agent import SupervisorAgent



test_memory = CommonMemory(session_id="test_session_id_001")
si_agent = SupervisorAgent(memory=test_memory)


def run_test():

    while True:
        question = input("请输出内容")
        if question == "end" :
            break
        result = si_agent.invoke(question)
        print("===== 完整链路输出结果 =====")
        print(result)


if __name__ == "__main__":
    run_test()
