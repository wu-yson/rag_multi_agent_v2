"""验证熔断包装接进工厂后：真实调用 + create_agent 兼容"""
import asyncio

from langchain.agents import create_agent

from src.llm.factory import llm_factory


async def main():
    llm = llm_factory.get_client("qwen3.7-flash")

    # ② 真调用走包装类
    resp = await llm.ainvoke("ping，请只回复ok")
    print("直接调用回复:", resp.content)

    # ③ create_agent 认不认这个包装
    agent = create_agent(model=llm, system_prompt="你是助手", tools=[])
    agent_resp = await agent.ainvoke({"messages": [("human", "你好")]})
    print("agent回复:", agent_resp["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())