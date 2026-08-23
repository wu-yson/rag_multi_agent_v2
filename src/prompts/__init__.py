from src.prompts.prompt_injection_detect_prompt import prompt_injection_detect_prompt
from src.prompts.rag import rag_agent_prompt
from src.prompts.doc import doc_agent_prompt
from src.prompts.graph import GRAPH_PROMPT
from src.prompts.supervisor import supervisor_agent_prompt

# 统一池子，key作为调用标识
PROMPT_LIBRARY = {
    "rag_agent_prompt": rag_agent_prompt,
    "doc_agent_prompt": doc_agent_prompt,
    "graph_prompt": GRAPH_PROMPT,
    "supervisor_agent_prompt": supervisor_agent_prompt,
    "prompt_injection_detect_prompt": prompt_injection_detect_prompt,
}


def get_prompt(key: str) -> str:
    """全局通用提示词获取函数，key不存在直接抛出异常"""
    prompt_text = PROMPT_LIBRARY.get(key)
    if prompt_text is None:
        raise KeyError(f"提示词库中不存在指定key，key={key}")
    return prompt_text
