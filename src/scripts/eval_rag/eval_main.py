import json
import os
from typing import List, Dict

# 配置
TOP_K = 5
# 使用你项目内已支持的聊天模型，改成你实际要用的
JUDGE_MODEL_NAME = "qwen3.7-flash"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

QUESTION_FILE = os.path.join(BASE_DIR, "eval_questions.json")
RESULT_FILE = os.path.join(BASE_DIR, "eval_result.json")
PASS_SCORE = 70


def load_questions(filepath: str) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_score(query: str, chunk_texts: List[str], chat_model):
    prompt = f"""
            问题：{query}
            检索片段：
            {chr(10).join(chunk_texts)}
            评估片段能否支撑完整回答问题，仅输出0-100整数，不要其他文字。
            """

    resp = chat_model.invoke(prompt)
    raw = resp.content.strip()
    return int("".join(c for c in raw if c.isdigit()))


def run_eval(rag_factory, llm_factory):
    questions = load_questions(QUESTION_FILE)
    result_list = []
    pass_cnt = 0

    chat_model = llm_factory.get_client(JUDGE_MODEL_NAME)

    for item in questions:
        q = item["query"]
        chunks = rag_factory.query(query=q, k=TOP_K)
        texts = [c["content"] for c in chunks]
        score = get_score(q, texts, chat_model)

        is_pass = score >= PASS_SCORE
        if is_pass:
            pass_cnt += 1

        result_list.append({
            "query": q,
            "top_k": TOP_K,
            "chunk_num": len(chunks),
            "score": score,
            "is_pass": is_pass,
            "chunks": chunks
        })

    total = len(questions)
    pass_rate = pass_cnt / total
    avg_score = sum(r["score"] for r in result_list) / total

    print(f"总数:{total} 达标:{pass_cnt} 平均分:{avg_score:.1f} 通过率:{pass_rate:.2%}")

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    from src.rag_agent.rag.factory import rag_factory
    from src.llm.factory import llm_factory
    run_eval(rag_factory, llm_factory)
    pass
