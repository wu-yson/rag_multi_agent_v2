import json
import os
from typing import List, Dict, Tuple

from src.utils.logger import log

# 配置
TOP_K = 5
JUDGE_MODEL_NAME = "qwen3.5:9b"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTION_FILE = os.path.join(BASE_DIR, "eval_questions.json")
RESULT_FILE = os.path.join(BASE_DIR, "eval_result.json")
PASS_SCORE = 70


def load_questions(filepath: str) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_score(query: str, chunk_texts: List[str], chat_model) -> Tuple[int, str]:
    """返回 (分数, 模型原始输出)，0 分时可通过原始输出判断是解析失败还是真低分。"""
    prompt = (
        f"问题：{query}\n"
        f"检索片段：\n"
        f"{chr(10).join(chunk_texts)}\n\n"
        "评分标准：\n"
        "- 90-100：检索内容完整覆盖问题，可以直接回答\n"
        "- 70-89：检索内容覆盖主要信息，缺少部分细节\n"
        "- 50-69：检索内容只覆盖部分信息\n"
        "- 0-49：检索内容基本无法支撑回答\n\n"
        "仅输出整数分数，不要其他文字。"
    )
    resp = chat_model.invoke(prompt, reasoning=False)
    raw = str(resp.content).strip()
    digits = "".join(c for c in raw if c.isdigit())
    score = min(int(digits), 100) if digits else 0
    return score, raw


def run_eval(rag_factory, llm_factory):
    questions = load_questions(QUESTION_FILE)
    total = len(questions)

    # 阶段1：全部检索，embedding 模型只加载一次，连续调用
    retrieved = []
    for idx, item in enumerate(questions, start=1):
        q = item["query"]
        chunks = rag_factory.query(query=q, k=TOP_K)
        retrieved.append({"query": q, "chunks": chunks})
        log.info(f"[Eval] 检索进度 {idx}/{total}")

    # 阶段2：统一评分，聊天模型只加载一次，连续调用
    chat_model = llm_factory.get_client(JUDGE_MODEL_NAME)
    result_list = []
    pass_cnt = 0
    for idx, item in enumerate(retrieved, start=1):
        q = item["query"]
        chunks = item["chunks"]
        texts = [c["content"] for c in chunks]
        score, raw_output = get_score(q, texts, chat_model)
        log.info(f"[Eval] 评分进度 {idx}/{total} 分数 {score}")

        is_pass = score >= PASS_SCORE
        if is_pass:
            pass_cnt += 1
        result_list.append({
            "query": q,
            "top_k": TOP_K,
            "chunk_num": len(chunks),
            "score": score,
            "raw_output": raw_output,
            "is_pass": is_pass,
            "chunks": chunks,
        })

    pass_rate = pass_cnt / total if total else 0
    avg_score = sum(r["score"] for r in result_list) / total if total else 0

    print("\n===== 评估结果 =====")
    for r in result_list:
        tag = "✓" if r["is_pass"] else "✗"
        print(f"  {tag} [{r['score']:>3}分] {r['query']}")
    print(f"\n总数:{total}  达标:{pass_cnt}  平均分:{avg_score:.1f}  通过率:{pass_rate:.2%}")

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from src.config.settings import settings

    sys.path.insert(0, str(Path(settings.mcp_server_path).parent))
    from mcp_src.rag_tool.rag.factory import rag_factory
    from src.llm.factory import llm_factory
    run_eval(rag_factory, llm_factory)
