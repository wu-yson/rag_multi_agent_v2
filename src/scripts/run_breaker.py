"""单独测试熔断器：closed → open → 拦截（用中间件同款方法）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.llm.resilience import get_breaker


def main():
    br = get_breaker("test-model")
    print("初始状态:", br.state)          # closed

    # 连续失败 3 次 → 熔断打开
    for i in range(3):
        br.call_failure()
        print(f"第{i+1}次失败后状态: {br.state}")

    # 打开后：allow_request 拒绝，call_success 也不恢复（还在冷却期）
    print("打开后 allow_request:", br.allow_request())   # 预期 False
    br.call_success()
    print("尝试成功调用后状态:", br.state)   # 预期还是 open（冷却期内）


if __name__ == "__main__":
    main()
