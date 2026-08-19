"""5-query demo of HybridMemoryAgent (see bonus/ARCHITECTURE.md).

Run:  python bonus/demo.py
Exits 0 and prints the assembled context for each query.
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_BONUS_DIR = Path(__file__).resolve().parent
if str(_BONUS_DIR) not in sys.path:
    sys.path.insert(0, str(_BONUS_DIR))

from agent import HybridMemoryAgent  # noqa: E402

USER = "u_001"

# Seed a small episodic memory set covering the topics the 5 demo queries probe.
MEMORIES = [
    "Đã đọc xong tài liệu về Kubernetes: horizontal pod autoscaler hoạt động "
    "thế nào và khi nào nên dùng cluster autoscaler thay vì HPA.",
    "Ghi chú về phương pháp tự động mở rộng hạ tầng (auto-scaling) theo lưu "
    "lượng người dùng trên AWS, dùng CloudWatch alarms để trigger scale-out.",
    "Đọc báo cáo cloud security: least-privilege IAM, encryption at rest, và "
    "audit logging cho hệ thống SaaS multi-tenant.",
    "Note về service mesh Istio: mTLS giữa các microservices và traffic shaping.",
    "Tài liệu về container orchestration best practices: rolling update "
    "không downtime, health check readiness/liveness probe.",
]

QUERIES = [
    "Tôi đã đọc gì về Kubernetes?",
    "Recommend đọc gì tiếp",
    "Tôi đang quan tâm gì gần đây?",
    "Tài liệu về tự động mở rộng hạ tầng?",
    "Cho tôi summary cloud security",
]


def main() -> int:
    agent = HybridMemoryAgent()
    for m in MEMORIES:
        agent.remember(m, user_id=USER)

    for i, q in enumerate(QUERIES, 1):
        print(f"=== Query {i}: {q!r} ===")
        print(agent.recall(q, user_id=USER))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
