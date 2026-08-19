"""HybridMemoryAgent — episodic memory (Qdrant) + stable profile (Feast).

See bonus/ARCHITECTURE.md for the design rationale behind the choices below
(chunking size, RRF over per-user memory, graceful Feast degradation).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from app.embeddings import Embedder

FEAST_DIR = _REPO_ROOT / "app" / "feast_repo"
COLLECTION = "agent_memory"
CHUNK_WORDS = 40   # small chunks: retrieval precision matters more than storage cost at this scale
RRF_K = 60         # same constant NB2 uses -- keep fusion behaviour consistent across the lab


def _chunk(text: str, max_words: int = CHUNK_WORDS) -> list[str]:
    """Sentence-aware greedy packing, capped at max_words per chunk."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    n = 0
    for s in sentences:
        words = s.split()
        if n + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current, n = [], 0
        current.extend(words)
        n += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


class HybridMemoryAgent:
    """Combines episodic memory (vector store) with a stable user profile (Feast).

    - `remember()` chunks + embeds text, upserts into a Qdrant collection
      filtered by `user_id` (single shared collection, payload-filtered --
      see ARCHITECTURE.md for why this beats per-user collections at this scale).
    - `recall()` pulls the user's Feast profile + recent activity, hybrid
      searches (BM25 + vector, RRF-fused) that user's own memories, and
      assembles a context string. No LLM call, per the bonus spec.
    """

    def __init__(self, feast_repo_path: Path | str = FEAST_DIR) -> None:
        self.embedder = Embedder()
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE),
        )
        self._next_id = 0
        self._by_user: dict[str, list[dict]] = {}

        self.feast = None
        try:
            from feast import FeatureStore
            self.feast = FeatureStore(repo_path=str(feast_repo_path))
        except Exception:
            # POC degrades gracefully if NB4 (feast apply/materialize) hasn't run yet.
            self.feast = None

    # ── write path ──────────────────────────────────────────────────────
    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user."""
        for chunk in _chunk(text):
            vec = next(self.embedder.embed([chunk])).tolist()
            point_id = self._next_id
            self._next_id += 1
            self.client.upsert(
                collection_name=COLLECTION,
                points=[models.PointStruct(
                    id=point_id, vector=vec,
                    payload={"user_id": user_id, "text": chunk},
                )],
            )
            self._by_user.setdefault(user_id, []).append({"id": point_id, "text": chunk})

    # ── read path ───────────────────────────────────────────────────────
    def _profile(self, user_id: str) -> dict:
        if self.feast is None:
            return {}
        try:
            feats = self.feast.get_online_features(
                features=[
                    "user_profile_features:topic_affinity",
                    "user_profile_features:preferred_language",
                    "user_profile_features:reading_speed_wpm",
                    "query_velocity_features:queries_last_hour",
                ],
                entity_rows=[{"user_id": user_id}],
            ).to_dict()
            return {k: v[0] for k, v in feats.items() if k != "user_id"}
        except Exception as exc:                          # noqa: BLE001
            return {"_error": str(exc)}

    def _hybrid_search(self, query: str, user_id: str, top_k: int = 3) -> list[str]:
        """Vector + BM25 over just this user's memories, fused with RRF."""
        mine = self._by_user.get(user_id, [])
        if not mine:
            return []

        qv = next(self.embedder.embed([query])).tolist()
        vec_hits = self.client.query_points(
            collection_name=COLLECTION,
            query=qv,
            query_filter=models.Filter(must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
            ]),
            limit=len(mine),
        ).points
        vec_order = [h.payload["text"] for h in vec_hits]

        bm25 = BM25Okapi([m["text"].lower().split() for m in mine])
        scores = bm25.get_scores(query.lower().split())
        bm25_order = [mine[i]["text"] for i in sorted(range(len(mine)), key=lambda i: -scores[i])]

        rrf: dict[str, float] = {}
        for ranked in (vec_order, bm25_order):
            for rank, text in enumerate(ranked, start=1):
                rrf[text] = rrf.get(text, 0.0) + 1.0 / (RRF_K + rank)
        return [t for t, _ in sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k]]

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve top-K memories + user profile features -> assembled context."""
        profile = self._profile(user_id)
        memories = self._hybrid_search(query, user_id)

        affinity = profile.get("topic_affinity", "unknown")
        speed = profile.get("reading_speed_wpm", "unknown")
        recent = profile.get("queries_last_hour", "unknown")

        lines = [
            f"User likes {affinity!r} topics, reading at {speed} wpm.",
            f"Recent activity: {recent} queries in the last hour.",
        ]
        if memories:
            lines.append("Top memories:")
            lines.extend(f"  {i}. {m}" for i, m in enumerate(memories, 1))
        else:
            lines.append("Top memories: (none stored yet for this user)")
        return "\n".join(lines)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    agent = HybridMemoryAgent()
    agent.remember(
        "Đã đọc bài về Kubernetes autoscaling và service mesh.", user_id="u_001"
    )
    print(agent.recall("Tôi đã đọc gì về Kubernetes?", user_id="u_001"))
