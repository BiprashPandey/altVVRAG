"""Two-stage retrieval (Section 7.1): pooled prefetch -> exact MaxSim rerank."""
from __future__ import annotations

import numpy as np

from vvrag.logging_utils import get_logger
from vvrag.retrieve.maxsim import maxsim_score
from vvrag.schemas import RetrievalHit

log = get_logger(__name__)


def retrieval_margin(hits: list[RetrievalHit]) -> float:
    if len(hits) < 2:
        return 0.0
    return hits[0].score - hits[1].score


class TwoStageRetriever:
    def __init__(
        self,
        client,
        collection_full: str = "pages_full",
        collection_pooled: str = "pages_pooled",
        top_k: int = 3,
        prefetch_k_multiplier: int = 4,
    ) -> None:
        self.client = client
        self.collection_full = collection_full
        self.collection_pooled = collection_pooled
        self.top_k = top_k
        self.prefetch_k_multiplier = prefetch_k_multiplier

    # ---------------------------------------------------------- stage 1 ---
    def prefetch(self, query_vecs: np.ndarray) -> list[tuple[str, int, float]]:
        """Search pooled vectors per query token; aggregate to candidate pages.

        Page stage1 score = sum over tokens of the best pooled-vector dot for
        that token (MaxSim over the pooled set). Returns top
        prefetch_k_multiplier * top_k (doc_id, page, score) candidates.
        """
        from qdrant_client import models

        query_vecs = np.asarray(query_vecs, dtype=np.float32)
        target = self.prefetch_k_multiplier * self.top_k
        best: dict[tuple[str, int], list[float]] = {}
        t = query_vecs.shape[0]
        for ti in range(t):
            res = self.client.query_points(
                self.collection_pooled,
                query=query_vecs[ti].tolist(),
                limit=target,
            )
            for pt in res.points:
                key = (pt.payload["doc_id"], pt.payload["page"])
                col = best.setdefault(key, [float("-inf")] * t)
                col[ti] = max(col[ti], pt.score)
        cands = [
            (doc, page, float(sum(v if v != float("-inf") else 0.0 for v in scores)))
            for (doc, page), scores in best.items()
        ]
        cands.sort(key=lambda c: c[2], reverse=True)
        return cands[:target]

    # ---------------------------------------------------------- stage 2 ---
    def _full_vectors(self, doc_id: str, page: int) -> np.ndarray:
        from qdrant_client import models

        flt = models.Filter(
            must=[
                models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                models.FieldCondition(key="page", match=models.MatchValue(value=page)),
            ]
        )
        pts, _ = self.client.scroll(
            self.collection_full, scroll_filter=flt, limit=1, with_vectors=True
        )
        if not pts:
            raise KeyError(f"page {doc_id}/{page} not in {self.collection_full}")
        return np.asarray(pts[0].vector, dtype=np.float32)

    def rerank(
        self, query_vecs: np.ndarray, candidates: list[tuple[str, int, float]]
    ) -> list[RetrievalHit]:
        query_vecs = np.asarray(query_vecs, dtype=np.float32)
        hits: list[RetrievalHit] = []
        for doc_id, page, s1 in candidates:
            try:
                full = self._full_vectors(doc_id, page)
            except KeyError:
                continue
            hits.append(
                RetrievalHit(doc_id=doc_id, page=page,
                             score=maxsim_score(query_vecs, full), stage1_score=s1)
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: self.top_k]

    def search(self, query_vecs: np.ndarray) -> list[RetrievalHit]:
        cands = self.prefetch(query_vecs)
        log.debug("prefetch: %d candidates", len(cands))
        return self.rerank(query_vecs, cands)
