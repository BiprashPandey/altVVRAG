"""Page fusion + visual scoring (Phase 1, 2026-10-18).

The live answer path historically ranked pages by OCR keyword overlap only
(`ui.qa.keyword_rank`). Phase 1 adds exact ColQwen MaxSim page scoring (from
the per-page patch embeddings already in the index) and a reciprocal-rank
fusion of the two page rankings. RRF is rank-based, so the keyword fraction
and MaxSim scores never need score calibration.

Deck aggregate = max page visual score per deck: a robust signal when the
gold page is mid-ranked but the right deck is confidently on top.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vvrag.ingest.embed import load_embedding
from vvrag.retrieve.maxsim import maxsim_score


def visual_page_scores(index_dir: str | Path, pages: list[tuple[str, int]],
                       query_vecs: np.ndarray) -> dict[tuple[str, int], float]:
    """Exact MaxSim of the query multivector against every listed page.

    Pages without stored embeddings are skipped (missing -> not scored).
    """
    index_dir = Path(index_dir)
    emb_dir = index_dir / "embeddings"
    q = np.asarray(query_vecs, dtype=np.float32)
    scores: dict[tuple[str, int], float] = {}
    for doc_id, page in pages:
        try:
            pe = load_embedding(emb_dir, doc_id, page)
        except Exception:  # noqa: BLE001 - unembedded page = no visual score
            continue
        scores[(doc_id, page)] = maxsim_score(q, pe.vectors)
    return scores


def deck_aggregate(page_scores: dict[tuple[str, int], float]) -> dict[str, float]:
    """max page score per doc_id (robust deck-level visual confidence)."""
    agg: dict[str, float] = {}
    for (doc_id, _page), s in page_scores.items():
        if doc_id not in agg or s > agg[doc_id]:
            agg[doc_id] = float(s)
    return agg


def rrf_fuse(rankings: list[list[tuple[str, int]]], k: int = 60,
             weights: list[float] | None = None) -> list[tuple[str, int, float]]:
    """Reciprocal-rank fusion. Inputs: per-source ordered (doc_id, page), best
    first. Returns [(doc_id, page, fused_score)] sorted desc; pages missing
    from a source contribute nothing from that source."""
    weights = list(weights) if weights else [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights length must match rankings")
    fused: dict[tuple[str, int], float] = {}
    for w, ranking in zip(weights, rankings):
        for i, key in enumerate(ranking):
            fused[key] = fused.get(key, 0.0) + w / (k + i + 1)
    return sorted(((d, p, s) for (d, p), s in fused.items()),
                  key=lambda t: t[2], reverse=True)


def rank_order(page_scores: dict[tuple[str, int], float]) -> list[tuple[str, int]]:
    """(doc_id, page) list sorted by score desc (stable: ties keep insertion)."""
    return [key for key, _ in sorted(page_scores.items(), key=lambda kv: kv[1],
                                     reverse=True)]
