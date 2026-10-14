"""MaxSim scoring (Section 7.1): for each query token take the max dot product
over page patches, then sum over query tokens.

Also exposes the raw similarity matrix S[t, p] reused by heatmap grounding (C1).
"""
from __future__ import annotations

import numpy as np


def similarity_matrix(query_vecs: np.ndarray, page_vecs: np.ndarray) -> np.ndarray:
    """S[t, p] = dot(Q[t], P[p]). Shapes (T, D) x (N, D) -> (T, N)."""
    q = np.asarray(query_vecs, dtype=np.float32)
    p = np.asarray(page_vecs, dtype=np.float32)
    if q.ndim != 2 or p.ndim != 2 or q.shape[1] != p.shape[1]:
        raise ValueError(f"shape mismatch: {q.shape} vs {p.shape}")
    return q @ p.T


def maxsim_score(query_vecs: np.ndarray, page_vecs: np.ndarray) -> float:
    """sum_t max_p S[t, p]."""
    return float(similarity_matrix(query_vecs, page_vecs).max(axis=1).sum())


def maxsim_rank(
    query_vecs: np.ndarray, pages: list[np.ndarray]
) -> tuple[list[int], list[float]]:
    """Score every page; return (order desc, scores). Pure NumPy reference."""
    scores = [maxsim_score(query_vecs, p) for p in pages]
    order = sorted(range(len(pages)), key=lambda i: scores[i], reverse=True)
    return order, scores
