"""MaxSim unit tests: optimized implementation vs brute-force reference."""
import numpy as np
import pytest

from vvrag.retrieve.maxsim import maxsim_rank, maxsim_score, similarity_matrix


def _bruteforce(query: np.ndarray, page: np.ndarray) -> float:
    total = 0.0
    for t in range(query.shape[0]):
        best = float("-inf")
        for p in range(page.shape[0]):
            best = max(best, float(query[t] @ page[p]))
        total += best
    return total


def test_maxsim_matches_bruteforce():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((5, 16)).astype(np.float32)
    p = rng.standard_normal((20, 16)).astype(np.float32)
    assert maxsim_score(q, p) == pytest.approx(_bruteforce(q, p), rel=1e-5)
    s = similarity_matrix(q, p)
    assert s.shape == (5, 20)
    assert float(s.max(axis=1).sum()) == pytest.approx(_bruteforce(q, p), rel=1e-5)


def test_maxsim_rank_orders_best_first():
    rng = np.random.default_rng(1)
    q = rng.standard_normal((4, 8)).astype(np.float32)
    pages = [rng.standard_normal((10, 8)).astype(np.float32) for _ in range(4)]
    pages[2] = np.tile(q, (3, 1))[:10].astype(np.float32)  # every token has an exact match
    order, scores = maxsim_rank(q, pages)
    assert order[0] == 2
    assert [scores[i] for i in order] == sorted(scores, reverse=True)
