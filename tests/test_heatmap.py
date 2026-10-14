"""Heatmap unit tests (Section 7.2): aggregation, filtering, normalization."""
import numpy as np
import pytest

from vvrag.ground.heatmap import (
    aggregate_heatmap,
    build_heatmap,
    filter_tokens,
    heat_concentration,
    normalize_grid,
    to_grid,
)


def test_filter_tokens_drops_specials():
    toks = ["what", "<pad>", "[SEP]", "revenue", ""]
    vecs = np.eye(5, dtype=np.float32)
    kept_t, kept_v = filter_tokens(toks, vecs)
    assert kept_t == ["what", "revenue"]
    assert kept_v.shape == (2, 5)


def test_filter_tokens_degenerate_keeps_all():
    toks = ["<pad>", "[SEP]"]
    vecs = np.eye(2, dtype=np.float32)
    kept_t, _ = filter_tokens(toks, vecs)
    assert kept_t == toks


def test_aggregate_max_vs_mean():
    sim = np.array([[0.9, 0.1, 0.0], [0.0, 0.8, 0.2]], dtype=np.float32)
    np.testing.assert_allclose(aggregate_heatmap(sim, "max"), [0.9, 0.8, 0.2], rtol=1e-6)
    np.testing.assert_allclose(aggregate_heatmap(sim, "mean"), [0.45, 0.45, 0.1], rtol=1e-6)
    with pytest.raises(ValueError):
        aggregate_heatmap(sim, "median")


def test_to_grid_size_mismatch():
    with pytest.raises(ValueError):
        to_grid(np.zeros(10, dtype=np.float32), 3, 3)


def test_normalize_constant_grid():
    g = np.full((4, 4), 0.7, dtype=np.float32)
    np.testing.assert_array_equal(normalize_grid(g), np.zeros((4, 4)))
    g2 = np.array([[0.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(normalize_grid(g2), [[0.0, 1.0]], rtol=1e-6)


def test_concentration_bounds():
    g = np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    full = np.ones_like(g, dtype=bool)
    assert heat_concentration(g, full) == pytest.approx(1.0)
    assert 0.0 <= heat_concentration(g, np.zeros_like(g, dtype=bool)) <= 1.0
    assert heat_concentration(np.zeros_like(g), full) == 0.0


def test_build_heatmap_peak_at_gold_patch():
    rng = np.random.default_rng(7)
    gh, gw, dim = 6, 8, 32
    page = rng.standard_normal((gh * gw, dim)).astype(np.float32)
    page /= np.linalg.norm(page, axis=1, keepdims=True)
    gold = 2 * gw + 4  # cell (2, 4)
    q = page[[gold, gold]] + 0.05 * rng.standard_normal((2, dim)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    hm, norm = build_heatmap(q, ["alpha", "beta"], page, gh, gw,
                             doc_id="d", page=0, mode="max")
    assert norm.shape == (gh, gw)
    assert hm.query_tokens == ["alpha", "beta"]
    peak = int(np.argmax(norm))
    pr, pc = peak // gw, peak % gw
    assert abs(pr - 2) <= 1 and abs(pc - 4) <= 1, (pr, pc)
