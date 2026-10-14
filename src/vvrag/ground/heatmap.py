"""Patch-similarity heatmap grounding (Section 7.2, contribution C1).

Query-token x patch similarity -> per-patch heat -> normalized grid.
Aggregation mode (max | mean) is a config option for the ablation.
"""
from __future__ import annotations

import numpy as np

from vvrag.retrieve.maxsim import similarity_matrix
from vvrag.schemas import Heatmap

# Tokens that carry no grounding signal (instruction/padding/specials).
# Verified against FakeQueryEmbedder (plain words); the real ColQwen query
# processor's specials must be confirmed on the GPU box and added here.
DEFAULT_DROP_TOKENS = frozenset({
    "<pad>", "<s>", "</s>", "<unk>", "<bos>", "<eos>",
    "[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]",
    "[/INST]", "[INST]", "<|im_start|>", "<|im_end|>",
    "", " ", "\n",
})


def filter_tokens(
    tokens: list[str],
    query_vecs: np.ndarray,
    drop: frozenset[str] = DEFAULT_DROP_TOKENS,
) -> tuple[list[str], np.ndarray]:
    """Drop special/padding/instruction tokens. Keeps at least one token."""
    keep = [i for i, t in enumerate(tokens) if t.strip() not in drop and t.strip()]
    if not keep:
        keep = list(range(len(tokens)))  # degenerate: keep all rather than empty
    return [tokens[i] for i in keep], np.asarray(query_vecs)[keep]


def aggregate_heatmap(
    sim: np.ndarray,
    mode: str = "max",
    token_weights: np.ndarray | None = None,
) -> np.ndarray:
    """sim: (T, N) -> heat (N,). mode in {max, mean}; optional token weighting."""
    sim = np.asarray(sim, dtype=np.float32)
    if token_weights is not None:
        w = np.asarray(token_weights, dtype=np.float32).reshape(-1, 1)
        sim = sim * w
    if mode == "max":
        return sim.max(axis=0)
    if mode == "mean":
        return sim.mean(axis=0)
    raise ValueError(f"unknown aggregation mode: {mode}")


def to_grid(heat: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    heat = np.asarray(heat, dtype=np.float32)
    if heat.size != grid_h * grid_w:
        raise ValueError(f"heat size {heat.size} != grid {grid_h}x{grid_w}")
    return heat.reshape(grid_h, grid_w)


def normalize_grid(grid: np.ndarray) -> np.ndarray:
    """Min-max normalize per page to [0, 1]. Constant grid -> zeros."""
    grid = np.asarray(grid, dtype=np.float32)
    lo, hi = float(grid.min()), float(grid.max())
    if hi - lo < 1e-9:
        return np.zeros_like(grid)
    return (grid - lo) / (hi - lo)


def smooth_grid(grid: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return grid
    return gaussian_filter(grid, sigma=sigma).astype(np.float32)


def heat_concentration(grid_norm: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of total heat inside the chosen boxes. Scalar in [0, 1]."""
    total = float(grid_norm.sum())
    if total <= 0:
        return 0.0
    return float(grid_norm[mask].sum() / total)


def build_heatmap(
    query_vecs: np.ndarray,
    query_tokens: list[str],
    page_vecs: np.ndarray,
    grid_h: int,
    grid_w: int,
    doc_id: str = "",
    page: int = 0,
    mode: str = "max",
    smooth_sigma: float = 0.0,
    keep_per_token: bool = False,
) -> tuple[Heatmap, np.ndarray]:
    """Returns (Heatmap schema with raw grid, normalized grid)."""
    tokens, qv = filter_tokens(query_tokens, query_vecs)
    sim = similarity_matrix(qv, page_vecs)  # (T_kept, N)
    heat = aggregate_heatmap(sim, mode)
    grid = to_grid(heat, grid_h, grid_w)
    if smooth_sigma > 0:
        grid = smooth_grid(grid, smooth_sigma)
    norm = normalize_grid(grid)
    hm = Heatmap(
        doc_id=doc_id,
        page=page,
        grid=grid.tolist(),
        query_tokens=tokens,
        per_token=sim.reshape(len(tokens), grid_h, grid_w).tolist() if keep_per_token else None,
    )
    return hm, norm
