"""Row/column mean pooling of patch vectors (Section 6.4).

Stage-1 prefetch only. Input: patch grid (grid_h, grid_w, dim).
Output: row means (grid_h, dim) + column means (grid_w, dim) [+ global mean].
"""
from __future__ import annotations

import numpy as np


def pool_patches(
    vectors: np.ndarray,
    grid_h: int,
    grid_w: int,
    include_global: bool = True,
) -> dict[str, np.ndarray]:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (grid_h * grid_w, vectors.shape[1]):
        raise ValueError(f"expected ({grid_h * grid_w}, D), got {vectors.shape}")
    grid = vectors.reshape(grid_h, grid_w, -1)
    rows = grid.mean(axis=1)  # (grid_h, dim)
    cols = grid.mean(axis=0)  # (grid_w, dim)
    out: dict[str, np.ndarray] = {"rows": rows, "cols": cols}
    if include_global:
        out["global"] = vectors.mean(axis=0, keepdims=True)  # (1, dim)
    return out


def pooled_matrix(
    vectors: np.ndarray, grid_h: int, grid_w: int, include_global: bool = True
) -> np.ndarray:
    """Stacked pooled vectors (grid_h + grid_w [+1], dim) for indexing."""
    parts = pool_patches(vectors, grid_h, grid_w, include_global)
    return np.concatenate([parts["rows"], parts["cols"]] + ([parts["global"]] if include_global else []), axis=0)


def expected_pooled_count(grid_h: int, grid_w: int, include_global: bool = True) -> int:
    return grid_h + grid_w + (1 if include_global else 0)
