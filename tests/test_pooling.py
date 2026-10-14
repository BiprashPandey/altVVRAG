"""Pooling + grid tests (Sections 6.3-6.4, 11). Pure numpy, no GPU."""
import numpy as np
import pytest

from vvrag.ingest import pooling
from vvrag.ingest.embed import (
    FakePatchEmbedder,
    GridMismatchError,
    embedding_exists,
    load_embedding,
    save_embedding,
)


def test_pooling_hand_computed():
    # 2x2 grid, dim 2, row-major: p00=[1,0] p01=[0,1] p10=[2,0] p11=[0,2]
    v = np.array([[1, 0], [0, 1], [2, 0], [0, 2]], dtype=np.float32)
    parts = pooling.pool_patches(v, 2, 2)
    np.testing.assert_allclose(parts["rows"], [[0.5, 0.5], [1.0, 1.0]], rtol=1e-6)
    np.testing.assert_allclose(parts["cols"], [[1.5, 0.0], [0.0, 1.5]], rtol=1e-6)
    np.testing.assert_allclose(parts["global"], [[0.75, 0.75]], rtol=1e-6)
    m = pooling.pooled_matrix(v, 2, 2)
    assert m.shape == (2 + 2 + 1, 2)
    assert pooling.expected_pooled_count(2, 2) == 5


def test_fake_embedder_grid_count():
    pe = FakePatchEmbedder(grid_h=8, grid_w=6, dim=32).embed_page("d", 0)
    assert pe.vectors.shape == (8 * 6, 32)  # n_patch == grid_h * grid_w
    m = pooling.pooled_matrix(pe.vectors, pe.grid_h, pe.grid_w)
    assert m.shape[0] == pooling.expected_pooled_count(8, 6) == 15


def test_save_load_roundtrip_and_mismatch(tmp_path):
    pe = FakePatchEmbedder(grid_h=4, grid_w=3, dim=8).embed_page("d", 1)
    save_embedding(tmp_path, "d", 1, pe.vectors, pe.grid_h, pe.grid_w)
    assert embedding_exists(tmp_path, "d", 1)
    back = load_embedding(tmp_path, "d", 1)
    assert (back.grid_h, back.grid_w, back.dim) == (4, 3, 8)
    np.testing.assert_allclose(back.vectors, pe.vectors.astype(np.float32), rtol=1e-2)
    with pytest.raises(GridMismatchError):
        save_embedding(tmp_path, "d", 2, pe.vectors, 5, 5)  # wrong grid
