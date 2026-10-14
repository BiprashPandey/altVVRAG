"""Query embedding (Section 7.1): same model as pages, one vector per query token.

FakeQueryEmbedder is deterministic (seeded by text) for CPU tests.
ColQwen query path reuses ColQwenEmbedder's model with the query processor —
UNVERIFIED until the GPU box (same Option A caveat as page embeddings).
"""
from __future__ import annotations

import hashlib

import numpy as np


class FakeQueryEmbedder:
    """Deterministic query vectors. Related-page hint: seed text 'page:{doc}/{n}'
    yields vectors close to that page's FakePatchEmbedder vectors (shared seed
    scheme) — used to build the debug retrieval set in tests."""

    def __init__(self, dim: int = 32, n_tokens: int = 8) -> None:
        self.dim = dim
        self.n_tokens = n_tokens

    def embed_query(self, text: str) -> tuple[np.ndarray, list[str]]:
        tokens = [t for t in text.lower().split() if t][: self.n_tokens] or ["q"]
        vecs = np.zeros((len(tokens), self.dim), dtype=np.float32)
        for i, tok in enumerate(tokens):
            seed = int(hashlib.sha256(f"{text}#{tok}#{i}".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.dim).astype(np.float32)
            vecs[i] = v / (np.linalg.norm(v) + 1e-9)
        return vecs, tokens
