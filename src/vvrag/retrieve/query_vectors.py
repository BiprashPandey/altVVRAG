"""Query multivector cache (Phase 1, 2026-10-18).

The live answer path must not block on ColQwen (weights are GPU-class): query
multivectors are precomputed ahead of time by
`scripts/precompute_query_vectors.py` (Colab or any machine with the weights),
stored on disk under `<root>/<sha1(text)>.json`, and read here. A cache miss
degrades retrieval to keyword-only and is recorded in the result dict
(`retrieval.mode_used`), never silently.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def query_key(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


class QueryVectorStore:
    """Filesystem store: one JSON per question (`vectors` + `tokens` + provenance)."""

    def __init__(self, root: str | Path = "data/query_cache") -> None:
        self.root = Path(root)

    def path(self, text: str) -> Path:
        return self.root / f"{query_key(text)}.json"

    def has(self, text: str) -> bool:
        return self.path(text).exists()

    def get(self, text: str) -> tuple[np.ndarray, list[str]] | None:
        p = self.path(text)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            vecs = np.asarray(d["vectors"], dtype=np.float32)
            if vecs.ndim != 2 or vecs.shape[0] == 0:
                return None
            return vecs, [str(t) for t in d.get("tokens", [])]
        except Exception:  # noqa: BLE001 - corrupt cache = miss
            return None

    def put(self, text: str, vectors: np.ndarray, tokens: list[str] | None = None,
            model: str = "", device: str = "") -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        vecs = np.asarray(vectors, dtype=np.float32)
        p = self.path(text)
        p.write_text(
            json.dumps({
                "text": text,
                "tokens": list(tokens or []),
                "model": model,
                "device": device,
                "shape": list(vecs.shape),
                "vectors": vecs.round(6).tolist(),
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        return p

    def coverage(self, texts: list[str]) -> tuple[int, int]:
        texts = list(texts)
        return sum(1 for t in texts if self.has(t)), len(texts)
