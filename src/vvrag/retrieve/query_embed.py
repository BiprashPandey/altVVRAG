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


class ColQwenQueryEmbedder:
    """Real ColQwen text-side query encoder (rec 5: replaces pseudo-queries).

    Uses the same checkpoint as page embeddings (vidore/colqwen2-v1.0,
    Option A) with the query processor path. GPU box only; CPU import
    raises a clear RuntimeError. Output dim (128) matches the index.
    """

    def __init__(self, model_id: str = "vidore/colqwen2-v1.0", device: str = "auto") -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None
        self.dim: int = 128

    def load(self) -> None:
        try:
            import torch
            from colpali_engine.models import ColQwen2, ColQwen2Processor  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "ColQwenQueryEmbedder needs torch + colpali-engine on the GPU box: "
                f"{e}"
            ) from e
        device = self.device
        if device == "auto":
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device
        self._processor = ColQwen2Processor.from_pretrained(self.model_id)
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._model = ColQwen2.from_pretrained(
            self.model_id, torch_dtype=torch_dtype).to(device).eval()
        self.dim = int(getattr(getattr(self._model, "config", None), "hidden_size", 128)
                       or 128)

    def embed_query(self, text: str) -> tuple[np.ndarray, list[str]]:
        if self._model is None or self._processor is None:
            raise RuntimeError("call load() first")
        import torch

        tokens = [t for t in text.lower().split() if t] or ["q"]
        with torch.no_grad():
            batch = self._processor.process_queries([text]).to(self.device)
            emb = self._model(**batch)
            if isinstance(emb, torch.Tensor):
                vecs = emb[0].detach().cpu().float().numpy()
            else:
                vecs = np.asarray(emb[0] if isinstance(emb, (list, tuple)) else emb,
                                    dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[0] == 0:
            raise RuntimeError(f"unexpected query embedding shape {vecs.shape}")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        return (vecs / norms).astype(np.float32), tokens[:vecs.shape[0]]


def create_query_embedder(kind: str = "fake", **kwargs):
    """Factory: kind in {"fake", "colqwen"}. ColQwen needs GPU + weights."""
    kind = (kind or "fake").lower()
    if kind == "fake":
        return FakeQueryEmbedder(dim=int(kwargs.get("dim", 32)))
    if kind in ("colqwen", "colqwen2", "real"):
        return ColQwenQueryEmbedder(model_id=str(kwargs.get(
            "model_id", "vidore/colqwen2-v1.0")),
            device=str(kwargs.get("device", "auto")))
    raise ValueError(f"unknown query encoder: {kind}")
