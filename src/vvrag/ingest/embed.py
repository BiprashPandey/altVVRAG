"""Patch embeddings (Section 6.3).

Two implementations behind one interface:
- FakePatchEmbedder: deterministic CPU embedder for tests / smoke runs.
- ColQwenEmbedder: real ColQwen2-family late-interaction embedder, GPU only.

GRID SHAPE WARNING (Section 6.3): ColQwen2 uses dynamic resolution, so the
patch grid differs per page. The grid MUST be derived from the processor's
output metadata, never assumed. Because torch/colpali-engine are not
installed on the dev laptop, the ColQwen path is UNVERIFIED here: on the GPU
machine, run `probe_environment()` first and record the output in
docs/decisions.md before ingesting (Option A: verify `vidore/colqwen2-v1.0`).
All embeddings are saved as float16 .npy + JSON sidecar so ingestion is
resumable and exportable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vvrag.logging_utils import get_logger

log = get_logger(__name__)


class GridMismatchError(RuntimeError):
    pass


@dataclass
class PatchEmbeddings:
    vectors: np.ndarray  # (n_patch, dim) float32, special tokens already dropped
    grid_h: int
    grid_w: int
    dim: int
    n_special_dropped: int = 0


def embedding_paths(out_dir: str | Path, doc_id: str, page: int) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    base = out_dir / f"{doc_id}_{page}"
    return base.with_suffix(".npy"), base.with_suffix(".json")


def embedding_exists(out_dir: str | Path, doc_id: str, page: int) -> bool:
    npy, js = embedding_paths(out_dir, doc_id, page)
    return npy.exists() and js.exists()


def save_embedding(
    out_dir: str | Path,
    doc_id: str,
    page: int,
    vectors: np.ndarray,
    grid_h: int,
    grid_w: int,
    n_special_dropped: int = 0,
) -> tuple[Path, Path]:
    """Save patch vectors (float16) + grid sidecar. Verifies grid count."""
    vectors = np.asarray(vectors)
    if vectors.ndim != 2:
        raise ValueError(f"expected (n_patch, dim), got {vectors.shape}")
    if vectors.shape[0] != grid_h * grid_w:
        raise GridMismatchError(
            f"page {doc_id}/{page}: {vectors.shape[0]} vectors != "
            f"grid {grid_h}x{grid_w}={grid_h * grid_w}"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy, js = embedding_paths(out_dir, doc_id, page)
    np.save(npy, vectors.astype(np.float16))
    js.write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "page": page,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "dim": int(vectors.shape[1]),
                "n_special_dropped": n_special_dropped,
                "dtype": "float16",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return npy, js


def load_embedding(out_dir: str | Path, doc_id: str, page: int) -> PatchEmbeddings:
    npy, js = embedding_paths(out_dir, doc_id, page)
    meta = json.loads(js.read_text(encoding="utf-8"))
    vectors = np.load(npy).astype(np.float32)
    return PatchEmbeddings(
        vectors=vectors,
        grid_h=int(meta["grid_h"]),
        grid_w=int(meta["grid_w"]),
        dim=int(meta["dim"]),
        n_special_dropped=int(meta.get("n_special_dropped", 0)),
    )


class FakePatchEmbedder:
    """Deterministic CPU embedder (seeded by doc_id/page). For tests only."""

    def __init__(self, grid_h: int = 8, grid_w: int = 6, dim: int = 32) -> None:
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.dim = dim

    def embed_page(self, doc_id: str, page: int) -> PatchEmbeddings:
        seed = int(hashlib.sha256(f"{doc_id}/{page}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vecs = rng.standard_normal((self.grid_h * self.grid_w, self.dim)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        return PatchEmbeddings(vectors=vecs, grid_h=self.grid_h, grid_w=self.grid_w, dim=self.dim)


def probe_environment() -> dict:
    """Print/inspect installed versions for the decision log. Run on the GPU box."""
    info: dict = {}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        info["torch"] = "NOT INSTALLED"
    try:
        import colpali_engine  # type: ignore

        info["colpali_engine"] = getattr(colpali_engine, "__version__", "unknown")
    except ImportError:
        info["colpali_engine"] = "NOT INSTALLED"
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = "NOT INSTALLED"
    log.info("ingest env: %s", json.dumps(info, indent=2))
    return info


class ColQwenEmbedder:
    """Real ColQwen2-family embedder (GPU). UNVERIFIED on dev laptop — see module docstring.

    Grid derivation strategy (to be confirmed with probe_environment() output
    on the GPU machine): the processor returns per-image grid metadata (patch
    grid before the model's spatial merge); the number of image-token vectors
    must equal grid_h * grid_w after dropping prefix/special tokens. Any
    mismatch raises GridMismatchError instead of silently shifting boxes.
    """

    def __init__(
        self,
        model_id: str = "vidore/colqwen2-v1.0",
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int = 4,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def load(self) -> None:
        try:
            import torch
            from colpali_engine.models import ColQwen2, ColQwen2Processor  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "ColQwenEmbedder needs torch + colpali-engine on a GPU machine "
                f"(see notebooks/ingest_colab.ipynb): {e}"
            ) from e
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}[self.dtype]
        self._processor = ColQwen2Processor.from_pretrained(self.model_id)
        self._model = ColQwen2.from_pretrained(self.model_id, torch_dtype=torch_dtype).to(
            self.device
        ).eval()
        log.info("loaded %s on %s (%s)", self.model_id, self.device, self.dtype)

    def embed_images(self, image_paths: list[str | Path]) -> list[PatchEmbeddings]:
        if self._model is None or self._processor is None:
            raise RuntimeError("call load() first")
        import torch

        from PIL import Image

        out: list[PatchEmbeddings] = []
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = [Path(p) for p in image_paths[i : i + self.batch_size]]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            with torch.no_grad():
                batch = self._processor.process_images(images).to(self.device)
                # NOTE: exact output keys depend on the colpali-engine version.
                # Confirm on GPU via probe: expected keys include input_ids /
                # attention_mask plus image-grid metadata. Do NOT guess — if the
                # keys below stop matching, update this block and decisions.md.
                emb = self._model(**batch)
                vecs = emb.cpu().float().numpy()  # (B, seq, dim)
            for b, p in enumerate(batch_paths):
                v, gh, gw, n_special = self._deduce_page_vectors(vecs[b], batch, b)
                out.append(PatchEmbeddings(vectors=v, grid_h=gh, grid_w=gw,
                                           dim=v.shape[1], n_special_dropped=n_special))
        return out

    def _deduce_page_vectors(
        self, vecs: np.ndarray, batch, idx: int
    ) -> tuple[np.ndarray, int, int, int]:
        """Derive (patch_vectors, grid_h, grid_w, n_special) for one image.

        Tries processor metadata in order; falls back to an explicit error
        telling the operator what to record. Never assumes a fixed grid.
        """
        seq, dim = vecs.shape
        candidates: list[tuple[int, int, int]] = []  # (gh, gw, n_special)
        # Attempt 1: image_grid_thw / grid metadata on the batch (ColQwen2 dynamic res).
        for attr in ("image_grid_thw", "image_grid", "grid_thw"):
            meta = getattr(batch, attr, None)
            if meta is None and isinstance(batch, dict):
                meta = batch.get(attr)
            if meta is not None:
                try:
                    import torch

                    if isinstance(meta, torch.Tensor):
                        meta = meta.cpu().tolist()
                    gh, gw = self._grid_from_thw(meta[idx] if isinstance(meta[0], (list, tuple)) else meta)
                    for n_special in (0, 1, 5):
                        if seq - n_special == gh * gw:
                            candidates.append((gh, gw, n_special))
                except Exception:
                    continue
        if not candidates:
            raise GridMismatchError(
                f"could not derive patch grid for batch item {idx} "
                f"(seq_len={seq}, dim={dim}). Run probe_environment() on the GPU box, "
                "inspect processor output keys, update _deduce_page_vectors, "
                "and record in docs/decisions.md."
            )
        gh, gw, n_special = candidates[0]
        return vecs[n_special:, :], gh, gw, n_special

    @staticmethod
    def _grid_from_thw(thw) -> tuple[int, int]:
        """ColQwen2 spatial merge: patches are merged 2x2 -> grid = (t*h*w merged)."""
        t, h, w = (int(x) for x in thw[-3:])
        merge = 2
        return h // merge, w // merge
