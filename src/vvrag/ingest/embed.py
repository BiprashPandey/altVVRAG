"""Patch embeddings (Section 6.3).

Two implementations behind one interface:
- FakePatchEmbedder: deterministic CPU embedder for tests / smoke runs.
- ColQwenEmbedder: real ColQwen2-family late-interaction embedder, GPU only.

GRID SHAPE WARNING (Section 6.3): ColQwen2 uses dynamic resolution, so the
patch grid differs per page. The grid MUST be derived from the processor's
output metadata, never assumed. Verified against ColQwen2.forward source
(colpali-engine): the model returns one vector per INPUT token (text
prefix/suffix included), so the patch block is located exactly via
``input_ids == image_token_id`` and the grid via ``image_grid_thw`` with the
spatial merge factor. See docs/decisions.md 2026-10-15 entry.
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
    """Real ColQwen2-family embedder (GPU; device='auto' resolves cuda->cpu).

    Grid derivation (verified against ColQwen2.forward source, 2026-10-15):
    output rows align 1:1 with input_ids; patch rows = positions where
    input_ids == image_token_id, count must equal prod(image_grid_thw) //
    merge^2 with grid (h//merge, w//merge). Any mismatch raises
    GridMismatchError instead of silently shifting boxes.
    """

    def __init__(
        self,
        model_id: str = "vidore/colqwen2-v1.0",
        device: str = "auto",
        dtype: str = "bfloat16",
        batch_size: int = 1,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.batch_size = max(1, batch_size)
        self._model = None
        self._processor = None

    def _resolve_device(self) -> str:
        """Map 'auto' -> 'cuda' when usable, else 'cpu'. Called in load()."""
        import torch

        if self.device == "auto":
            cuda_built = getattr(torch.version, "cuda", None) is not None
            return "cuda" if (cuda_built and torch.cuda.is_available()) else "cpu"
        return self.device

    def load(self) -> None:
        try:
            import torch
            from colpali_engine.models import ColQwen2, ColQwen2Processor  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "ColQwenEmbedder needs torch + colpali-engine on a GPU machine "
                f"(see notebooks/ingest_colab.ipynb): {e}"
            ) from e
        # Fail fast BEFORE the multi-GB download: .to("cuda") on a CPU-only
        # torch wheel raises cryptic "Torch not compiled with CUDA enabled".
        resolved = self._resolve_device()
        if "cuda" in resolved:
            if getattr(torch.version, "cuda", None) is None:
                raise RuntimeError(
                    "CPU-only torch installed (torch.version.cuda is None) but "
                    f"device={self.device!r} needs CUDA. On Colab, reinstall the "
                    "CUDA wheel BEFORE importing torch, e.g.: "
                    "pip install -q torch --index-url "
                    "https://download.pytorch.org/whl/cu121 ; then Runtime > "
                    "Restart session and re-run. Do NOT pip install plain "
                    "'torch' from PyPI on Colab — it overwrites the CUDA build "
                    "with a CPU wheel."
                )
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA torch installed but no GPU is visible "
                    "(torch.cuda.is_available() is False). In Colab set Runtime "
                    "> Change runtime type > Hardware accelerator = T4 GPU, then "
                    "re-run. Verify with: python -c "
                    "\"import torch; print(torch.__version__, "
                    "torch.version.cuda, torch.cuda.is_available())\" and "
                    "!nvidia-smi."
                )
        self.device = resolved
        if self.device == "cpu":
            log.warning(
                "ingesting on CPU (%s): ColQwen2 is ~9GB and very slow without "
                "a GPU — use a T4 runtime for the full corpus.",
                self.model_id,
            )
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}[self.dtype]
        self._processor = ColQwen2Processor.from_pretrained(self.model_id)
        self._model = ColQwen2.from_pretrained(self.model_id, torch_dtype=torch_dtype).to(
            self.device
        ).eval()
        try:
            vram = torch.cuda.memory_allocated(self.device) / 1e9
            log.info("loaded %s on %s (%s), weights VRAM %.1f GB",
                     self.model_id, self.device, self.dtype, vram)
        except Exception:
            log.info("loaded %s on %s (%s)", self.model_id, self.device, self.dtype)

    def embed_images(self, image_paths: list[str | Path]) -> list[PatchEmbeddings]:
        if self._model is None or self._processor is None:
            raise RuntimeError("call load() first")
        import torch

        from PIL import Image

        out: list[PatchEmbeddings] = []
        batch_size = self.batch_size
        i = 0
        while i < len(image_paths):
            batch_paths = [Path(p) for p in image_paths[i : i + batch_size]]
            try:
                out.extend(self._embed_batch(batch_paths))
                i += len(batch_paths)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size <= 1:
                    raise RuntimeError(
                        "CUDA OOM with batch_size=1. Lower --dpi (e.g. 100) or use "
                        "fewer pixels per page, then re-run (embeddings are resumable)."
                    ) from None
                batch_size //= 2
                log.warning("CUDA OOM: retrying with batch_size=%d", batch_size)
        return out

    def _embed_batch(self, batch_paths: list[Path]) -> list[PatchEmbeddings]:
        import torch

        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in batch_paths]
        with torch.no_grad():
            batch = self._processor.process_images(images).to(self.device)
            # NOTE: exact output keys depend on the colpali-engine version.
            # Confirm on GPU via probe: expected keys include input_ids /
            # attention_mask plus image-grid metadata. Do NOT guess — if the
            # keys below stop matching, update this block and decisions.md.
            emb = self._model(**batch)
            vecs = emb.cpu().float().numpy()  # (B, seq, dim)
        out = []
        for b, p in enumerate(batch_paths):
            v, gh, gw, n_special = self._deduce_page_vectors(vecs[b], batch, b)
            out.append(PatchEmbeddings(vectors=v, grid_h=gh, grid_w=gw,
                                       dim=v.shape[1], n_special_dropped=n_special))
        return out

    def _merge_size(self) -> int:
        """Spatial merge factor (default 2). Read from model/processor when available."""
        for obj in (self._model, self._processor,
                    getattr(self._processor, "image_processor", None)):
            for attr in ("spatial_merge_size", "merge_size"):
                v = getattr(obj, attr, None) if obj is not None else None
                if isinstance(v, int) and v >= 1:
                    return v
        return 2

    def _image_token_id(self, batch) -> int | None:
        """Resolve the image-pad token id used in batch['input_ids']."""
        for obj in (getattr(self._model, "config", None), self._processor,
                    getattr(self._processor, "tokenizer", None)):
            for attr in ("image_token_id",):
                v = getattr(obj, attr, None) if obj is not None else None
                if isinstance(v, int) and v >= 0:
                    return v
        tok = getattr(self._processor, "tokenizer", None)
        img_tok = getattr(self._processor, "image_token", None) or "<|image_pad|>"
        for conv in ("convert_tokens_to_ids",):
            fn = getattr(tok, conv, None) if tok is not None else None
            if fn is not None:
                try:
                    v = fn(img_tok)
                    if isinstance(v, int) and v >= 0:
                        return v
                except Exception:
                    continue
        return None

    @staticmethod
    def _batch_get(batch, key: str):
        if isinstance(batch, dict):
            v = batch.get(key, None)
            if v is not None:
                return v
        return getattr(batch, key, None)

    def _deduce_page_vectors(
        self, vecs: np.ndarray, batch, idx: int
    ) -> tuple[np.ndarray, int, int, int]:
        """Derive (patch_vectors, grid_h, grid_w, n_dropped) for one image.

        Verified against ColQwen2.forward (colpali-engine): the model returns
        one vector per INPUT token, i.e. text prefix/suffix tokens are included
        alongside image patches (mask_non_image_embeddings=False by default).
        So the patch block is located exactly via
        ``batch['input_ids'][idx] == image_token_id`` and the grid comes from
        ``batch['image_grid_thw'][idx]`` (t, h, w) with spatial merge
        ``grid = (h//merge, w//merge)``. Never assumes a fixed grid or a fixed
        prefix length.
        """
        import torch

        seq, dim = vecs.shape
        keys = list(batch.keys()) if isinstance(batch, dict) else []
        diag = f"batch item {idx} (seq_len={seq}, dim={dim}, batch_keys={keys})"

        thw_raw = None
        for attr in ("image_grid_thw", "image_grid", "grid_thw"):
            thw_raw = self._batch_get(batch, attr)
            if thw_raw is not None:
                break
        if thw_raw is None:
            raise GridMismatchError(
                f"could not derive patch grid for {diag}: no image-grid key "
                f"in batch (tried image_grid_thw/image_grid/grid_thw)."
            )
        if isinstance(thw_raw, torch.Tensor):
            thw_raw = thw_raw.detach().cpu().tolist()
        thw = thw_raw[idx] if isinstance(thw_raw[0], (list, tuple)) else thw_raw
        t, h, w = (int(x) for x in thw[-3:])
        merge = self._merge_size()
        n_image = t * h * w // (merge * merge)
        gh, gw = h // merge, w // merge
        if t != 1 or gh * gw != n_image:
            raise GridMismatchError(
                f"could not derive patch grid for {diag}: thw={(t, h, w)} "
                f"merge={merge} gives grid {(gh, gw)} != {n_image} image tokens."
            )

        input_ids = self._batch_get(batch, "input_ids")
        img_id = self._image_token_id(batch)
        if input_ids is not None and img_id is not None:
            ids = input_ids[idx]
            if isinstance(ids, torch.Tensor):
                mask = (ids == img_id).detach().cpu().numpy()
            else:
                mask = (np.asarray(ids) == img_id)
            if int(mask.sum()) == n_image:
                return vecs[mask, :], gh, gw, int(seq - n_image)
            raise GridMismatchError(
                f"could not derive patch grid for {diag}: image-token mask "
                f"selects {int(mask.sum())} positions, expected {n_image} "
                f"(thw={(t, h, w)}, image_token_id={img_id})."
            )
        raise GridMismatchError(
            f"could not derive patch grid for {diag}: need input_ids + "
            f"image_token_id to locate the {n_image} patch vectors "
            f"(input_ids present: {input_ids is not None}, "
            f"image_token_id: {img_id})."
        )
