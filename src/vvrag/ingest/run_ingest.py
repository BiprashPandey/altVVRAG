"""Ingestion CLI (Section 6.6): PDF/images -> page PNGs + word boxes -> embeddings -> Qdrant/SQLite.

Resumable + idempotent: pages with saved embeddings are not re-embedded;
SQLite rows are replaced; Qdrant points use deterministic ids (upsert).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from vvrag.config import env, load_config
from vvrag.ingest import embed as embed_mod
from vvrag.ingest import index as index_mod
from vvrag.ingest import render as render_mod
from vvrag.ingest.ocr import needs_ocr, ocr_page
from vvrag.ingest.pooling import pooled_matrix
from vvrag.logging_utils import get_logger

log = get_logger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def discover_inputs(input_dir: str | Path, limit: int = 0) -> list[Path]:
    files = sorted(
        [p for p in Path(input_dir).rglob("*")
         if p.suffix.lower() in (".pdf", *IMAGE_SUFFIXES) and p.is_file()]
    )
    return files[:limit] if limit else files


def run_ingest(
    input_dir: str | Path,
    out_dir: str | Path,
    dpi: int = 150,
    lang: str = "eng",
    embedder: str = "fake",
    model_id: str = "",
    skip_ocr: bool = False,
    limit: int = 0,
    collection_full: str = "pages_full",
    collection_pooled: str = "pages_pooled",
    qdrant_mode: str = "local",
    qdrant_path: str = "",
) -> dict:
    from vvrag.schemas import PageMeta

    input_dir, out_dir = Path(input_dir), Path(out_dir)
    pages_dir = out_dir / "pages"
    emb_dir = out_dir / "embeddings"
    pages_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    files = discover_inputs(input_dir, limit)
    log.info("found %d input files in %s", len(files), input_dir)

    if embedder == "colqwen":
        eq = embed_mod.ColQwenEmbedder(model_id=model_id or "vidore/colqwen2-v1.0")
        eq.load()
    elif embedder == "fake":
        fake = embed_mod.FakePatchEmbedder()
    else:
        raise ValueError(f"unknown embedder: {embedder}")

    con = index_mod.connect_sqlite(out_dir / "meta.db")
    done = index_mod.indexed_pages(con)
    client = None
    dim = 32 if embedder == "fake" else None
    stats = {"docs": 0, "pages": 0, "skipped": 0, "ocr_pages": 0}

    for f in files:
        doc_id = f.stem
        if f.suffix.lower() == ".pdf":
            metas, words = render_mod.render_pdf(f, pages_dir, doc_id, dpi)
        else:
            meta = render_mod.load_image_as_page(f, pages_dir, doc_id, dpi=dpi)
            metas, words = [meta], []
        stats["docs"] += 1

        by_page: dict[int, list] = {}
        for w in words:
            by_page.setdefault(w.page, []).append(w)
        # OCR fallback for pages without a text layer.
        for m in metas:
            nw = len(by_page.get(m.page, []))
            if not skip_ocr and needs_ocr(nw):
                try:
                    extra = ocr_page(m.image_path, doc_id, m.page, lang=lang)
                    by_page.setdefault(m.page, []).extend(extra)
                    stats["ocr_pages"] += 1
                except Exception as e:
                    log.warning("OCR failed for %s p%d: %s", doc_id, m.page, e)

        for m in metas:
            key = (doc_id, m.page)
            if key in done and embed_mod.embedding_exists(emb_dir, doc_id, m.page):
                stats["skipped"] += 1
                continue
            if embedder == "fake":
                pe = fake.embed_page(doc_id, m.page)
                if dim is None:
                    dim = pe.dim
            else:
                pe = eq.embed_images([m.image_path])[0]
                if dim is None:
                    dim = pe.dim
            embed_mod.save_embedding(emb_dir, doc_id, m.page, pe.vectors, pe.grid_h, pe.grid_w,
                                     pe.n_special_dropped)
            if client is None:
                client = index_mod.get_qdrant_client(
                    qdrant_mode, qdrant_path or str(out_dir / "qdrant"),
                    url=env("QDRANT_URL"), api_key=env("QDRANT_API_KEY"))
                index_mod.ensure_collections(client, dim, collection_full, collection_pooled)
            pooled = pooled_matrix(pe.vectors, pe.grid_h, pe.grid_w)
            kinds = index_mod.pooled_kind_labels(pe.grid_h, pe.grid_w)
            assert len(kinds) == pooled.shape[0], (len(kinds), pooled.shape)
            index_mod.upsert_page_vectors(client, doc_id, m.page, pe.vectors, pooled, kinds,
                                          collection_full, collection_pooled)
            m.grid_h, m.grid_w = pe.grid_h, pe.grid_w
            index_mod.upsert_page(con, m)
            index_mod.replace_word_boxes(con, by_page.get(m.page, []))
            done.add(key)
            stats["pages"] += 1

    manifest = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input": str(input_dir),
        "stats": stats,
        "word_boxes": index_mod.count_word_boxes(con),
        "sqlite_pages": index_mod.count_pages(con),
    }
    if client is not None:
        manifest["qdrant_full"] = index_mod.count_points(client, collection_full)
        manifest["qdrant_pooled"] = index_mod.count_points(client, collection_pooled)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("ingest done: %s", json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="VVRAG offline ingestion (run once, GPU)")
    ap.add_argument("--input", default="data/raw")
    ap.add_argument("--out", default="data/index")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dpi", type=int, default=None)
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--embedder", default="fake", choices=["fake", "colqwen"])
    ap.add_argument("--model-id", default="")
    ap.add_argument("--skip-ocr", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    manifest = run_ingest(
        args.input, args.out,
        dpi=args.dpi or cfg.render.dpi,
        lang=args.lang,
        embedder=args.embedder,
        model_id=args.model_id or cfg.retriever.model_id,
        skip_ocr=args.skip_ocr,
        limit=args.limit,
        collection_full=cfg.qdrant.collection_full,
        collection_pooled=cfg.qdrant.collection_pooled,
        qdrant_mode=cfg.qdrant.mode,
        qdrant_path=cfg.qdrant.path,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
