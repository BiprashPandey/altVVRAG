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
from vvrag.ingest.sentences import build_sentences
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
    device: str = "auto",
    dtype: str = "bfloat16",
    skip_ocr: bool = False,
    limit: int = 0,
    collection_full: str = "pages_full",
    collection_pooled: str = "pages_pooled",
    qdrant_mode: str = "local",
    qdrant_path: str = "",
    batch_size: int = 1,
    upscale: float = 2.0,
    vlm_transcribe_empty: bool = False,
    vlm_model: str = "",
) -> dict:
    from vvrag.schemas import PageMeta

    input_dir, out_dir = Path(input_dir), Path(out_dir)
    pages_dir = out_dir / "pages"
    emb_dir = out_dir / "embeddings"
    pages_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    files = discover_inputs(input_dir, limit)
    log.info("found %d input files in %s", len(files), input_dir)
    if not files:
        raise SystemExit(
            f"no PDFs/images found in {input_dir}. Upload files there first, then re-run."
        )

    if embedder == "colqwen":
        model_id = model_id or "vidore/colqwen2-v1.0"
        log.info(
            "loading ColQwen embedder %s on %s (%s, multi-GB download on first run)",
            model_id, device, dtype,
        )
        eq = embed_mod.ColQwenEmbedder(
            model_id=model_id, device=device, dtype=dtype, batch_size=batch_size
        )
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
        # OCR fallback for pages without a text layer (2x upscale; Phase 5a).
        ocr_pages: set[int] = set()
        vlm_pages: set[int] = set()
        for m in metas:
            nw = len(by_page.get(m.page, []))
            if not skip_ocr and needs_ocr(nw):
                try:
                    extra = ocr_page(m.image_path, doc_id, m.page, lang=lang,
                                     upscale=upscale)
                    if extra:
                        by_page.setdefault(m.page, []).extend(extra)
                        ocr_pages.add(m.page)
                        stats["ocr_pages"] += 1
                except Exception as e:
                    log.warning("OCR failed for %s p%d: %s", doc_id, m.page, e)
            if vlm_transcribe_empty and not by_page.get(m.page):
                extra = _vlm_page_words(m, doc_id, vlm_model)
                if extra:
                    by_page.setdefault(m.page, []).extend(extra)
                    vlm_pages.add(m.page)
                    stats["vlm_pages"] = stats.get("vlm_pages", 0) + 1

        # Sentence units (suggestion #1): cite sentences, not words.
        by_page_sents: dict[int, list] = {}
        for m in metas:
            page_words = by_page.get(m.page, [])
            sents, word_to = build_sentences(page_words)
            src = ("vlm" if m.page in vlm_pages
                   else "ocr" if m.page in ocr_pages else "textlayer")
            for w, sid in zip(page_words, word_to):
                w.sent_id = sid
            for s in sents:
                s.source = src
            by_page_sents[m.page] = sents
            if sents:
                log.info("doc %s p%d: %d sentences from %d words (source=%s)",
                         doc_id, m.page, len(sents), len(page_words), src)

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
            index_mod.replace_sentences(con, by_page_sents.get(m.page, []))
            done.add(key)
            stats["pages"] += 1

    manifest = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input": str(input_dir),
        "stats": stats,
        "word_boxes": index_mod.count_word_boxes(con),
        "sentences": index_mod.count_sentences(con),
        "sqlite_pages": index_mod.count_pages(con),
    }
    if client is not None:
        manifest["qdrant_full"] = index_mod.count_points(client, collection_full)
        manifest["qdrant_pooled"] = index_mod.count_points(client, collection_pooled)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("ingest done: %s", json.dumps(manifest, indent=2))
    return manifest


def _vlm_page_words(meta, doc_id: str, vlm_model: str):
    """Optional VLM transcription of a zero-word page -> page-level WordBoxes."""
    from vvrag.ingest.vlm_transcribe import transcribe_page, transcript_word_boxes

    model = vlm_model
    if not model:
        try:
            from vvrag.config import load_config

            model = load_config("config.yaml").reader.model
        except Exception:  # noqa: BLE001
            model = ""
    if not model:
        log.warning("no vlm model configured; skipping %s p%d", doc_id, meta.page)
        return []
    try:
        text = transcribe_page(meta.image_path, model)
    except Exception as e:  # noqa: BLE001
        log.warning("VLM transcription failed for %s p%d: %s", doc_id, meta.page, e)
        return []
    boxes = transcript_word_boxes(text, doc_id, meta.page, meta.width, meta.height)
    log.info("VLM %s p%d: %d sentence boxes (page-level geometry)",
             doc_id, meta.page, len(boxes))
    return boxes


def run_reocr(
    out_dir: str | Path,
    lang: str = "eng",
    min_words: int = 10,
    upscale: float = 2.0,
    vlm_transcribe_empty: bool = False,
    vlm_model: str = "",
    limit: int = 0,
) -> dict:
    """Re-OCR sparse pages WITHOUT re-embedding (Phase 5a).

    Embeddings and Qdrant vectors are untouched; word_boxes + sentences are
    replaced with OCR output (2x upscale) for pages below `min_words`.
    Optionally VLM-transcribes pages that stay at zero words. Intended to run
    where tesseract + (optionally) a Gemini key exist (e.g. Colab).
    """
    out_dir = Path(out_dir)
    con = index_mod.connect_sqlite(out_dir / "meta.db")
    pages = con.execute(
        "SELECT doc_id,page,width,height,image_path,has_text_layer FROM pages "
        "ORDER BY doc_id,page").fetchall()
    stats = {"scanned": 0, "reocr": 0, "vlm": 0, "words_added": 0, "failed": 0}
    for doc_id, page, width, height, image_path, _htl in pages:
        if limit and stats["scanned"] >= limit:
            break
        stats["scanned"] += 1
        nw = con.execute(
            "SELECT COUNT(*) FROM word_boxes WHERE doc_id=? AND page=?",
            (doc_id, page)).fetchone()[0]
        if nw >= min_words:
            continue
        from vvrag.schemas import PageMeta

        meta = PageMeta(doc_id=doc_id, page=page, width=width, height=height,
                        dpi=0, has_text_layer=bool(_htl), image_path=image_path)
        boxes, src = [], ""
        try:
            boxes = ocr_page(image_path, doc_id, page, lang=lang, upscale=upscale)
            src = "ocr-reocr"
        except Exception as e:  # noqa: BLE001
            log.warning("reocr failed %s p%d: %s", doc_id, page, e)
            stats["failed"] += 1
        if not boxes and vlm_transcribe_empty:
            boxes = _vlm_page_words(meta, doc_id, vlm_model)
            src = "vlm"
        if not boxes:
            continue
        sents, word_to = build_sentences(boxes)
        for w, sid in zip(boxes, word_to):
            w.sent_id = sid
        for s in sents:
            s.source = src
        index_mod.replace_word_boxes(con, boxes)
        index_mod.replace_sentences(con, sents)
        stats["reocr" if src == "ocr-reocr" else "vlm"] += 1
        stats["words_added"] += len(boxes)
        log.info("reocr %s p%d: %d words / %d sentences (was %d)",
                 doc_id, page, len(boxes), len(sents), nw)
    manifest_path = out_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reocr"] = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stats": stats,
        "min_words": min_words,
        "upscale": upscale,
        "word_boxes": index_mod.count_word_boxes(con),
        "sentences": index_mod.count_sentences(con),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("reocr done: %s", json.dumps(stats))
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
    ap.add_argument("--device", default=None,
                    help="cuda | cpu | auto (default: config retriever.device or auto)")
    ap.add_argument("--dtype", default=None,
                    help="bfloat16 | float16 | float32 (default: config retriever.dtype)")
    ap.add_argument("--skip-ocr", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="GPU embed batch (auto-halved on CUDA OOM)")
    ap.add_argument("--upscale", type=float, default=2.0,
                    help="image upscale factor before Tesseract (Phase 5a)")
    ap.add_argument("--reocr", action="store_true",
                    help="re-OCR sparse pages WITHOUT re-embedding (Phase 5a)")
    ap.add_argument("--reocr-min-words", type=int, default=10,
                    help="pages with fewer words are re-OCR'd")
    ap.add_argument("--vlm-transcribe-empty", action="store_true",
                    default=None,
                    help="VLM-transcribe pages still at zero words after OCR "
                         "(default: ingest.vlm_transcribe_empty from config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.vlm_transcribe_empty is None:
        # Run 9 Phase 4: config-driven default (guideline s5 ingest block)
        args.vlm_transcribe_empty = bool(
            getattr(getattr(cfg, "ingest", None), "vlm_transcribe_empty", False))

    if args.reocr:
        manifest = run_reocr(
            args.out,
            lang=args.lang,
            min_words=args.reocr_min_words,
            upscale=args.upscale,
            vlm_transcribe_empty=args.vlm_transcribe_empty,
            vlm_model=cfg.reader.model,
            limit=args.limit,
        )
        print(json.dumps(manifest, indent=2))
        return

    manifest = run_ingest(
        args.input, args.out,
        dpi=args.dpi or cfg.render.dpi,
        lang=args.lang,
        embedder=args.embedder,
        model_id=args.model_id or cfg.retriever.model_id,
        device=args.device or getattr(cfg.retriever, "device", "auto"),
        dtype=args.dtype or getattr(cfg.retriever, "dtype", "bfloat16"),
        skip_ocr=args.skip_ocr,
        limit=args.limit,
        collection_full=cfg.qdrant.collection_full,
        collection_pooled=cfg.qdrant.collection_pooled,
        qdrant_mode=cfg.qdrant.mode,
        qdrant_path=cfg.qdrant.path,
        batch_size=args.batch_size,
        upscale=args.upscale,
        vlm_transcribe_empty=args.vlm_transcribe_empty,
        vlm_model=cfg.reader.model,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
