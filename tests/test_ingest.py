"""Ingest smoke: render + fake-embed + pool + index on 3 small PDFs (CPU).

The real ColQwen2 GPU run happens on Colab (notebooks/ingest_colab.ipynb);
this exercises everything except the model weights.
"""
import pymupdf

from vvrag.ingest import index as index_mod
from vvrag.ingest import render as render_mod
from vvrag.ingest.ocr import OCRError, needs_ocr, ocr_page
from vvrag.ingest.run_ingest import run_ingest


def test_ocr_missing_backend_raises_clean_error(tmp_path):
    assert needs_ocr(0) and not needs_ocr(50)
    try:
        ocr_page(tmp_path / "nope.png", "d", 0)
    except OCRError:
        pass  # expected on machines without tesseract (Colab installs it)


def _make_pdf(path, lines, pages=2):
    with pymupdf.open() as doc:
        for p in range(pages):
            page = doc.new_page()
            y = 72
            for ln in lines:
                page.insert_text((72, y + p * 10), f"{ln} p{p}")
                y += 20
        doc.save(path)


def test_render_word_boxes_scaled(tmp_path):
    pdf = tmp_path / "a.pdf"
    _make_pdf(pdf, ["Hello world, this is a test document with many words " * 2], pages=1)
    metas, words = render_mod.render_pdf(pdf, tmp_path / "pages", "a", dpi=150)
    assert len(metas) == 1
    assert metas[0].width > 0 and metas[0].has_text_layer
    assert len(words) >= 10
    x0, y0, x1, y1 = words[0].box
    assert 0 <= x0 < x1 <= metas[0].width and 0 <= y0 < y1 <= metas[0].height


def test_ingest_three_pdfs_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("docA", "docB", "docC"):
        _make_pdf(raw / f"{name}.pdf",
                  ["The quick brown fox jumps over charts and tables " * 3], pages=2)

    out = tmp_path / "index"
    m1 = run_ingest(raw, out, dpi=100, embedder="fake", skip_ocr=True,
                    qdrant_path=str(out / "qdrant"))
    assert m1["stats"]["pages"] == 6, m1
    assert m1["sqlite_pages"] == 6
    assert m1["qdrant_full"] == 6
    # pooled: 6 pages x (8 rows + 6 cols + 1 global) = 90
    assert m1["qdrant_pooled"] == 6 * (8 + 6 + 1), m1
    assert m1["word_boxes"] > 0

    # Re-run: everything skipped (resumable/idempotent).
    m2 = run_ingest(raw, out, dpi=100, embedder="fake", skip_ocr=True,
                    qdrant_path=str(out / "qdrant"))
    assert m2["stats"]["pages"] == 0 and m2["stats"]["skipped"] == 6, m2
    con = index_mod.connect_sqlite(out / "meta.db")
    assert index_mod.count_pages(con) == 6
