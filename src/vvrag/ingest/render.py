"""PDF/image rendering + word-box extraction (Section 6.1).

Uses PyMuPDF. Renders each PDF page at render.dpi to PNG. For PDFs with a
text layer, extracts word boxes via page.get_text("words") and scales from
PDF points to pixels (scale = dpi / 72). Image inputs are loaded directly.
"""
from __future__ import annotations

from pathlib import Path

import pymupdf  # PyMuPDF >= 1.23 (installed: check `pip show pymupdf`)

from vvrag.logging_utils import get_logger
from vvrag.schemas import PageMeta, WordBox

log = get_logger(__name__)

POINTS_PER_INCH = 72.0
MIN_WORDS_FOR_TEXT_LAYER = 10


def _scale_point_to_pixel(v: float, dpi: int) -> int:
    return int(round(v * dpi / POINTS_PER_INCH))


def render_pdf(
    pdf_path: str | Path,
    out_dir: str | Path,
    doc_id: str | None = None,
    dpi: int = 150,
) -> tuple[list[PageMeta], list[WordBox]]:
    """Render all pages of a PDF. Returns (page_metas, word_boxes)."""
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_id = doc_id or pdf_path.stem

    pages: list[PageMeta] = []
    words: list[WordBox] = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi)
            img_path = out_dir / f"{doc_id}_p{i}.png"
            pix.save(str(img_path))

            raw_words = page.get_text("words")  # (x0,y0,x1,y1,word,block,line,word_no)
            page_words: list[WordBox] = []
            for w in raw_words:
                x0, y0, x1, y1, text, block, line, _wno = w
                text = str(text).strip()
                if not text:
                    continue
                page_words.append(
                    WordBox(
                        doc_id=doc_id,
                        page=i,
                        text=text,
                        box=(
                            _scale_point_to_pixel(float(x0), dpi),
                            _scale_point_to_pixel(float(y0), dpi),
                            _scale_point_to_pixel(float(x1), dpi),
                            _scale_point_to_pixel(float(y1), dpi),
                        ),
                        block_id=int(block),
                        line_id=int(line),
                    )
                )
            words.extend(page_words)
            pages.append(
                PageMeta(
                    doc_id=doc_id,
                    page=i,
                    width=pix.width,
                    height=pix.height,
                    dpi=dpi,
                    has_text_layer=len(page_words) >= MIN_WORDS_FOR_TEXT_LAYER,
                    image_path=str(img_path),
                )
            )
    log.info("rendered %s: %d pages (%s)", pdf_path, len(pages), doc_id)
    return pages, words


def load_image_as_page(
    image_path: str | Path,
    out_dir: str | Path,
    doc_id: str | None = None,
    page: int = 0,
    dpi: int = 150,
) -> PageMeta:
    """Use a page image directly (no text layer). Copies into out_dir for uniformity."""
    from PIL import Image

    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_id = doc_id or image_path.stem
    dest = out_dir / f"{doc_id}_p{page}.png"
    if image_path.resolve() != dest.resolve():
        with Image.open(image_path) as im:
            im.save(dest)
        with Image.open(dest) as im:
            w, h = im.size
    else:
        with Image.open(dest) as im:
            w, h = im.size
    return PageMeta(
        doc_id=doc_id,
        page=page,
        width=w,
        height=h,
        dpi=dpi,
        has_text_layer=False,
        image_path=str(dest),
    )
