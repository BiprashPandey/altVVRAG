"""Tesseract OCR fallback (Section 6.2). Runs only when a page has no text layer."""
from __future__ import annotations

from pathlib import Path

from vvrag.logging_utils import get_logger
from vvrag.schemas import PageMeta, WordBox

log = get_logger(__name__)


class OCRError(RuntimeError):
    pass


def find_tesseract() -> str | None:
    """Locate the tesseract binary: PATH, TESSERACT_CMD env, or common
    Windows install locations (winget installs per-user)."""
    import os
    import shutil

    if os.environ.get("TESSERACT_CMD"):
        return os.environ["TESSERACT_CMD"]
    which = shutil.which("tesseract")
    if which:
        return which
    cands = [
        os.path.expandvars("%LOCALAPPDATA%/Programs/Tesseract-OCR/tesseract.exe"),
        "C:/Program Files/Tesseract-OCR/tesseract.exe",
    ]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def tesseract_available() -> bool:
    return find_tesseract() is not None


def needs_ocr(n_words: int, min_words: int = 10) -> bool:
    return n_words < min_words


def ocr_page(
    image_path: str | Path,
    doc_id: str,
    page: int,
    lang: str = "eng",
    min_conf: int = 30,
) -> list[WordBox]:
    """Run Tesseract on a page image -> WordBoxes. Raises OCRError if unavailable."""
    try:
        import pytesseract
        from PIL import Image

        exe = find_tesseract()
        if exe:
            pytesseract.pytesseract.tesseract_cmd = exe
    except ImportError as e:
        raise OCRError(f"pytesseract/Pillow not installed: {e}") from e

    try:
        with Image.open(image_path) as im:
            width, height = im.size
        data = pytesseract.image_to_data(
            str(image_path), lang=lang, output_type=pytesseract.Output.DICT
        )
    except Exception as e:
        raise OCRError(
            f"Tesseract failed on {image_path} (is the tesseract binary installed? "
            f"Windows: install via UB Mannheim build and set TESSDATA_PREFIX): {e}"
        ) from e

    boxes: list[WordBox] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = str(data["text"][i]).strip()
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            continue
        if not text or conf < 0 or conf < min_conf:
            continue
        boxes.append(
            WordBox(
                doc_id=doc_id,
                page=page,
                text=text,
                box=(
                    int(data["left"][i]),
                    int(data["top"][i]),
                    int(data["left"][i]) + int(data["width"][i]),
                    int(data["top"][i]) + int(data["height"][i]),
                ),
                block_id=int(data["block_num"][i]),
                line_id=int(data["line_num"][i]),
            )
        )
    log.info("OCR %s p%d: %d words (lang=%s)", doc_id, page, len(boxes), lang)
    return boxes


def page_has_text_layer(meta: PageMeta, n_words: int, min_words: int = 10) -> bool:
    return meta.has_text_layer or n_words >= min_words
