"""Text-in-box -> crop_text (Section 7.2 step 8).

Collect WordBoxes whose centers fall inside the evidence boxes, join in
reading order (block, line, x). If no word boxes, run Tesseract on the crop;
if still empty, crop_has_text = False (NLI not applicable downstream).
"""
from __future__ import annotations

from pathlib import Path

from vvrag.logging_utils import get_logger
from vvrag.schemas import WordBox

log = get_logger(__name__)


def words_in_boxes(
    words: list[WordBox], boxes: list[tuple[int, int, int, int]]
) -> list[WordBox]:
    out: list[WordBox] = []
    for w in words:
        cx, cy = (w.box[0] + w.box[2]) / 2, (w.box[1] + w.box[3]) / 2
        if any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in boxes):
            out.append(w)
    out.sort(key=lambda w: (w.block_id, w.line_id, w.box[0]))
    return out


def group_lines(words: list[WordBox]) -> list[list[WordBox]]:
    """Cluster words into visual lines by y-center (tolerance = half median height)."""
    if not words:
        return []
    heights = sorted(w.box[3] - w.box[1] for w in words)
    tol = max(2.0, float(heights[len(heights) // 2]) / 2)
    by_y = sorted(words, key=lambda w: (w.box[1] + w.box[3]) / 2)
    lines: list[list[WordBox]] = []
    means: list[float] = []
    for w in by_y:
        yc = (w.box[1] + w.box[3]) / 2
        if lines and abs(yc - means[-1]) <= tol:
            lines[-1].append(w)
            means[-1] = sum((x.box[1] + x.box[3]) / 2 for x in lines[-1]) / len(lines[-1])
        else:
            lines.append([w])
            means.append(yc)
    return [[w for w in sorted(line, key=lambda w: w.box[0])] for line in lines]


def join_words(words: list[WordBox]) -> str:
    """Join in visual reading order: lines by y, words by x."""
    return " ".join(w.text for line in group_lines(words) for w in line).strip()


def extract_crop_text(
    words: list[WordBox],
    boxes: list[tuple[int, int, int, int]],
    crop_image_path: str | Path | None = None,
    lang: str = "eng",
) -> tuple[str, bool]:
    """Returns (crop_text, crop_has_text).

    Lines are expanded to full visual lines: any line with >=1 word center
    inside the boxes contributes all its words (complete sentences for NLI).
    """
    inside_ids = {id(w) for w in words_in_boxes(words, boxes)}
    if inside_ids:
        kept = [w for line in group_lines(words)
                for w in line if any(id(x) in inside_ids for x in line)]
        return join_words(kept), True
    if crop_image_path is not None:
        try:
            from vvrag.ingest.ocr import ocr_page

            p = Path(str(crop_image_path))
            extra = ocr_page(p, doc_id="crop", page=0, lang=lang)
            if extra:
                return join_words(extra), True
        except Exception as e:
            log.warning("crop OCR failed: %s", e)
    return "", False
