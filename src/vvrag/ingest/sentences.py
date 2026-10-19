"""Sentence-level evidence units (suggestion #1: cite sentences, not words).

Splits a page's WordBoxes into numbered sentences, each with ONE bounding
box (multi-line sentences merge their line boxes into a single union box).
The reader cites sentence ids; final display boxes come from these units.

Segmentation rules (pure functions over WordBoxes, works for PyMuPDF text
layer AND Tesseract OCR output — both carry block_id/line_id):
- reading order = (block_id, line_id, x0); falls back to y-cluster lines
  when block/line ids are degenerate (all zeros);
- sentence break on words ending in . ? ! (closing quotes/brackets skipped)
  or on a new block;
- LIST items (1. 2) a) - bullet, each start a new unit (#7);
- TABLE cells: an intra-line x-gap > gap_factor * median word height splits
  the unit, so cells never bleed into the next column (#7);
- headings/paragraph breaks: a vertical jump > v_factor * median height
  within the same block starts a new unit (#7).

Sentence boxes are stored in PIXEL space; `box_frac` helpers convert to
page fractions (0-1) for zoom-stable viewer overlays (#9).
"""
from __future__ import annotations

import re

from vvrag.logging_utils import get_logger
from vvrag.schemas import SentenceSpan, WordBox

log = get_logger(__name__)

_SENT_END = re.compile(r'[.!?]["\'\u201d\u2019)\]]*$')
_LIST_MARKER = re.compile(r'^(\d+[.)]|[a-zA-Z][.)]|[-\u2022\u2013\u2014*])$')
DEFAULT_GAP_FACTOR = 2.5
DEFAULT_VGAP_FACTOR = 3.0


def _reading_order(words: list[WordBox]) -> list[WordBox]:
    """Sort into visual reading order.

    Prefers extractor block/line ids (PyMuPDF + Tesseract both provide them);
    when those are degenerate (single block/line), clusters by y-center so
    plain word lists still segment correctly.
    """
    if not words:
        return []
    has_structure = len({(w.block_id, w.line_id) for w in words}) > 1
    if has_structure:
        return sorted(words, key=lambda w: (w.block_id, w.line_id, w.box[0]))
    # y-cluster fallback (mirrors ground.text_in_box.group_lines)
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
    out: list[WordBox] = []
    for line in lines:
        out.extend(sorted(line, key=lambda w: w.box[0]))
    return out


def _median_height(words: list[WordBox]) -> float:
    heights = sorted(w.box[3] - w.box[1] for w in words)
    return float(heights[len(heights) // 2]) if heights else 10.0


def _ends_sentence(text: str) -> bool:
    return bool(_SENT_END.search(text.strip()))


def _is_list_marker(text: str) -> bool:
    return bool(_LIST_MARKER.match(text.strip()))


def build_sentences(
    words: list[WordBox],
    gap_factor: float = DEFAULT_GAP_FACTOR,
    vgap_factor: float = DEFAULT_VGAP_FACTOR,
) -> tuple[list[SentenceSpan], list[int]]:
    """Segment a page's words into sentences.

    Returns (sentences, word_to_sent) where word_to[i] is the sentence id
    for words[i] (input order). Sentence boxes are pixel unions of member
    words; ids are 0-based per page.
    """
    if not words:
        return [], []
    ordered = _reading_order(words)
    med_h = _median_height(words)
    x_gap = gap_factor * med_h
    y_gap = vgap_factor * med_h

    groups: list[list[WordBox]] = []
    cur: list[WordBox] = []
    for w in ordered:
        if cur:
            new_block = w.block_id != cur[-1].block_id
            same_line = w.line_id == cur[-1].line_id and w.block_id == cur[-1].block_id
            xgap = w.box[0] - cur[-1].box[2] if same_line else 0.0
            yjump = w.box[1] - cur[-1].box[3]
            break_now = (
                new_block
                or (_ends_sentence(cur[-1].text)
                    and not _is_list_marker(cur[-1].text))  # "1." is a marker, not a period
                or _is_list_marker(w.text)               # list item starts fresh
                or (same_line and xgap > x_gap)          # table cell boundary
                or yjump > y_gap                         # heading / paragraph jump
            )
            if break_now:
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)

    sentences: list[SentenceSpan] = []
    word_to_sent: dict[tuple, int] = {}
    for gid, group in enumerate(groups):
        text = " ".join(w.text for w in group).strip()
        box = (
            min(w.box[0] for w in group),
            min(w.box[1] for w in group),
            max(w.box[2] for w in group),
            max(w.box[3] for w in group),
        )
        sentences.append(SentenceSpan(
            doc_id=group[0].doc_id,
            page=group[0].page,
            sent_id=gid,
            text=text,
            box=box,
            block_id=group[0].block_id,
        ))
        for w in group:
            word_to_sent[(w.text, w.box, w.block_id, w.line_id)] = gid

    word_to = [
        word_to_sent.get((w.text, w.box, w.block_id, w.line_id), -1) for w in words
    ]
    return sentences, word_to


def sentences_from_page(
    con, doc_id: str, page: int,
) -> tuple[list[SentenceSpan], list[WordBox]]:
    """Load sentences (and words) for a page; build on the fly if the index
    predates sentence storage (back-compat with existing data/index)."""
    rows_w = con.execute(
        "SELECT text,x0,y0,x1,y1,block_id,line_id,sent_id FROM word_boxes "
        "WHERE doc_id=? AND page=?", (doc_id, page)).fetchall()
    words = [WordBox(doc_id, page, r[0], (r[1], r[2], r[3], r[4]), r[5], r[6],
                     r[7] if r[7] is not None else -1) for r in rows_w]
    try:
        rows = con.execute(
            "SELECT sent_id, text, x0, y0, x1, y1, block_id, source FROM sentences "
            "WHERE doc_id=? AND page=? ORDER BY sent_id",
            (doc_id, page),
        ).fetchall()
    except Exception:  # noqa: BLE001 - legacy DB without the source column
        rows = con.execute(
            "SELECT sent_id, text, x0, y0, x1, y1, block_id FROM sentences "
            "WHERE doc_id=? AND page=? ORDER BY sent_id",
            (doc_id, page),
        ).fetchall()
        rows = [(*r, "textlayer") for r in rows]
    if rows:
        sents = [SentenceSpan(doc_id=doc_id, page=page, sent_id=r[0], text=r[1],
                              box=(r[2], r[3], r[4], r[5]), block_id=r[6],
                              source=r[7] or "textlayer") for r in rows]
        return sents, words
    sents, _ = build_sentences(words)
    return sents, words


def sentence_ids_for_answer(
    answer: str,
    sentences: list[SentenceSpan],
    fuzzy_threshold: float = 0.8,
) -> list[int]:
    """Sentence ids whose text contains the answer (P3: fuzzy gold sids).

    Exact normalized substring first; otherwise token-F1 >= threshold (after
    splitting alpha/digit runs so "Col3" ~ "Col 3"), which tolerates OCR typos
    instead of leaving gold sentence ids empty.
    """
    import re as _re
    from vvrag.reader.claims import normalize_answer, token_f1

    def _alnum(s: str) -> str:
        return _re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", s)

    target = _alnum(normalize_answer(answer))
    if not target:
        return []
    ids: list[int] = []
    for s in sentences:
        nt = _alnum(normalize_answer(s.text))
        if not nt:
            continue
        if target in nt or target.split() and all(
                nt.split().count(t) >= target.split().count(t) for t in set(target.split())):
            ids.append(s.sent_id)
        elif token_f1(target, nt) >= fuzzy_threshold:
            ids.append(s.sent_id)
    return ids


def box_frac(
    box: tuple[int, int, int, int], page_width: int, page_height: int,
) -> tuple[float, float, float, float]:
    """Pixel box -> page-fraction box (0-1), zoom-stable for the viewer (#9)."""
    if page_width <= 0 or page_height <= 0:
        raise ValueError("page dimensions must be positive")
    x0, y0, x1, y1 = box
    return (
        max(0.0, min(1.0, x0 / page_width)),
        max(0.0, min(1.0, y0 / page_height)),
        max(0.0, min(1.0, x1 / page_width)),
        max(0.0, min(1.0, y1 / page_height)),
    )
