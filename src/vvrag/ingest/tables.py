"""Table-aware citation units (Run 9, Phase 4).

Problem (Runs 6-8): the sentence segmenter reads a financial table row like
"Revenue 12.5 12.0 +4.2%" -- when the reader quotes the single cell "12.5"
(the answer), the cited *sentence* is a cell fragment and the quote check
fails, or the reader cites the wrong split row entirely. Run 7's dominant
false-abstention source was exactly this table segmentation.

Fix (deterministic, no model): detect table-like blocks from the OCR word
geometry and append extra citation units for them:

- one unit per ROW: cell texts joined with " | "
  ("Revenue | 12.5 | 12.0 | +4.2%") -- preserves context;
- one unit per CELL: the cell text alone ("12.5") -- makes short quotes match
  verbatim, which is what the quote check wants.

Units carry `source="table"` and never overlap text sentences (new ids after
the last text id). `sentences_from_page(..., include_tables=True)` appends
them on the fly, so existing indexes benefit without re-ingestion; the stored
sentences table keeps only text units for back-compat.
"""
from __future__ import annotations

import re
from collections import defaultdict

from vvrag.ingest.sentences import _reading_order  # reuse the same order
from vvrag.schemas import SentenceSpan, WordBox

DEFAULT_MIN_LINES = 3
DEFAULT_MIN_WORDS_PER_LINE = 2
DEFAULT_COL_GAP_FRAC = 0.03  # a gap wider than 3% of the page width splits
_NUMBER = re.compile(r"^\d+(?:[.,]\d+)?$")
_SCALE = re.compile(
    r"^(?:%|percent|percentage|bn|mn|tn|billion|million|trillion|thousand|k)$",
    re.I)
_ATTACHED_NUMBER = re.compile(
    r"^\d+(?:[.,]\d+)?(?:%|bn|mn|tn|billion|million|trillion|thousand|k)$",
    re.I)


def _split_cells(line: list[WordBox], page_width: int,
                 col_gap_frac: float = DEFAULT_COL_GAP_FRAC,
                 ) -> list[list[WordBox]]:
    """Split one line's words into cells at large horizontal gaps."""
    if not line:
        return []
    gap_min = max(col_gap_frac * max(1, page_width), 8.0)
    cells: list[list[WordBox]] = [[line[0]]]
    for w in line[1:]:
        if w.box[0] - cells[-1][-1].box[2] > gap_min:
            cells.append([w])
        else:
            cells[-1].append(w)
    return cells


def _is_table_block(lines: list[list[WordBox]], page_width: int) -> bool:
    """A block is table-like when most lines hold >= 3 cells-with-gaps..."""
    if len(lines) < DEFAULT_MIN_LINES:
        return False
    multi = 0
    for line in lines:
        if len(line) < DEFAULT_MIN_WORDS_PER_LINE:
            continue
        if len(_split_cells(line, page_width)) >= 2:
            multi += 1
    return multi >= max(DEFAULT_MIN_LINES, int(0.6 * len(lines)))


def _union_box(words: list[WordBox]) -> tuple[int, int, int, int]:
    return (min(w.box[0] for w in words), min(w.box[1] for w in words),
            max(w.box[2] for w in words), max(w.box[3] for w in words))


def table_units(words: list[WordBox], page_width: int,
                first_id: int = 0,
                min_lines: int = DEFAULT_MIN_LINES
                ) -> list[SentenceSpan]:
    """Extra citation units for table-like blocks (rows first, then cells).

    Returns [] when the page has no table-like block. Ids start at `first_id`.
    Deterministic: geometry only, no text heuristics beyond the split gaps.
    """
    if not words:
        return []
    by_block: dict[int, list[WordBox]] = defaultdict(list)
    for w in words:
        by_block[w.block_id].append(w)
    units: list[SentenceSpan] = []
    sid = first_id
    doc_id = words[0].doc_id
    page = words[0].page
    for block_id in sorted(by_block):
        bwords = by_block[block_id]
        by_line: dict[int, list[WordBox]] = defaultdict(list)
        for w in bwords:
            by_line[w.line_id].append(w)
        ordered_lines = [sorted(by_line[lid], key=lambda w: w.box[0])
                         for lid in sorted(by_line)]
        # a single "line" without line_id metadata: derive lines by y-clusters
        if len(ordered_lines) == 1 and len(ordered_lines[0]) > 12:
            ordered_lines = _cluster_lines(ordered_lines[0])
        ordered_lines = [ln for ln in ordered_lines if ln]
        if (len(ordered_lines) < min_lines
                or not _is_table_block(ordered_lines, page_width)):
            continue
        for line in ordered_lines:
            cells = _split_cells(line, page_width)
            if not cells:
                continue
            row_text = " | ".join(" ".join(w.text for w in c) for c in cells)
            units.append(SentenceSpan(doc_id=doc_id, page=page, sent_id=sid,
                                      text=row_text, box=_union_box(line),
                                      block_id=block_id, source="table"))
            sid += 1
            if len(cells) >= 2:
                for c in cells:
                    ctext = " ".join(w.text for w in c).strip()
                    if not ctext:
                        continue
                    units.append(SentenceSpan(doc_id=doc_id, page=page,
                                              sent_id=sid, text=ctext,
                                              box=_union_box(c),
                                              block_id=block_id,
                                              source="table"))
                    sid += 1
    return units


def number_unit_units(words: list[WordBox], page_width: int = 0,
                      first_id: int = 0) -> list[SentenceSpan]:
    """Deterministic number+unit citation units ("12.5 bn", "15.0%").

    Table detection is geometry-based and misses OCR layouts where one visual
    row is split across multiple blocks/line ids. A reader's answer quote is
    often just the number and its scale unit, so expose those pairs as
    first-class citation units too. Additive and conservative: only adjacent
    number/unit tokens on the same visual line, or a single attached token.
    """
    if not words:
        return []
    units: list[SentenceSpan] = []
    seen: set[tuple] = set()
    sid = first_id
    gap_max = max(8.0, 0.08 * max(1, page_width))

    def _same_line(a: WordBox, b: WordBox) -> bool:
        h = min(a.box[3] - a.box[1], b.box[3] - b.box[1])
        overlap = min(a.box[3], b.box[3]) - max(a.box[1], b.box[1])
        return h > 0 and overlap >= 0.5 * h

    for w in words:
        text = w.text.strip()
        partner = None
        if _ATTACHED_NUMBER.match(text):
            unit_text, box = text, w.box
        elif _NUMBER.match(text):
            # Pair only with the nearest token to the right on the same visual
            # line, and only if that token is itself a scale unit. This avoids
            # pairing across an intervening column ("12.5 | 7.5 bn").
            right = [v for v in words
                     if v is not w and _same_line(w, v)
                     and v.box[0] >= w.box[2]]
            if not right:
                continue
            partner = min(right, key=lambda v: v.box[0] - w.box[2])
            gap = partner.box[0] - w.box[2]
            if gap > gap_max or not _SCALE.match(partner.text.strip()):
                continue
            unit_text = f"{text} {partner.text.strip()}"
            box = _union_box([w, partner])
        else:
            continue
        key = (unit_text.lower(),
               tuple(round(x) for x in box))
        if key in seen:
            continue
        seen.add(key)
        units.append(SentenceSpan(doc_id=w.doc_id, page=w.page,
                                  sent_id=sid, text=unit_text, box=box,
                                  block_id=w.block_id, source="table"))
        sid += 1
    return units


def build_sentences_with_tables(words: list[WordBox], page_width: int
                                ) -> list[SentenceSpan]:
    """Sentence units + table/number-unit units (ingestion convenience)."""
    from vvrag.ingest.sentences import build_sentences

    sents, _ = build_sentences(words)
    first = (max((s.sent_id for s in sents), default=-1) + 1)
    extra = table_units(words, page_width, first_id=first)
    extra += number_unit_units(words, page_width,
                               first_id=first + len(extra))
    return sents + extra


def _cluster_lines(words: list[WordBox]) -> list[list[WordBox]]:
    """Fallback line clustering by y when line_id is unavailable."""
    ordered = sorted(words, key=lambda w: (w.box[1], w.box[0]))
    heights = sorted(w.box[3] - w.box[1] for w in ordered)
    med_h = heights[len(heights) // 2] if heights else 10
    lines: list[list[WordBox]] = []
    for w in ordered:
        if lines and abs(w.box[1] - lines[-1][0].box[1]) <= 0.6 * med_h:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [_reading_order(ln) for ln in lines]
