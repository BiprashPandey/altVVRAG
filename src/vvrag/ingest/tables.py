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

from collections import defaultdict

from vvrag.ingest.sentences import _reading_order  # reuse the same order
from vvrag.schemas import SentenceSpan, WordBox

DEFAULT_MIN_LINES = 3
DEFAULT_MIN_WORDS_PER_LINE = 2
DEFAULT_COL_GAP_FRAC = 0.03  # a gap wider than 3% of the page width splits


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


def build_sentences_with_tables(words: list[WordBox], page_width: int
                                ) -> list[SentenceSpan]:
    """Sentence units + table units (run 9 ingestion convenience)."""
    from vvrag.ingest.sentences import build_sentences

    sents, _ = build_sentences(words)
    first = (max((s.sent_id for s in sents), default=-1) + 1)
    return sents + table_units(words, page_width, first_id=first)


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
