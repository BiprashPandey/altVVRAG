"""Gold region derivation (Section 10.2): answer span -> word-box union.

No manual annotation: exact normalized match first, then fuzzy
(difflib ratio) fallback. Questions without a match are excluded from
LOCALIZATION metrics only (still used for answer metrics).
"""
from __future__ import annotations

import difflib

from vvrag.reader.claims import normalize_answer
from vvrag.schemas import WordBox


def derive_gold_box(
    answer: str, words: list[WordBox], fuzzy_threshold: float = 0.8
) -> tuple[int, int, int, int] | None:
    target = normalize_answer(answer)
    if not target:
        return None
    # exact: contiguous word run whose normalized join contains the target.
    # NOTE: words normalizing to "" (removed articles like "the") are dropped
    # BEFORE joining — otherwise their empty slots leave double spaces and
    # any span crossing an article never matches (found 2026-10-15 on a real
    # OCR page: "... about [the] Department's ..." missed).
    norm_words = [normalize_answer(w.text) for w in words]
    kept = [(w, nw) for w, nw in zip(words, norm_words) if nw]
    joined = " ".join(nw for _, nw in kept)
    if target in joined:
        # map back: find minimal word span covering the match
        start_char = joined.find(target)
        end_char = start_char + len(target)
        pos, span = 0, []
        for w, nw in kept:
            w_end = pos + len(nw)
            if w_end > start_char and pos < end_char:
                span.append(w)
            pos = w_end + 1
        if span:
            return _union([w.box for w in span])
    # fuzzy: best single word, then expand to its line
    best, best_r = None, 0.0
    for w, nw in zip(words, norm_words):
        r = difflib.SequenceMatcher(None, target, nw).ratio()
        if r > best_r:
            best, best_r = w, r
    if best is not None and best_r >= fuzzy_threshold:
        line = [w for w in words if w.block_id == best.block_id and w.line_id == best.line_id]
        return _union([w.box for w in line] or [best.box])
    return None


def _union(boxes: list[tuple]) -> tuple[int, int, int, int]:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))
