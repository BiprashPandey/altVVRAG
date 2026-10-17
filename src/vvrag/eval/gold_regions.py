"""Gold region derivation (Section 10.2): answer span -> word-box union.

No manual annotation: exact normalized match first, then fuzzy
(difflib ratio) fallback. Questions without a match are excluded from
LOCALIZATION metrics only (still used for answer metrics).
"""
from __future__ import annotations

import difflib

from vvrag.reader.claims import normalize_answer
from vvrag.schemas import WordBox


def _kept_words(words: list[WordBox]) -> list[tuple[WordBox, str]]:
    """Words that survive normalization (articles drop to "" and are skipped)."""
    norm_words = [normalize_answer(w.text) for w in words]
    return [(w, nw) for w, nw in zip(words, norm_words) if nw]


def derive_answer_words(
    answer: str, words: list[WordBox], fuzzy_threshold: float = 0.8
) -> tuple[list[WordBox], float]:
    """Find the best run of consecutive OCR words matching `answer`.

    Returns (covering_words, match_score). Exact normalized substring match
    scores 1.0. Otherwise a sliding-window fuzzy match over normalized words
    (window sizes around the answer length) handles OCR typos like
    "Rasta" vs "Rastra". Miss -> ([], 0.0).
    """
    target = normalize_answer(answer)
    if not target or not words:
        return [], 0.0
    kept = _kept_words(words)
    if not kept:
        return [], 0.0
    joined = " ".join(nw for _, nw in kept)
    if target in joined:
        start_char = joined.find(target)
        end_char = start_char + len(target)
        pos, span = 0, []
        for w, nw in kept:
            w_end = pos + len(nw)
            if w_end > start_char and pos < end_char:
                span.append(w)
            pos = w_end + 1
        if span:
            return span, 1.0
    # fuzzy: sliding window over kept words, best SequenceMatcher ratio wins
    n_tok = len(target.split())
    sizes = sorted({s for s in (n_tok - 1, n_tok, n_tok + 1) if s >= 1})
    best_span: list[WordBox] = []
    best_r = 0.0
    kept_nws = [nw for _, nw in kept]
    for size in sizes:
        if size > len(kept):
            continue
        for i in range(len(kept) - size + 1):
            cand = " ".join(kept_nws[i:i + size])
            r = difflib.SequenceMatcher(None, target, cand).ratio()
            if r > best_r:
                best_r = r
                best_span = [w for w, _ in kept[i:i + size]]
    if best_span and best_r >= fuzzy_threshold:
        return best_span, float(best_r)
    # last resort: legacy best single word -> its full line (keeps old recall)
    norm_words = [normalize_answer(w.text) for w in words]
    best, best_single = None, 0.0
    for w, nw in zip(words, norm_words):
        r = difflib.SequenceMatcher(None, target, nw).ratio()
        if r > best_single:
            best, best_single = w, r
    if best is not None and best_single >= fuzzy_threshold:
        line = [w for w in words if w.block_id == best.block_id and w.line_id == best.line_id]
        return (line or [best]), float(best_single)
    return [], 0.0


def derive_gold_box(
    answer: str, words: list[WordBox], fuzzy_threshold: float = 0.8
) -> tuple[int, int, int, int] | None:
    span, _ = derive_answer_words(answer, words, fuzzy_threshold)
    if not span:
        return None
    return _union([w.box for w in span])


def _union(boxes: list[tuple]) -> tuple[int, int, int, int]:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))
