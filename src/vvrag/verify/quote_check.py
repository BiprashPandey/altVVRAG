"""Deterministic quote check (suggestion #4): the reader's quote must actually
appear in the cited sentences. No model involved.

- Exact (normalized) substring match -> 1.0.
- Otherwise fuzzy: best token-F1 between the quote and each cited sentence
  (or the joined citation block). Pass when >= threshold (default 0.8).
- Citation check: the cited sentence ids must exist on the page.

Used two ways:
- as a hard verifier signal (`quote_match` feature for the gate), and
- as a retry trigger: a failing quote is re-asked once, then abstained.
"""
from __future__ import annotations

import re

from vvrag.reader.claims import normalize_answer, token_f1

DEFAULT_QUOTE_THRESHOLD = 0.8

_ALNUM = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")


def _normalize_ocr(text: str) -> str:
    """OCR-tolerant normalization: drop spaces (col 3 == col3 == c o l 3 -> col3).

    Used only for quote verification so a reader quote with different spacing
    than the OCR'd sentence still matches verbatim words. Deterministic, no model.
    """
    return _ALNUM.sub("", normalize_answer(text).replace(" ", ""))


def quote_match_score(quote: str, cited_texts: list[str],
                      ocr_normalize: bool = False) -> float:
    """Best support score in [0, 1] of `quote` within the cited sentence texts."""
    q = normalize_answer(quote or "")
    if not q or not cited_texts:
        return 0.0
    if ocr_normalize:
        qo = _normalize_ocr(quote)
        if qo:
            if any(qo in _normalize_ocr(t) for t in cited_texts):
                return 1.0
    joined = normalize_answer(" ".join(cited_texts))
    if q in joined:
        return 1.0
    best = 0.0
    for text in cited_texts:
        nt = normalize_answer(text)
        if not nt:
            continue
        if q in nt:
            return 1.0
        best = max(best, token_f1(q, nt))
    return float(best)


def quote_matches(
    quote: str, cited_texts: list[str],
    threshold: float = DEFAULT_QUOTE_THRESHOLD,
    ocr_normalize: bool = False,
) -> bool:
    return quote_match_score(quote, cited_texts, ocr_normalize=ocr_normalize) >= threshold


def citation_texts(
    sentence_ids: list[int] | None,
    sentences: list,  # list[SentenceSpan]
) -> tuple[list[str], bool]:
    """Resolve sentence ids -> texts. Returns (texts, all_valid).

    `all_valid` is False when any id is out of range (made-up citation).
    """
    if not sentence_ids or not sentences:
        return [], False
    by_id = {s.sent_id: s.text for s in sentences}
    texts: list[str] = []
    all_valid = True
    for sid in dict.fromkeys(int(x) for x in sentence_ids):
        if sid in by_id:
            texts.append(by_id[sid])
        else:
            all_valid = False
    return texts, all_valid and bool(texts)


def build_quote_retry_prompt(original_prompt: str, quote: str) -> str:
    """One-shot corrective prompt after a failed quote check."""
    correction = (
        f"Your previous quote {quote!r} was NOT found verbatim in the sentences you cited. "
        "Re-answer and make sure the \"quote\" is copied word-for-word from a sentence shown "
        "above, and that \"sentence_ids\" lists exactly the sentences containing that quote. "
        "If no sentence supports an answer, return \"answer\": \"UNANSWERABLE\"."
    )
    return original_prompt + "\n\n" + correction
