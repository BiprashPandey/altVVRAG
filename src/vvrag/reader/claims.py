"""Claim parsing + answer normalization (Section 7.3).

Normalization: lowercase, strip punctuation, drop articles (a/an/the),
normalize thousand-separators (1,000 -> 1000), collapse whitespace.
Used for EM/F1 and self-consistency agreement (C2).
"""
from __future__ import annotations

import json
import re
import string

from vvrag.schemas import Claim

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_WS = re.compile(r"\s+")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = _THOUSANDS.sub("", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return _WS.sub(" ", s).strip()


def token_f1(pred: str, gold: str) -> float:
    pt, gt = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    overlap = 0
    counts: dict[str, int] = {}
    for t in gt:
        counts[t] = counts.get(t, 0) + 1
    for t in pt:
        if counts.get(t, 0) > 0:
            counts[t] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(pt), overlap / len(gt)
    return 2 * prec * rec / (prec + rec)


def answers_match(a: str, b: str, f1_threshold: float = 0.8) -> bool:
    if normalize_answer(a) == normalize_answer(b):
        return True
    return token_f1(a, b) >= f1_threshold


def parse_reader_output(raw: str, max_claims: int = 5) -> tuple[str, list[Claim], bool]:
    """Returns (answer, claims, ok). ok=False when output was not valid JSON."""
    answer, _quote, _sids, claims, ok = parse_reader_output_full(raw, max_claims)
    return answer, claims, ok


def parse_reader_output_full(
    raw: str, max_claims: int = 5
) -> tuple[str, str, list[int], list[Claim], bool]:
    """Returns (answer, quote, sentence_ids, claims, ok).

    Suggestion #1: the reader cites numbered sentences. `sentence_ids` is the
    top-level answer citation; each claim may also carry
    `evidence_sentence_ids` (legacy word-id fields are ignored).
    """
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("```")).strip()
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        return "UNANSWERABLE", "", [], [], False
    answer = str(obj.get("answer", "UNANSWERABLE"))
    quote = str(obj.get("quote", "") or "")
    top_ids = _parse_int_list(obj.get("sentence_ids"))
    claims: list[Claim] = []
    for c in obj.get("claims", [])[:max_claims]:
        if isinstance(c, dict) and str(c.get("text", "")).strip():
            cids = _parse_int_list(c.get("evidence_sentence_ids", c.get("sentence_ids")))
            claims.append(Claim(text=str(c["text"]).strip(),
                                answer_span=c.get("answer_span"),
                                evidence_sentence_ids=cids))
    if not top_ids and claims:
        # fall back to the first claim's citations
        top_ids = claims[0].evidence_sentence_ids or []
    return answer, quote, top_ids, claims, True


def _parse_int_list(value) -> list[int] | None:
    if isinstance(value, list) and value:
        try:
            return [int(x) for x in value]
        except (TypeError, ValueError):
            return None
    return None


def validate_sentence_ids(ids: list[int] | None, n_sentences: int) -> tuple[bool, bool]:
    """Check reader-returned sentence ids against the page's sentence units.

    Returns (in_range, non_empty). Sentence ids need NOT be consecutive (a
    multi-sentence answer cites several units). Out-of-range ids -> (False, False).
    """
    if not ids or n_sentences <= 0:
        return False, False
    try:
        ints = [int(x) for x in ids]
    except (TypeError, ValueError):
        return False, False
    in_range = all(0 <= i < n_sentences for i in ints)
    return bool(in_range), bool(in_range and len(ints) > 0)


# Backwards-compatible alias for old imports/tests (word ids -> sentence ids).
def validate_evidence_ids(ids: list[int] | None, n_units: int) -> tuple[bool, bool]:
    return validate_sentence_ids(ids, n_units)
