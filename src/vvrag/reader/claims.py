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
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("```")).strip()
    try:
        obj = json.loads(text)
        answer = str(obj.get("answer", "UNANSWERABLE"))
        claims: list[Claim] = []
        for c in obj.get("claims", [])[:max_claims]:
            if isinstance(c, dict) and str(c.get("text", "")).strip():
                claims.append(Claim(text=str(c["text"]).strip(), answer_span=c.get("answer_span")))
        return answer, claims, True
    except (json.JSONDecodeError, AttributeError):
        return "UNANSWERABLE", [], False
