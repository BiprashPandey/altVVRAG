"""Deterministic citation-specific checks.

The report's verifier judged whether the *page* supported a claim and missed a
true claim attached to a fabricated citation. These helpers add the cheap,
model-free part of the fix:

- content terms of a claim (numbers and named values included),
- shared-term overlap between the claim and the cited region/page text,
- a page-number-sink check (a region that only contains a page number cannot
  support a substantive claim).

They never replace the independent VLM verifier; they run before it and can
downgrade a citation to "not specific" when the cited text shares nothing with
the claim.
"""
from __future__ import annotations

import re

from vvrag.reader.claims import normalize_answer
from vvrag.verify.support import normalize_numbers

_STOP = frozenset(
    "what which who whom whose when where why how is are was were be been being "
    "do does did done have has had having a an the and or of on in to for with "
    "by from as at it its this that these those then than so such no not nor "
    "please tell give show find answer question page document according report "
    "reported shows shown stated indicates indicated".split()
)
_WORD = re.compile(r"[a-z0-9][a-z0-9.%-]*", re.IGNORECASE)


def content_terms(text: str) -> list[str]:
    """Normalized content-bearing terms, preserving numbers/decimals/units.

    A term is kept when it is not a stopword, has at least two characters, and
    contains a letter or digit. Punctuation-only tokens are discarded.
    """
    normalized = normalize_numbers(text)
    out: list[str] = []
    for tok in normalized.split():
        tok = tok.strip(".-")
        if not tok or tok in _STOP:
            continue
        if len(tok) < 2 and not tok.isdigit():
            continue
        if not _WORD.search(tok):
            continue
        out.append(tok)
    return out


def shared_terms(claim: str, evidence_texts: list[str] | None) -> list[str]:
    """Sorted terms present in both the claim and at least one evidence text."""
    claim_terms = set(content_terms(claim))
    if not claim_terms:
        return []
    evidence: set[str] = set()
    for text in (evidence_texts or []):
        evidence.update(content_terms(text))
    return sorted(claim_terms & evidence)


def citation_overlap(
    claim: str,
    evidence_texts: list[str] | None,
    min_shared: int = 1,
) -> tuple[bool, float, list[str]]:
    """(is_specific, overlap, shared_terms) for a claim -> cited text check.

    The overlap is the fraction of the claim's content terms found in the
    evidence. Short numeric answers are handled by term equality, not raw
    substring matching ("12.5" must not match "112.5"). If the claim has no
    content terms at all (e.g. an empty answer), the check fails closed.
    """
    claim_terms = set(content_terms(claim))
    shared = shared_terms(claim, evidence_texts)
    if not claim_terms:
        return False, 0.0, []
    score = len(shared) / max(1, len(claim_terms))
    ok = len(shared) >= max(1, int(min_shared))
    return bool(ok), float(score), shared


def page_number_sink(claim: str, evidence_texts: list[str] | None) -> bool:
    """True when the evidence is numeric-only and shares nothing with a
    substantive claim. A region holding only a page number is the measured
    failure mode this guard exists to stop."""
    evidence_terms: list[str] = []
    for text in (evidence_texts or []):
        evidence_terms.extend(content_terms(text))
    if not evidence_terms:
        return False
    if not all(t.replace(".", "").isdigit() for t in evidence_terms):
        return False
    claim_terms = set(content_terms(claim))
    return bool(claim_terms) and not (claim_terms & set(evidence_terms))
