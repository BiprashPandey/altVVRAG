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
    answer, quote, ids, _refs, claims, ok = parse_reader_output_refs(raw, max_claims)
    return answer, quote, ids, claims, ok


_REF_RE = re.compile(r"^p?(\d+)\s*[.\-:/]\s*s?(\d+)$", re.IGNORECASE)


def _parse_ref_list(value) -> tuple[list[int], list[list]]:
    """Parse citation values: legacy ints (page=None) or namespaced "p3.s12".

    Returns (sentence_ids, refs) where refs are JSON-able [page|None, sent_id].
    """
    if not isinstance(value, list) or not value:
        return [], []
    ints: list[int] = []
    refs: list[list] = []
    for x in value:
        if isinstance(x, bool):
            continue
        if isinstance(x, int):
            ints.append(x)
            refs.append([None, x])
            continue
        if isinstance(x, float) and float(x).is_integer():
            ints.append(int(x))
            refs.append([None, int(x)])
            continue
        s = str(x).strip()
        m = _REF_RE.match(s)
        if m:
            page, sid = int(m.group(1)), int(m.group(2))
            refs.append([page, sid])
            ints.append(sid)
            continue
        try:
            iv = int(float(s))
        except (TypeError, ValueError):
            continue
        ints.append(iv)
        refs.append([None, iv])
    return ints, refs


def parse_reader_output_refs(
    raw: str, max_claims: int = 5
) -> tuple[str, str, list[int], list[list], list[Claim], bool]:
    """Phase 2 variant: also returns per-citation (page, sent_id) refs.

    Returns (answer, quote, sentence_ids, sentence_refs, claims, ok).
    `sentence_ids` keeps only the sent_id ints (backward compatible);
    `sentence_refs` is [page|None, sent_id] with page=None meaning "the
    primary (image-attached) page".
    """
    answer, quote, ids, refs, _region, claims, ok = parse_reader_output_ext(
        raw, max_claims)
    return answer, quote, ids, refs, claims, ok


def _parse_region(obj: dict) -> dict | None:
    """Parse an optional visual-region citation.

    Accepts {"region"|"visual_region": {"page": "p1"|1|null,
    "box_frac": [x0,y0,x1,y1]} } (fractions 0-1 of that page). Returns
    {"page": label|None, "box_frac": [4 floats], "evidence_type": str} or None.
    """
    r = obj.get("region") or obj.get("visual_region")
    if not isinstance(r, dict):
        return None
    box = r.get("box_frac") or r.get("box")
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return None
    try:
        f = [float(x) for x in box]
    except (TypeError, ValueError):
        return None
    if not (0.0 <= f[0] < f[2] <= 1.01 and 0.0 <= f[1] < f[3] <= 1.01):
        return None
    page = r.get("page")
    label = None
    if page is not None:
        if isinstance(page, bool):
            page = None
        elif isinstance(page, (int, float)) and float(page).is_integer():
            label = int(page)
        else:
            m = _REF_RE.match(str(page).strip()) or _PAGE_RE.match(str(page).strip())
            if m:
                label = int(m.group(1))
    etype = str(obj.get("evidence_type", r.get("evidence_type", "")) or "").lower()
    return {"page": label, "box_frac": f,
            "evidence_type": etype if etype in ("text", "visual") else ""}


_PAGE_RE = re.compile(r"^p?(\d+)$", re.IGNORECASE)


def parse_reader_output_ext(
    raw: str, max_claims: int = 5
) -> tuple[str, str, list[int], list[list], dict | None, list[Claim], bool]:
    """Full parse incl. the optional visual region (Phase 9/Run 8).

    Returns (answer, quote, sentence_ids, sentence_refs, region, claims, ok).
    """
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("```")).strip()
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        return "UNANSWERABLE", "", [], [], None, [], False
    answer = str(obj.get("answer", "UNANSWERABLE"))
    quote = str(obj.get("quote", "") or "")
    top_ids, top_refs = _parse_ref_list(obj.get("sentence_ids"))
    region = _parse_region(obj)
    claims: list[Claim] = []
    for c in obj.get("claims", [])[:max_claims]:
        if isinstance(c, dict) and str(c.get("text", "")).strip():
            cids, _crefs = _parse_ref_list(
                c.get("evidence_sentence_ids", c.get("sentence_ids")))
            claims.append(Claim(text=str(c["text"]).strip(),
                                answer_span=c.get("answer_span"),
                                evidence_sentence_ids=cids))
    if not top_refs and claims:
        # fall back to the first claim's citations
        top_ids = list(claims[0].evidence_sentence_ids or [])
        top_refs = [[None, i] for i in top_ids]
    return answer, quote, top_ids, top_refs, region, claims, True


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
