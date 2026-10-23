"""Deterministic document-grounded candidates for one claim.

The reference system's grounding stage only had a crop/heatmap; a paraphrase
could not be snapped to a citable unit. These helpers make the document itself
propose candidate regions, so a claim can be checked and boxed against:

- cited sentence / table-row / table-cell units already stored in the index,
- word-aligned line groups on the page,
- the reader's declared visual region (still a candidate, never trusted).

Ranking is lexical and model-free (number/name aware). The independent VLM
verifier remains the judge of whether a candidate actually supports the claim.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from vvrag.ingest.sentences import box_frac
from vvrag.schemas import GroundedRegion, SentenceSpan, WordBox
from vvrag.verify.citation import citation_overlap


@dataclass
class GroundCandidate:
    region: GroundedRegion
    kind: str = "sentence"
    score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)


def _clip(box: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
    y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
    return (x0, y0, max(x0, x1), max(y0, y1))


def _region(doc_id: str, page: int, box, text: str, w: int, h: int,
            *, modality: str = "text", resolution: str = "sentence",
            source: str = "sentence", score: float = 0.0) -> GroundedRegion:
    box = _clip(box, w, h)
    return GroundedRegion(
        doc_id=doc_id, page=int(page), box=box,
        box_frac=box_frac(box, w, h), text=str(text or ""),
        modality=modality, resolution=resolution, score=float(score),
        source=source,
    )


def from_sentence(sentence: SentenceSpan, doc_id: str, page: int,
                  page_w: int, page_h: int) -> GroundedRegion:
    """Candidate from an ingested sentence/table unit (pixel + normalized box)."""
    kind = str(getattr(sentence, "source", "") or "")
    resolution = "cell" if kind == "table" and len(sentence.text.split()) <= 3 else (
        "row" if kind == "table" else "sentence")
    return _region(doc_id, page, sentence.box, sentence.text, page_w, page_h,
                   resolution=resolution, source="sentence")


def from_words(words: list[WordBox], doc_id: str, page: int,
               page_w: int, page_h: int, text: str = "",
               resolution: str = "line") -> GroundedRegion | None:
    """Candidate from a word-aligned group (a line or a tight phrase)."""
    if not words:
        return None
    box = (min(w.box[0] for w in words), min(w.box[1] for w in words),
           max(w.box[2] for w in words), max(w.box[3] for w in words))
    return _region(doc_id, page, box, text or " ".join(w.text for w in words),
                   page_w, page_h, resolution=resolution, source="snap")


def from_reader_region(frac, doc_id: str, page: int, page_w: int, page_h: int,
                       text: str = "") -> GroundedRegion | None:
    """Candidate from the reader's visual citation (never trusted directly)."""
    if not frac or len(frac) != 4:
        return None
    x0, y0, x1, y1 = (float(v) for v in frac)
    box = (int(round(x0 * page_w)), int(round(y0 * page_h)),
           int(round(x1 * page_w)), int(round(y1 * page_h)))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return _region(doc_id, page, box, text, page_w, page_h,
                   modality="visual", resolution="region", source="reader")


_NUM_RE = re.compile(
    r"\d+(?:[.,]\d+)*\s*(?:%|bn|billion|million|mn|tn|trillion|k|m)?",
    re.IGNORECASE)
_UNIT = {"bn": "billion", "mn": "million", "tn": "trillion", "m": "million",
         "k": "thousand"}


def _number_units(text: str) -> set[str]:
    out: set[str] = set()
    for m in _NUM_RE.finditer(str(text or "")):
        raw = m.group(0).lower().replace(",", "").replace(" ", "")
        num = re.match(r"[\d.]+", raw)
        if not num:
            continue
        unit = raw[num.end():]
        unit = _UNIT.get(unit, unit)
        out.add(num.group(0) + ((" " + unit) if unit else ""))
    return out


def score_candidate(claim: str, region: GroundedRegion,
                    min_shared: int = 1,
                    focus: str = "") -> tuple[float, list[str], bool]:
    ok, score, shared = citation_overlap(claim, [region.text], min_shared)
    if not ok:
        return 0.0, shared, False
    # The answer text is the claim's focus. A relational claim ("A happens
    # before B") otherwise favors the longer B-side sentence over the actual
    # answer unit A; the focus bonus keeps the cited unit anchored to A.
    if focus:
        f_ok, f_score, _f_shared = citation_overlap(
            focus, [region.text], min_shared=1)
        if f_ok:
            score = float(score) + 0.5 * float(f_score)
    # Claims that contain numbers are specific: prefer the candidate that
    # carries the same number/unit over a wider lexical label.
    claim_nums = _number_units(claim)
    cand_nums = _number_units(region.text)
    specificity = 0.0
    if claim_nums:
        specificity = 0.75 * (len(claim_nums & cand_nums) / len(claim_nums))
        if not cand_nums:
            specificity -= 0.15
    return float(score) + specificity, shared, True


def rank_candidates(claim: str, regions: list[GroundedRegion],
                    min_shared: int = 1, top_k: int = 8,
                    focus: str = "") -> list[GroundCandidate]:
    """Rank document candidates for a claim; specific matches first.

    Ties prefer the smaller region (more precise citation) and then stable
    page/box order, so results are reproducible.
    """
    out: list[GroundCandidate] = []
    seen: set[tuple] = set()
    for r in regions:
        if not r or not r.text.strip():
            continue
        key = (r.doc_id, r.page, r.box, r.text.strip())
        if key in seen:
            continue
        seen.add(key)
        score, shared, ok = score_candidate(claim, r, min_shared, focus)
        if not ok:
            continue
        area = max(1, (r.box[2] - r.box[0]) * (r.box[3] - r.box[1]))
        out.append(GroundCandidate(region=r, kind=r.resolution,
                                   score=score, matched_terms=shared))
    out.sort(key=lambda c: (-c.score,
                            (c.region.box[2] - c.region.box[0])
                            * (c.region.box[3] - c.region.box[1]),
                            c.region.page, c.region.box))
    return out[:max(1, int(top_k))]


def candidates_for_claim(claim: str, sentences: list[SentenceSpan], doc_id: str,
                         page: int, page_w: int, page_h: int,
                         reader_region=None, words: list[WordBox] | None = None,
                         min_shared: int = 1, top_k: int = 8,
                         focus: str = "",
                         ) -> list[GroundCandidate]:
    """All deterministic candidates for one claim on one page, ranked."""
    regions = [from_sentence(s, doc_id, page, page_w, page_h) for s in sentences]
    if words:
        # One candidate per stored line/block; words are already ordered by
        # the page reader, so grouping is stable.
        groups: dict[tuple[int, int], list[WordBox]] = {}
        for w in words:
            groups.setdefault((w.block_id, w.line_id), []).append(w)
        for group in groups.values():
            r = from_words(group, doc_id, page, page_w, page_h)
            if r is not None:
                regions.append(r)
    rr = from_reader_region(reader_region, doc_id, page, page_w, page_h)
    if rr is not None:
        regions.append(rr)
    return rank_candidates(claim, regions, min_shared=min_shared, top_k=top_k,
                           focus=focus)
