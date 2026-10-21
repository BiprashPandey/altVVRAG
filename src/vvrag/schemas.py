"""All data contracts (Section 4). Pixel space = rendered page image, origin top-left."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class PageMeta:
    doc_id: str
    page: int
    width: int
    height: int
    dpi: int = 150
    grid_h: int = 0
    grid_w: int = 0
    has_text_layer: bool = False
    image_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PageMeta":
        return cls(**d)


@dataclass
class WordBox:
    doc_id: str
    page: int
    text: str
    box: tuple[int, int, int, int]  # (x0, y0, x1, y1) pixels
    block_id: int = 0
    line_id: int = 0
    sent_id: int = -1  # sentence unit index on this page (-1 = not assigned)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["box"] = list(self.box)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WordBox":
        d = dict(d)
        d["box"] = tuple(d["box"])
        d.setdefault("sent_id", -1)
        return cls(**{k: d.get(k) for k in
                      ("doc_id", "page", "text", "box", "block_id", "line_id", "sent_id")})


@dataclass
class SentenceSpan:
    """Numbered sentence unit with ONE box (multi-line sentences merged).

    The reader cites these ids; final display boxes are the union of the
    cited sentences' boxes. Coordinates are rendered-page pixels.
    """
    doc_id: str
    page: int
    sent_id: int
    text: str
    box: tuple[int, int, int, int]  # (x0, y0, x1, y1) pixels
    block_id: int = 0
    source: str = "textlayer"  # textlayer | ocr | ocr-reocr | vlm

    def to_dict(self) -> dict:
        d = asdict(self)
        d["box"] = list(self.box)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SentenceSpan":
        d = dict(d)
        d["box"] = tuple(d["box"])
        d.setdefault("source", "textlayer")
        return cls(**{k: d.get(k) for k in
                      ("doc_id", "page", "sent_id", "text", "box", "block_id",
                       "source")})


@dataclass
class RetrievalHit:
    doc_id: str
    page: int
    score: float
    stage1_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RetrievalHit":
        return cls(**d)


@dataclass
class Heatmap:
    doc_id: str
    page: int
    grid: list[list[float]]  # [grid_h][grid_w]
    query_tokens: list[str] = field(default_factory=list)
    per_token: list[list[list[float]]] | None = None  # [T][grid_h][grid_w]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Heatmap":
        return cls(**d)


@dataclass
class Evidence:
    doc_id: str
    page: int
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    crop_path: str = ""
    crop_text: str = ""
    heat_concentration: float = 0.0
    sentence_ids: list[int] = field(default_factory=list)
    box_frac: list[tuple[float, float, float, float]] = field(default_factory=list)
    page_width: int = 0
    page_height: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["boxes"] = [list(b) for b in self.boxes]
        d["box_frac"] = [list(b) for b in self.box_frac]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Evidence":
        d = dict(d)
        d["boxes"] = [tuple(b) for b in d.get("boxes", [])]
        d["box_frac"] = [tuple(b) for b in d.get("box_frac", [])]
        d.setdefault("sentence_ids", [])
        d.setdefault("page_width", 0)
        d.setdefault("page_height", 0)
        return cls(**{k: d.get(k) for k in
                      ("doc_id", "page", "boxes", "crop_path", "crop_text",
                       "heat_concentration", "sentence_ids", "box_frac",
                       "page_width", "page_height")})


@dataclass
class Claim:
    text: str
    answer_span: str | None = None
    evidence_sentence_ids: list[int] | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Claim":
        d = dict(d)
        # legacy word-id fields are NOT mapped: word indices are not sentence
        # ids; old cached outputs simply have no valid citations.
        return cls(text=str(d.get("text", "")),
                   answer_span=d.get("answer_span"),
                   evidence_sentence_ids=d.get("evidence_sentence_ids"))


@dataclass
class ReaderSample:
    sample_id: int
    answer: str
    claims: list[Claim] = field(default_factory=list)
    raw: str = ""
    quote: str = ""
    sentence_ids: list[int] = field(default_factory=list)
    # Phase 2: per-citation (page, sent_id) refs parsed from namespaced ids
    # ("p3.s12") or legacy ints (page=None -> primary page). JSON-able pairs.
    sentence_refs: list | None = None
    # Phase 9 (Run 8): optional visual region citation
    # {"page": label|None, "box_frac": [x0,y0,x1,y1], "evidence_type": "visual"}
    region: dict | None = None
    # Run 9 (guideline s4.2): the reader may flag "no evidence at all"
    # (only then is the page skipped), plus a location hint and confidence.
    no_evidence: bool = False
    found_in: str = ""
    confidence: float | None = None

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "answer": self.answer,
            "claims": [c.to_dict() for c in self.claims],
            "raw": self.raw,
            "quote": self.quote,
            "sentence_ids": list(self.sentence_ids),
            "sentence_refs": ([list(r) for r in self.sentence_refs]
                              if self.sentence_refs is not None else None),
            "region": self.region,
            "no_evidence": self.no_evidence,
            "found_in": self.found_in,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReaderSample":
        return cls(
            sample_id=d["sample_id"],
            answer=d["answer"],
            claims=[Claim.from_dict(c) for c in d.get("claims", [])],
            raw=d.get("raw", ""),
            quote=d.get("quote", ""),
            sentence_ids=list(d.get("sentence_ids", [])),
            sentence_refs=d.get("sentence_refs"),
            region=d.get("region"),
            no_evidence=bool(d.get("no_evidence", False)),
            found_in=d.get("found_in", ""),
            confidence=d.get("confidence"),
        )


@dataclass
class Features:
    agreement: float = 0.0
    entail_prob: float | None = None
    retrieval_margin: float = 0.0
    heat_concentration: float = 0.0
    n_claims: int = 0
    crop_has_text: bool = True
    localize_score: float = 0.0
    evidence_valid: bool = False
    quote_match: float = 0.0
    # Phase 4: verification veto + retrieval-quality signals
    answer_supported: float = 1.0
    cited_is_top1: float = 1.0
    visual_margin: float = 0.0
    # Phase 9 (Run 8): citation/question-aware verification
    citation_resolved: float = 1.0
    support_type: float = 1.0
    # Run 9 (guideline s4.7): examiner verdict (strongest signal) + details
    examiner_ran: float = 0.0
    examiner_supported: float = 0.5   # 1 yes | 0.5 partly/unknown | 0 no
    quote_seen_match: float = 0.5
    details_ok: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Features":
        # backward compat: older records lack the new gate fields
        d = dict(d)
        d.setdefault("localize_score", 0.0)
        d.setdefault("evidence_valid", False)
        d.setdefault("quote_match", 0.0)
        d.setdefault("answer_supported", 1.0)
        d.setdefault("cited_is_top1", 1.0)
        d.setdefault("visual_margin", 0.0)
        d.setdefault("citation_resolved", 1.0)
        d.setdefault("support_type", 1.0)
        d.setdefault("examiner_ran", 0.0)
        d.setdefault("examiner_supported", 0.5)
        d.setdefault("quote_seen_match", 0.5)
        d.setdefault("details_ok", 1.0)
        return cls(**{k: d.get(k, v) for k, v in [
            ("agreement", 0.0), ("entail_prob", None),
            ("retrieval_margin", 0.0), ("heat_concentration", 0.0),
            ("n_claims", 0), ("crop_has_text", True),
            ("localize_score", 0.0), ("evidence_valid", False),
            ("quote_match", 0.0), ("answer_supported", 1.0),
            ("cited_is_top1", 1.0), ("visual_margin", 0.0),
            ("citation_resolved", 1.0), ("support_type", 1.0),
            ("examiner_ran", 0.0), ("examiner_supported", 0.5),
            ("quote_seen_match", 0.5), ("details_ok", 1.0)]})


@dataclass
class Result:
    question: str
    answer: str
    abstained: bool = False
    score: float = 0.0
    tau: float = 0.0
    evidence: Evidence = field(default_factory=lambda: Evidence(doc_id="", page=0))
    samples: list[ReaderSample] = field(default_factory=list)
    features: Features = field(default_factory=Features)
    timings: dict = field(default_factory=dict)
    api_calls: int = 0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "abstained": self.abstained,
            "score": self.score,
            "tau": self.tau,
            "evidence": self.evidence.to_dict(),
            "samples": [s.to_dict() for s in self.samples],
            "features": self.features.to_dict(),
            "timings": self.timings,
            "api_calls": self.api_calls,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "Result":
        return cls(
            question=d["question"],
            answer=d["answer"],
            abstained=d.get("abstained", False),
            score=d.get("score", 0.0),
            tau=d.get("tau", 0.0),
            evidence=Evidence.from_dict(d.get("evidence", {"doc_id": "", "page": 0})),
            samples=[ReaderSample.from_dict(s) for s in d.get("samples", [])],
            features=Features.from_dict(d.get("features", {})),
            timings=d.get("timings", {}),
            api_calls=d.get("api_calls", 0),
        )


# ---------------------------------------------------------------------------
# Coordinate helpers (single source of truth for grid -> pixel conversion).
# All boxes are in rendered page image pixels, origin top-left.
# ---------------------------------------------------------------------------

def grid_cell_range_to_pixel_box(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    page_width: int,
    page_height: int,
    grid_h: int,
    grid_w: int,
    padding_px: int = 0,
) -> tuple[int, int, int, int]:
    """Convert inclusive-exclusive cell range [r0,r1)x[c0,c1) to pixel box.

    Cell (r, c) covers x in [c*W/gw, (c+1)*W/gw), y in [r*H/gh, (r+1)*H/gh).
    Result is clipped to the page and padded (then re-clipped).
    """
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid_h and grid_w must be positive")
    r0 = max(0, min(r0, grid_h))
    r1 = max(0, min(r1, grid_h))
    c0 = max(0, min(c0, grid_w))
    c1 = max(0, min(c1, grid_w))
    if r1 <= r0 or c1 <= c0:
        raise ValueError(f"empty cell range r[{r0},{r1}) c[{c0},{c1})")
    x0 = int(c0 * page_width / grid_w)
    x1 = int(round(c1 * page_width / grid_w))
    y0 = int(r0 * page_height / grid_h)
    y1 = int(round(r1 * page_height / grid_h))
    if padding_px:
        x0 -= padding_px
        y0 -= padding_px
        x1 += padding_px
        y1 += padding_px
    return clip_box_to_page((x0, y0, x1, y1), page_width, page_height)


def clip_box_to_page(
    box: tuple[int, int, int, int], page_width: int, page_height: int
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0 = max(0, min(int(x0), page_width))
    y0 = max(0, min(int(y0), page_height))
    x1 = max(0, min(int(x1), page_width))
    y1 = max(0, min(int(y1), page_height))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return (x0, y0, x1, y1)
