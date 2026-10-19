"""Feature vector assembly (Section 7.4, `verify/features.py`).

FEATURE_ORDER is the single source of truth for scorer input order:
[agreement, entail_prob, retrieval_margin, heat_concentration, n_claims, crop_has_text]
Missing entailment (textless crop) is encoded as entail_prob=0.5 + crop_has_text=0.
"""
from __future__ import annotations

from vvrag.schemas import Features

FEATURE_ORDER = ["agreement", "entail_prob", "retrieval_margin",
                 "heat_concentration", "n_claims", "crop_has_text",
                 "localize_score", "evidence_valid", "quote_match",
                 "answer_supported", "cited_is_top1", "visual_margin",
                 "citation_resolved", "answer_type_match", "support_type"]
MISSING_ENTAIL_VALUE = 0.5


def build_features(
    agreement: float = 0.0,
    entail_min: float | None = None,
    entail_mean: float | None = None,
    retrieval_margin: float = 0.0,
    heat_concentration: float = 0.0,
    n_claims: int = 0,
    crop_has_text: bool = True,
    localize_score: float = 0.0,
    evidence_valid: bool = False,
    quote_match: float = 0.0,
    answer_supported: bool | float = 1.0,
    cited_is_top1: bool | float = 1.0,
    visual_margin: float = 0.0,
    citation_resolved: bool | float = 1.0,
    answer_type_match: bool | float | None = 1.0,
    support_type: float = 1.0,
) -> Features:
    entail = entail_min if entail_min is not None else MISSING_ENTAIL_VALUE
    return Features(
        agreement=float(agreement),
        entail_prob=float(entail),
        retrieval_margin=float(retrieval_margin),
        heat_concentration=float(heat_concentration),
        n_claims=int(n_claims),
        crop_has_text=bool(crop_has_text),
        localize_score=float(localize_score),
        evidence_valid=bool(evidence_valid),
        quote_match=float(quote_match),
        answer_supported=float(answer_supported),
        cited_is_top1=float(cited_is_top1),
        visual_margin=float(visual_margin),
        citation_resolved=float(citation_resolved),
        answer_type_match=(1.0 if answer_type_match is None
                           else float(answer_type_match)),
        support_type=float(support_type),
    )


def to_vector(f) -> list[float]:
    """Feature row in FEATURE_ORDER order. Accepts a Features object OR a dict
    (e.g. loaded from a live_labels.jsonl produced by feats.to_dict())."""

    def g(key, default=None):
        if isinstance(f, dict):
            return f.get(key, default)
        return getattr(f, key, default)

    return [
        float(g("agreement", 0.0)),
        float(g("entail_prob", MISSING_ENTAIL_VALUE)
              if g("entail_prob", None) is not None else MISSING_ENTAIL_VALUE),
        float(g("retrieval_margin", 0.0)),
        float(g("heat_concentration", 0.0)),
        float(g("n_claims", 0)),
        1.0 if g("crop_has_text", True) else 0.0,
        float(g("localize_score", 0.0) or 0.0),
        1.0 if g("evidence_valid", False) else 0.0,
        float(g("quote_match", 0.0) or 0.0),
        float(g("answer_supported", 1.0) if g("answer_supported", 1.0) is not None else 1.0),
        float(g("cited_is_top1", 1.0) if g("cited_is_top1", 1.0) is not None else 1.0),
        float(g("visual_margin", 0.0) or 0.0),
        float(g("citation_resolved", 1.0) if g("citation_resolved", 1.0) is not None else 1.0),
        float(g("answer_type_match", 1.0) if g("answer_type_match", 1.0) is not None else 1.0),
        float(g("support_type", 1.0) if g("support_type", 1.0) is not None else 1.0),
    ]


def aggregate_for_gate(entail_probs: list[float]) -> tuple[float, float]:
    from vvrag.verify.nli import aggregate_entailment

    return aggregate_entailment(entail_probs)
