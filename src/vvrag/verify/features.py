"""Feature vector assembly (Section 7.4, `verify/features.py`).

FEATURE_ORDER is the single source of truth for scorer input order. Run 9
(guideline s4.7) drops `answer_type_match` (the check is a warning now) and
adds the examiner's verdict as the strongest feature:

  ... citation_resolved, support_type, examiner_ran, examiner_supported,
      quote_seen_match, details_ok

Missing entailment (textless crop) is encoded as entail_prob=0.5 +
crop_has_text=0. Missing examiner columns default to examiner_ran=0 (unknown)
with neutral imputations, so old callers keep working.
"""
from __future__ import annotations

from vvrag.schemas import Features

FEATURE_ORDER = ["agreement", "entail_prob", "retrieval_margin",
                 "heat_concentration", "n_claims", "crop_has_text",
                 "localize_score", "evidence_valid", "quote_match",
                 "answer_supported", "cited_is_top1", "visual_margin",
                 "citation_resolved", "support_type", "examiner_ran",
                 "examiner_supported", "quote_seen_match", "details_ok"]
MISSING_ENTAIL_VALUE = 0.5
MISSING_SUPPORTED_VALUE = 0.5   # verdict unknown (examiner did not run)


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
    support_type: float = 1.0,
    examiner_ran: bool | float = 0.0,
    examiner_supported: float | None = None,
    quote_seen_match: float | None = None,
    details_ok: bool | float = 1.0,
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
        support_type=float(support_type),
        examiner_ran=float(examiner_ran),
        examiner_supported=(MISSING_SUPPORTED_VALUE
                            if examiner_supported is None
                            else float(examiner_supported)),
        quote_seen_match=(MISSING_SUPPORTED_VALUE
                          if quote_seen_match is None
                          else float(quote_seen_match)),
        details_ok=float(details_ok),
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
        float(g("support_type", 1.0) if g("support_type", 1.0) is not None else 1.0),
        1.0 if g("examiner_ran", 0.0) else 0.0,
        float(g("examiner_supported", MISSING_SUPPORTED_VALUE)
              if g("examiner_supported", None) is not None
              else MISSING_SUPPORTED_VALUE),
        float(g("quote_seen_match", MISSING_SUPPORTED_VALUE)
              if g("quote_seen_match", None) is not None
              else MISSING_SUPPORTED_VALUE),
        float(g("details_ok", 1.0) if g("details_ok", 1.0) is not None else 1.0),
    ]


def aggregate_for_gate(entail_probs: list[float]) -> tuple[float, float]:
    from vvrag.verify.nli import aggregate_entailment

    return aggregate_entailment(entail_probs)
