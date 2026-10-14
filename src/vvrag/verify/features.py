"""Feature vector assembly (Section 7.4, `verify/features.py`).

FEATURE_ORDER is the single source of truth for scorer input order:
[agreement, entail_prob, retrieval_margin, heat_concentration, n_claims, crop_has_text]
Missing entailment (textless crop) is encoded as entail_prob=0.5 + crop_has_text=0.
"""
from __future__ import annotations

from vvrag.schemas import Features

FEATURE_ORDER = ["agreement", "entail_prob", "retrieval_margin",
                 "heat_concentration", "n_claims", "crop_has_text"]
MISSING_ENTAIL_VALUE = 0.5


def build_features(
    agreement: float = 0.0,
    entail_min: float | None = None,
    entail_mean: float | None = None,
    retrieval_margin: float = 0.0,
    heat_concentration: float = 0.0,
    n_claims: int = 0,
    crop_has_text: bool = True,
) -> Features:
    entail = entail_min if entail_min is not None else MISSING_ENTAIL_VALUE
    return Features(
        agreement=float(agreement),
        entail_prob=float(entail),
        retrieval_margin=float(retrieval_margin),
        heat_concentration=float(heat_concentration),
        n_claims=int(n_claims),
        crop_has_text=bool(crop_has_text),
    )


def to_vector(f: Features) -> list[float]:
    return [
        float(f.agreement),
        float(f.entail_prob if f.entail_prob is not None else MISSING_ENTAIL_VALUE),
        float(f.retrieval_margin),
        float(f.heat_concentration),
        float(f.n_claims),
        1.0 if f.crop_has_text else 0.0,
    ]


def aggregate_for_gate(entail_probs: list[float]) -> tuple[float, float]:
    from vvrag.verify.nli import aggregate_entailment

    return aggregate_entailment(entail_probs)
