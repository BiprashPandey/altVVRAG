"""Claim-level verifier contract.

The reference system verifies atomic claims, not only the final answer string.
This module normalizes the verifier's per-claim labels and decides whether a
claim may contribute to a VERIFIED state. It is deterministic and independent
of any provider so cached and mock outputs use the same rules.
"""
from __future__ import annotations

CLAIM_LABELS = ("supported", "partially_supported", "insufficient",
                "unsupported")

_ALIASES = {
    "supported": "supported",
    "support": "supported",
    "yes": "supported",
    "true": "supported",
    "verified": "supported",
    "entailed": "supported",
    "partially_supported": "partially_supported",
    "partially-supported": "partially_supported",
    "partial": "partially_supported",
    "partly": "partially_supported",
    "partly_supported": "partially_supported",
    "insufficient": "insufficient",
    "insufficient_evidence": "insufficient",
    "not_enough_evidence": "insufficient",
    "unclear": "insufficient",
    "unknown": "insufficient",
    "unsupported": "unsupported",
    "not_supported": "unsupported",
    "no": "unsupported",
    "false": "unsupported",
    "contradicted": "unsupported",
    "wrong": "unsupported",
}


def normalize_claim_label(
    value: str,
    answer_supported: str = "",
    citation_specific: bool | None = None,
    region_supports: bool | None = None,
) -> str:
    """Return one of CLAIM_LABELS.

    A claim whose citation is structurally wrong is at most partially
    supported: the statement may be true elsewhere, but the shown region is
    not its evidence. It can never be `supported` for verification purposes.
    """
    raw = str(value or "").strip().lower()
    label = _ALIASES.get(raw, "")
    if not label:
        sup = str(answer_supported or "").strip().lower()
        if sup == "yes":
            label = "supported"
        elif sup in ("partly", "partial"):
            label = "partially_supported"
        elif sup == "no":
            label = "unsupported"
        else:
            label = "insufficient"
    if citation_specific is False or region_supports is False:
        return "unsupported" if label == "unsupported" else "partially_supported"
    return label


def claim_confidence(label: str) -> float:
    return {"supported": 1.0, "partially_supported": 0.5,
            "insufficient": 0.25, "unsupported": 0.0}.get(label, 0.0)


def claim_is_supported(verdict: dict | None,
                       require_citation_specific: bool = True) -> bool:
    """True only for a fully supported claim with a specific citation."""
    if not verdict:
        return False
    label = str(verdict.get("label") or "")
    if label != "supported":
        return False
    if verdict.get("citation_specific") is False:
        return False
    if verdict.get("region_supports") is False:
        return False
    if require_citation_specific:
        if (verdict.get("citation_specific") is not True
                or verdict.get("region_supports") is not True):
            return False
    return True


def parse_claims(obj) -> list[dict]:
    """Normalize an examiner `claims` array; never raises."""
    if not isinstance(obj, list):
        return []
    out: list[dict] = []
    for i, row in enumerate(obj):
        if not isinstance(row, dict):
            continue
        try:
            index = int(row.get("index", i))
        except (TypeError, ValueError):
            index = i
        citation = row.get("citation_specific")
        region = row.get("region_supports")
        citation = citation if isinstance(citation, bool) else None
        region = region if isinstance(region, bool) else None
        label = normalize_claim_label(
            row.get("label", ""), row.get("answer_supported", ""),
            citation, region)
        out.append({
            "index": index,
            "label": label,
            "citation_specific": citation,
            "region_supports": region,
            "quote_found": bool(row.get("quote_found", False)),
            "quote_seen": str(row.get("quote_seen", "") or "")[:400],
            "problem": str(row.get("problem", "") or "")[:400],
        })
    return out
