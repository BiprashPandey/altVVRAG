"""Fixed-threshold abstention baseline (Section 8.3). Accept iff score >= tau."""
from __future__ import annotations


def apply_fixed(score: float, tau: float = 0.5) -> bool:
    """Returns True = answer, False = abstain."""
    return float(score) >= float(tau)
