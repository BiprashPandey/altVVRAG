"""Self-consistency agreement (Section 7.4, contribution C2)."""
from __future__ import annotations

from vvrag.reader.claims import answers_match
from vvrag.schemas import ReaderSample

# Phase 4: with a single sample agreement is uninformative (always 1.0); the
# neutral imputation keeps the feature from looking like perfect confidence.
MISSING_AGREEMENT = 0.5


def agreement(
    samples: list[ReaderSample], f1_threshold: float = 0.8
) -> tuple[float, int]:
    """Returns (agreement, n_distinct).

    agreement = fraction of the k samples whose normalized answer matches the
    primary answer (sample 0), exact-after-normalization or token-F1 >= threshold.
    n=1 -> neutral imputation 0.5 (see MISSING_AGREEMENT).
    """
    if not samples:
        return 0.0, 0
    if len(samples) == 1:
        return MISSING_AGREEMENT, 1
    primary = samples[0].answer
    matched = sum(1 for s in samples if answers_match(s.answer, primary, f1_threshold))
    distinct = len({s.answer.strip().lower() for s in samples})
    return matched / len(samples), distinct
