"""Self-consistency agreement (Section 7.4, contribution C2)."""
from __future__ import annotations

from vvrag.reader.claims import answers_match
from vvrag.schemas import ReaderSample


def agreement(
    samples: list[ReaderSample], f1_threshold: float = 0.8
) -> tuple[float, int]:
    """Returns (agreement, n_distinct).

    agreement = fraction of the k samples whose normalized answer matches the
    primary answer (sample 0), exact-after-normalization or token-F1 >= threshold.
    """
    if not samples:
        return 0.0, 0
    primary = samples[0].answer
    matched = sum(1 for s in samples if answers_match(s.answer, primary, f1_threshold))
    distinct = len({s.answer.strip().lower() for s in samples})
    return matched / len(samples), distinct
