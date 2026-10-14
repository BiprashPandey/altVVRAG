"""Evaluation metrics (Section 10.5): answer, retrieval, localization,
selective-prediction, conformal-check, efficiency + bootstrap CIs."""
from __future__ import annotations

import numpy as np

from vvrag.reader.claims import normalize_answer, token_f1


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def f1(pred: str, gold: str) -> float:
    return token_f1(pred, gold)


def hit_at_k(ranked_pages: list[tuple[str, int]], gold: tuple[str, int], k: int) -> float:
    return 1.0 if gold in ranked_pages[:k] else 0.0


def mrr(ranked_pages: list[tuple[str, int]], gold: tuple[str, int]) -> float:
    for i, p in enumerate(ranked_pages):
        if p == gold:
            return 1.0 / (i + 1)
    return 0.0


def iou(a: tuple, b: tuple) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_inside(pred: tuple, gold: tuple) -> float:
    cx, cy = (gold[0] + gold[2]) / 2, (gold[1] + gold[3]) / 2
    return 1.0 if pred[0] <= cx <= pred[2] and pred[1] <= cy <= pred[3] else 0.0


def selective_stats(answered: np.ndarray, correct: np.ndarray) -> dict:
    """answered/correct: bool arrays. Coverage, selective acc/err, confident-wrong."""
    answered = np.asarray(answered, dtype=bool)
    correct = np.asarray(correct, dtype=bool)
    cov = float(answered.mean()) if answered.size else 0.0
    n_ans = int(answered.sum())
    sel_acc = float(correct[answered].mean()) if n_ans else 0.0
    return {
        "coverage": cov,
        "selective_accuracy": sel_acc,
        "selective_error": 1.0 - sel_acc if n_ans else 0.0,
        "confident_wrong_rate": float((~correct & answered).sum() / answered.size) if answered.size else 0.0,
        "n_answered": n_ans,
    }


def risk_coverage_curve(scores: np.ndarray, errors: np.ndarray,
                        n_points: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Sweep tau over score quantiles: (coverages, selective errors)."""
    scores = np.asarray(scores, dtype=float)
    errors = np.asarray(errors, dtype=int)
    taus = np.unique(np.quantile(scores, np.linspace(0, 1, n_points)))
    covs, risks = [], []
    for t in taus:
        acc = scores >= t
        covs.append(float(acc.mean()))
        risks.append(float(errors[acc].mean()) if acc.any() else 0.0)
    order = np.argsort(covs)
    return np.array(covs)[order], np.array(risks)[order]


def aurc(scores: np.ndarray, errors: np.ndarray) -> float:
    covs, risks = risk_coverage_curve(scores, errors)
    return float(np.trapezoid(risks, covs))


def bootstrap_ci(values: np.ndarray, stat=np.mean, n_resamples: int = 1000,
                 seed: int = 42, ci: float = 0.95) -> tuple[float, float, float]:
    """Returns (stat, lo, hi) bootstrap CI."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return 0.0, 0.0, 0.0
    boots = [stat(values[rng.integers(0, n, n)]) for _ in range(n_resamples)]
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return float(stat(values)), float(np.percentile(boots, lo_q)), float(np.percentile(boots, hi_q))
