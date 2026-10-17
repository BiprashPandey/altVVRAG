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


def union_box(boxes: list[tuple]) -> tuple | None:
    """Tight union of a per-line box list (None when empty)."""
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def union_iou(pred_boxes: list[tuple], gold_box: tuple | None) -> float:
    """IoU of union(pred) vs gold union (secondary metric)."""
    if not pred_boxes or gold_box is None:
        return 0.0
    u = union_box(pred_boxes)
    return iou(u, gold_box) if u else 0.0


def per_line_iou(pred_boxes: list[tuple], gold_box: tuple | None) -> float:
    """Mean over pred line boxes of IoU(line, gold). Secondary metric.

    Per-line boxes are the primary output (rec 4); this complements
    union_iou by penalizing stray lines while word-F1 stays primary.
    """
    if not pred_boxes or gold_box is None:
        return 0.0
    return float(sum(iou(b, gold_box) for b in pred_boxes) / len(pred_boxes))


def word_precision_recall_f1(
    pred_words: list, gold_words: list,
) -> tuple[float, float, float]:
    """Word-level P/R/F1 on (text, box) identity. PRIMARY localization metric.

    Accepts WordBox objects or (text, box) tuples. Empty-vs-nonempty -> 0.
    Both empty -> (0, 0, 0) (nothing to localize; caller decides).
    """
    def _key(w):
        if hasattr(w, "text") and hasattr(w, "box"):
            return (w.text, tuple(w.box))
        t, b = w
        return (t, tuple(b))
    if not pred_words or not gold_words:
        return 0.0, 0.0, 0.0
    g = {_key(w) for w in gold_words}
    p = [_key(w) for w in pred_words]
    hit = sum(1 for t in p if t in g)
    prec = hit / max(1, len(p))
    rec = hit / max(1, len(g))
    f1v = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1v


def max_over_golds(pred: str, golds: list[str], metric) -> float:
    """Score pred against several acceptable golds, take the max (rec 7).

    E.g. q5 accepts both "Nepal Rastra Bank" and the full sentence.
    """
    if not golds:
        return 0.0
    return max(metric(pred, g) for g in golds)


def sentence_precision_recall(
    pred_ids: list[int], gold_ids: list[int],
) -> tuple[float, float, float]:
    """Sentence-citation P/R/F1 (suggestion #1): did the reader cite the right units?

    Set-based over sentence ids. Empty pred with non-empty gold -> (0, 0, 0);
    both empty -> (1, 1, 1) (nothing to cite was correctly cited as nothing).
    """
    p, g = set(pred_ids or []), set(gold_ids or [])
    if not p and not g:
        return 1.0, 1.0, 1.0
    if not p or not g:
        return 0.0, 0.0, 0.0
    hit = len(p & g)
    prec, rec = hit / len(p), hit / len(g)
    f1v = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1v


def abstention_metrics(
    answered: np.ndarray, correct: np.ndarray,
    unanswerable: np.ndarray | None = None,
) -> dict:
    """Abstention quality (suggestion #8): coverage, selective error,
    answerable-coverage and unanswerable abstention rate.

    - `answered`/`correct`: bool arrays over all questions.
    - `unanswerable`: bool mask (True = gold has no answer in the doc). When
      given, splits coverage into answerable vs unanswerable populations:
      a good gate answers answerable questions and abstains on unanswerable.
    """
    answered = np.asarray(answered, dtype=bool)
    correct = np.asarray(correct, dtype=bool)
    n = answered.size
    stats = selective_stats(answered, correct)
    stats["n"] = n
    stats["abstention_rate"] = float((~answered).mean()) if n else 0.0
    if unanswerable is not None and n:
        una = np.asarray(unanswerable, dtype=bool)
        ans_mask = ~una
        stats["n_answerable"] = int(ans_mask.sum())
        stats["n_unanswerable"] = int(una.sum())
        stats["answerable_coverage"] = float(answered[ans_mask].mean()) if ans_mask.any() else 0.0
        stats["unanswerable_abstention_rate"] = (
            float((~answered[una]).mean()) if una.any() else 0.0)
        stats["false_answer_rate_on_unanswerable"] = (
            float(answered[una].mean()) if una.any() else 0.0)
    return stats


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
