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


def equivalent_f1(pred: str, golds: list[str]) -> float:
    """Max token-F1 of `pred` over acceptable golds (equivalent-answer scoring).

    `golds` should include the primary gold. Empty -> 0.0.
    """
    if not golds:
        return 0.0
    return float(max(token_f1(pred, g) for g in golds))


def _tokens(s: str) -> list[str]:
    return normalize_answer(s).split()


def equivalent_correct(
    pred: str, golds: list[str], f1_threshold: float = 0.8,
) -> bool:
    """True when `pred` matches ANY acceptable gold (equivalent-answer scoring).

    P3 fix: an answer is credited if (a) it equals a gold, (b) token-F1 with a
    gold is >= threshold, or (c) it is a token-superset of a gold (a longer
    answer that contains an acceptable gold, e.g. "important information or
    citations" contains "important information").
    """
    if not golds:
        return False
    p = normalize_answer(pred)
    for g in golds:
        if p == normalize_answer(g):
            return True
    if equivalent_f1(pred, golds) >= f1_threshold:
        return True
    ptoks = _tokens(pred)
    from collections import Counter
    pc = Counter(ptoks)
    for g in golds:
        gtoks = _tokens(g)
        if gtoks and all(pc[t] >= c for t, c in Counter(gtoks).items()):
            return True
    return False


def equivalent_f1_norm(pred: str, golds: list[str]) -> float:
    """Number-normalized max token-F1 over acceptable golds (Phase 6).

    "12.5 bn" -> "12.5 billion" so unit variants score as equivalent.
    """
    from vvrag.verify.support import normalize_numbers

    if not golds:
        return 0.0
    return float(max(token_f1(normalize_numbers(pred), normalize_numbers(g))
                     for g in golds))


def equivalent_correct_norm(
    pred: str, golds: list[str], f1_threshold: float = 0.8,
) -> bool:
    """`equivalent_correct` after number/unit normalization (Phase 6)."""
    from vvrag.verify.support import normalize_numbers

    if not golds:
        return False
    return equivalent_correct(
        normalize_numbers(pred),
        [normalize_numbers(g) for g in golds],
        f1_threshold=f1_threshold)


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
        # P3: abstention quality split (answerable vs unanswerable)
        stats["false_abstention_rate"] = (
            float((~answered[ans_mask]).mean()) if ans_mask.any() else 0.0)
        n_abst = int((~answered).sum())
        stats["abstention_precision"] = (
            float((~answered[una]).sum() / n_abst) if n_abst else 0.0)
        stats["abstention_recall"] = stats["unanswerable_abstention_rate"]
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


def clopper_pearson_interval(k: int, n: int, ci: float = 0.95) -> tuple[float, float]:
    """Two-sided Clopper-Pearson (exact binomial) interval for k successes / n.

    Used for small-n safety metrics (probe abstention, false-answer rate,
    gate accepted-error) where normal/bootstrap approximations are too
    optimistic. k<=0 -> [0, beta], k>=n -> [alpha, 1].
    """
    from scipy.stats import beta

    if n <= 0:
        return 0.0, 0.0
    alpha = 1.0 - ci
    lo = 0.0 if k <= 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


# ---------------------------------------------------------------------------
# Run 9 (guideline s8): the three-state answer contract metrics.
# Operates on run-harness row dicts (rows.json / *_rows.jsonl).
# ---------------------------------------------------------------------------

def verified_stats(rows: list[dict], examiner_only: bool = False) -> dict:
    """Verified precision, flag recall, coverage, source accuracy.

    - verified_precision: correct / verified (target >= 95%)
    - flag_recall: flagged / incorrect  (target ~90%)
    - unflagged_wrong_rate: incorrect & unflagged / incorrect
    - answer_coverage: gold rows with a shown answer / gold rows (was 0.33)
    - source_accuracy: shown gold rows with source boxes/refs / shown gold rows
    - examiner_leniency: examiner-confirmed rows later judged wrong / confirmed
    - probe_flag_rate: probes flagged (nothing_found OR not_confident) / probes
    - false_verified_on_probes: probes reaching VERIFIED (the dangerous one)

    ``examiner_only`` restricts the *gold* denominators to rows where the
    examiner actually ran. Rows with no examiner are ``not_confident`` by
    contract (not verification failures), so including them in flag/coverage
    statistics would overstate the system. Probe metrics always use all probes
    (a probe that was never shown needs no examiner to be correctly flagged).
    """
    gold_all = [r for r in rows if "error" not in r and not r.get("unanswerable")]
    probes = [r for r in rows if "error" not in r and r.get("unanswerable")]

    def _ran_examiner(r):
        return bool(r.get("examiner_ran")
                    or (r.get("examiner") or {}).get("ran")
                    or (r.get("examiner") or {}).get("confirmed"))

    if examiner_only:
        gold = [r for r in gold_all if _ran_examiner(r)]
    else:
        gold = gold_all

    def _verified(r):
        if r.get("state") is not None:
            return r.get("state") == "verified"
        return bool(r.get("verified"))

    def _flagged(r):
        return not _verified(r)

    correct = [r for r in gold if r.get("equiv_correct")]
    incorrect = [r for r in gold if not r.get("equiv_correct")]
    verified = [r for r in gold if _verified(r)]
    verified_correct = [r for r in verified if r.get("equiv_correct")]
    flagged_incorrect = [r for r in incorrect if _flagged(r)]
    shown = [r for r in gold_all if not r.get("abstained")]
    with_source = [r for r in shown
                   if r.get("pred_boxes") or r.get("pred_sentence_ids")]
    confirmed = [r for r in gold
                 if (r.get("examiner") or {}).get("confirmed")]
    lenient = [r for r in confirmed if not r.get("equiv_correct")]
    probes_flagged = [r for r in probes if _flagged(r)]
    probes_verified = [r for r in probes if _verified(r)]

    def _rate(x, n):
        return (len(x) / n) if n else None

    k = len(verified_correct)
    n = len(verified)
    vp_cp = list(clopper_pearson_interval(k, n)) if n else None

    return {
        "n_gold": len(gold_all),
        "n_gold_examinable": len(gold) if examiner_only else len(gold_all),
        "n_examiner_unavailable": (len(gold_all) - len(gold))
        if examiner_only else 0,
        "examiner_only": examiner_only,
        "n_probes": len(probes),
        "verified_precision": _rate(verified_correct, len(verified)),
        "verified_precision_cp95": vp_cp,
        "n_verified": len(verified),
        "flag_recall": _rate(flagged_incorrect, len(incorrect)),
        "unflagged_wrong_rate": _rate(
            [r for r in incorrect if not _flagged(r)], len(incorrect)),
        "answer_coverage": _rate(shown, len(gold_all)),
        "source_accuracy": _rate(with_source, len(shown)),
        "examiner_leniency": _rate(lenient, len(confirmed)),
        "n_confirmed": len(confirmed),
        "probe_flag_rate": _rate(probes_flagged, len(probes)),
        "false_verified_on_probes": _rate(probes_verified, len(probes)),
    }
