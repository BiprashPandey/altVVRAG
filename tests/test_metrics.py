"""Metric unit tests: hand-computed EM/F1/IoU/selective/AURC/CI + gold regions."""
import numpy as np

from vvrag.eval.gold_regions import derive_gold_box
from vvrag.eval.metrics import (
    aurc,
    bootstrap_ci,
    center_inside,
    exact_match,
    f1,
    hit_at_k,
    iou,
    mrr,
    risk_coverage_curve,
    selective_stats,
)
from vvrag.schemas import WordBox


def test_answer_metrics():
    assert exact_match("The 42 units!", "42 units") == 1.0
    assert exact_match("apples", "oranges") == 0.0
    assert f1("a b c", "a b c") == 1.0
    assert f1("x y", "x y z") == 0.8  # P=1, R=2/3


def test_retrieval_metrics():
    rank = [("d", 0), ("d", 1), ("d", 2)]
    assert hit_at_k(rank, ("d", 1), 2) == 1.0
    assert hit_at_k(rank, ("d", 2), 2) == 0.0
    assert mrr(rank, ("d", 1)) == 0.5
    assert mrr(rank, ("x", 9)) == 0.0


def test_localization_metrics():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (5, 5, 15, 15)) == 25 / 175
    assert iou((0, 0, 1, 1), (5, 5, 6, 6)) == 0.0
    assert center_inside((0, 0, 10, 10), (4, 4, 6, 6)) == 1.0
    assert center_inside((0, 0, 1, 1), (4, 4, 6, 6)) == 0.0


def test_selective_and_risk_curve():
    s = selective_stats(np.array([True, True, False]), np.array([True, False, False]))
    assert s["coverage"] == 2 / 3 and s["selective_accuracy"] == 0.5
    assert s["confident_wrong_rate"] == 1 / 3
    covs, risks = risk_coverage_curve(np.array([0.9, 0.1]), np.array([0, 1]), n_points=3)
    assert list(covs) == sorted(covs) and all(0.0 <= r <= 1.0 for r in risks)
    assert 0.0 <= aurc(np.array([0.9, 0.8, 0.1]), np.array([0, 0, 1])) <= 1.0


def test_bootstrap_ci_covers_mean():
    m, lo, hi = bootstrap_ci(np.array([0.0, 1.0, 1.0, 1.0]))
    assert lo <= m <= hi and m == 0.75


def test_gold_box_exact_and_fuzzy():
    words = [WordBox("d", 0, t, (i * 50, 0, i * 50 + 40, 20), 0, 0)
             for i, t in enumerate("The Aurora revenue was 42 million today".split())]
    box = derive_gold_box("Aurora revenue was 42 million", words)
    assert box == (50, 0, 290, 20)
    assert derive_gold_box("zzz-no-match", words) is None
    fuzzy = derive_gold_box("Aurorra", words)  # typo -> fuzzy line match
    assert fuzzy is not None
