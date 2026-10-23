"""Metric unit tests: hand-computed EM/F1/IoU/selective/AURC/CI + gold regions."""
import numpy as np

from vvrag.eval.gold_regions import derive_gold_box
from vvrag.eval.metrics import (
    aurc,
    bootstrap_ci,
    center_inside,
    equivalent_correct,
    equivalent_f1,
    exact_match,
    f1,
    hit_at_k,
    iou,
    max_over_golds,
    mrr,
    per_line_iou,
    risk_coverage_curve,
    selective_stats,
    union_iou,
    word_precision_recall_f1,
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


def test_clopper_pearson_interval():
    from vvrag.eval.metrics import clopper_pearson_interval

    lo, hi = clopper_pearson_interval(9, 9)
    assert lo > 0.6 and hi == 1.0          # 9/9 -> lower bound < 1
    lo, hi = clopper_pearson_interval(0, 9)
    assert lo == 0.0 and hi < 0.4          # 0/9 -> upper bound well below 1
    lo, hi = clopper_pearson_interval(5, 10)
    assert 0.15 < lo < 0.5 < hi < 0.85
    # narrower than 0/9 at n=100
    _lo, hi100 = clopper_pearson_interval(0, 100)
    assert hi100 < 0.05
    assert clopper_pearson_interval(0, 0) == (0.0, 0.0)


def test_gold_box_exact_and_fuzzy():
    words = [WordBox("d", 0, t, (i * 50, 0, i * 50 + 40, 20), 0, 0)
             for i, t in enumerate("The Aurora revenue was 42 million today".split())]
    box = derive_gold_box("Aurora revenue was 42 million", words)
    assert box == (50, 0, 290, 20)
    assert derive_gold_box("zzz-no-match", words) is None
    fuzzy = derive_gold_box("Aurorra", words)  # typo -> fuzzy line match
    assert fuzzy is not None


def test_word_prf_primary():
    # rec 7: word-level P/R/F1 is the main localization metric
    gold = [WordBox("d", 0, t, (i * 10, 0, i * 10 + 8, 10), 0, 0)
            for i, t in enumerate("Nepal Rastra Bank".split())]
    pred_exact = list(gold)
    p, r, f1v = word_precision_recall_f1(pred_exact, gold)
    assert (p, r, f1v) == (1.0, 1.0, 1.0)
    pred_extra = pred_exact + [WordBox("d", 0, "x", (999, 0, 1000, 10), 0, 0)]
    p2, r2, f2 = word_precision_recall_f1(pred_extra, gold)
    assert p2 < 1.0 and r2 == 1.0 and 0.0 < f2 < 1.0
    assert word_precision_recall_f1([], gold) == (0.0, 0.0, 0.0)


def test_union_and_per_line_iou():
    gold = (0, 0, 100, 40)  # two stacked lines
    per_line = [(0, 0, 100, 18), (0, 22, 100, 40)]
    assert union_iou(per_line, gold) > 0.9  # split lines still cover gold
    assert 0.0 < per_line_iou(per_line, gold) < union_iou(per_line, gold)
    assert union_iou([], gold) == 0.0 and per_line_iou(per_line, None) == 0.0


def test_multi_gold_max():
    # rec 7: q5 short + long both correct
    assert max_over_golds("Nepal Rastra Bank",
                          ["Nepal Rastra Bank",
                           "Nepal Rastra Bank reserves all rights"], f1) == 1.0
    assert max_over_golds("x", [], f1) == 0.0


def test_equivalent_answers():
    golds = ["important information", "citations"]
    # longer reader answer contains an acceptable gold -> correct
    assert equivalent_correct("important information or citations", golds) is True
    # equivalent_f1 is the raw max token-F1 (still penalised for extra words)
    assert abs(equivalent_f1("important information or citations", golds) - 2 / 3) < 1e-6
    assert equivalent_correct("completely unrelated", golds) is False
    assert equivalent_correct("anything", []) is False
    assert equivalent_f1("anything", []) == 0.0


def test_equivalent_correct_norm_units():
    from vvrag.eval.metrics import equivalent_correct_norm, equivalent_f1_norm

    # Phase 6: unit variants normalize before matching
    assert equivalent_correct_norm("12.5 bn", ["12.5 billion"]) is True
    assert equivalent_correct_norm("53%", ["53 percent"]) is True
    assert equivalent_correct_norm("1,234", ["1234"]) is True
    assert equivalent_correct_norm("12.5 bn", ["12.5 million"]) is False
    assert equivalent_f1_norm("47%", ["47 percent"]) == 1.0
    assert equivalent_f1_norm("x", []) == 0.0


def test_verified_stats_run9():
    from vvrag.eval.metrics import verified_stats

    rows = [
        {"state": "verified", "equiv_correct": True,
         "pred_boxes": [(0, 0, 1, 1)],
         "examiner": {"confirmed": True, "citation_specific": True},
         "citation_check": {"ok": True},
         "claims": [{"text": "a", "label": "supported",
                     "citation_specific": True},
                    {"text": "b", "label": "partially_supported",
                     "citation_specific": False}],
         "details": {"ok": False}},
        {"state": "verified", "equiv_correct": False,
         "examiner": {"confirmed": True, "citation_specific": False},
         "citation_check": {"ok": False, "numeric_sink": True}},
        {"state": "not_confident", "equiv_correct": False,
         "abstained": False, "pred_sentence_ids": [3]},
        {"state": "nothing_found", "equiv_correct": False, "abstained": True},
        {"state": "not_confident", "equiv_correct": True, "abstained": False,
         "visual_verified": True,
         "citation_check": {"ok": True},
         "citations": {"derived_from_quote": True}},
        {"state": "nothing_found", "unanswerable": True, "equiv_correct": True},
        {"state": "verified", "unanswerable": True, "equiv_correct": False},
        {"error": "boom"},
    ]
    st = verified_stats(rows)
    assert st["n_gold"] == 5 and st["n_probes"] == 2
    assert st["n_verified"] == 2
    assert st["verified_precision"] == 0.5          # 1 correct of 2 verified
    assert st["flag_recall"] == 2 / 3               # 2 of 3 wrong flagged
    assert st["unflagged_wrong_rate"] == 1 / 3
    assert st["answer_coverage"] == 0.8
    assert st["source_accuracy"] == 0.5             # boxes + sentence refs
    assert st["examiner_leniency"] == 0.5           # 1 of 2 confirmed is wrong
    assert st["probe_flag_rate"] == 0.5
    assert st["probe_nothing_found_rate"] == 0.5
    assert st["probe_answered_rate"] == 0.5         # verified probe is shown
    assert st["false_verified_on_probes"] == 0.5
    assert st["examiner_availability"] == 0.4       # 2 of 5 gold rows
    assert st["verified_yield"] == 0.4              # 2 of 5 gold rows
    assert st["n_visual_verified"] == 1
    assert st["visual_verified_yield"] == 0.2
    assert st["visual_verified_precision"] == 1.0
    assert st["false_visual_verified_on_probes"] == 0.0
    assert st["n_derived_citations"] == 1
    assert st["derived_citation_rate"] == 0.2
    assert st["derived_citation_precision"] == 1.0
    assert st["details_block_rate_on_correct"] == 0.5
    assert st["n_citation_checked"] == 3
    assert st["citation_reject_rate"] == 1 / 3
    assert st["n_numeric_sink_rejected"] == 1
    assert st["examiner_citation_specific_rate"] == 0.5
    assert st["n_claims"] == 2
    assert st["n_claims_supported"] == 1
    assert st["claim_support_rate"] == 0.5
    assert st["claim_partial_rate"] == 0.5
    assert st["claim_failure_rate"] == 0.0
    assert st["claim_verification_rate"] == 1.0
    assert st["claim_citation_specific_rate"] == 0.5
    empty = verified_stats([])
    assert empty["verified_precision"] is None and empty["n_gold"] == 0


def test_verified_stats_examiner_only_subset():
    from vvrag.eval.metrics import verified_stats

    rows = [
        {"state": "verified", "equiv_correct": True, "examiner_ran": True,
         "pred_boxes": [(0, 0, 1, 1)], "examiner": {"confirmed": True}},
        {"state": "not_confident", "equiv_correct": True,
         "examiner_ran": False},
        {"state": "not_confident", "equiv_correct": False,
         "examiner_ran": False},
        {"state": "not_confident", "equiv_correct": False,
         "examiner_ran": True, "examiner": {"confirmed": False}},
        {"state": "nothing_found", "equiv_correct": False, "abstained": True,
         "examiner_ran": False},
        {"state": "nothing_found", "unanswerable": True, "equiv_correct": False,
         "examiner_ran": False},
    ]
    st = verified_stats(rows, examiner_only=True)
    assert st["n_gold"] == 5
    assert st["n_gold_examinable"] == 2
    assert st["n_examiner_unavailable"] == 3
    assert st["n_verified"] == 1
    assert st["verified_precision"] == 1.0
    assert st["verified_precision_cp95"] is not None
    assert st["flag_recall"] == 1.0          # the one examinable wrong row
    assert st["unflagged_wrong_rate"] == 0.0
    assert st["answer_coverage"] == 0.8      # 4 of 5 gold rows are shown
    assert st["source_accuracy"] == 0.25     # 1 of 4 shown rows has boxes
    assert st["examiner_leniency"] == 0.0    # the confirmed row is correct
    # Probes stay on the full probe set: a never-shown probe needs no examiner.
    assert st["n_probes"] == 1
    assert st["probe_flag_rate"] == 1.0
    assert st["false_verified_on_probes"] == 0.0

    all_st = verified_stats(rows)
    assert all_st["n_gold_examinable"] == 5
    assert all_st["n_examiner_unavailable"] == 0
    assert all_st["answer_coverage"] == 0.8  # 4 of 5 gold rows are shown


def test_partial_answer_credit_is_report_only():
    from vvrag.eval.metrics import partial_answer_correct
    assert partial_answer_correct(
        "National Monitoring Committee",
        ["National Monitoring Committee, Oversight at Central Level for all projects"])
    assert not partial_answer_correct("11", ["12"])
    assert not partial_answer_correct("business", ["business model"])
