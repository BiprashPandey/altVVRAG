"""Eval reporting (Section 10.6): summaries, tables, figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vvrag.eval.arms import PredRecord
from vvrag.eval.metrics import (
    aurc,
    bootstrap_ci,
    center_inside,
    iou,
    risk_coverage_curve,
    selective_stats,
)


def summarize(records: list[PredRecord], arm: str) -> dict:
    em = np.array([r.correct_em for r in records])
    f1s = np.array([r.correct_f1 for r in records])
    correct = f1s >= 0.5
    answered = np.array([not r.result.abstained for r in records])
    scores = np.array([r.result.score for r in records])
    sel = selective_stats(answered, correct)
    loc = [r for r in records if r.gold_box is not None and r.result.evidence.boxes]
    ious = np.array([iou(r.result.evidence.boxes[0], r.gold_box) for r in loc])
    lat = np.array([r.result.timings.get("total", 0.0) for r in records])
    em_m, em_lo, em_hi = bootstrap_ci(em)
    f1_m, f1_lo, f1_hi = bootstrap_ci(f1s)
    return {
        "arm": arm,
        "n": len(records),
        "answer_em": {"mean": em_m, "ci95": [em_lo, em_hi]},
        "answer_f1": {"mean": f1_m, "ci95": [f1_lo, f1_hi]},
        "retrieval_hit3": float(np.mean([r.hit3 for r in records])) if records else 0.0,
        "retrieval_mrr": float(np.mean([r.mrr for r in records])) if records else 0.0,
        "n_localizable": len(loc),
        "iou_mean": float(ious.mean()) if len(ious) else 0.0,
        "iou_hit03": float((ious >= 0.3).mean()) if len(ious) else 0.0,
        "iou_hit05": float((ious >= 0.5).mean()) if len(ious) else 0.0,
        "center_in_box": float(np.mean([center_inside(r.result.evidence.boxes[0], r.gold_box) for r in loc])) if loc else 0.0,
        **sel,
        "aurc": aurc(scores, (~correct).astype(int)) if len(records) else 0.0,
        "realized_accepted_error": float((~correct[answered]).mean()) if answered.any() else 0.0,
        "latency_mean_s": float(lat.mean()) if len(lat) else 0.0,
        "latency_p95_s": float(np.percentile(lat, 95)) if len(lat) else 0.0,
        "api_calls_per_q": float(np.mean([r.result.api_calls for r in records])) if records else 0.0,
        "tau": float(records[0].result.tau) if records else 0.0,
    }


def main_table_rows(summaries: list[dict]) -> tuple[str, str]:
    cols = ["arm", "n", "EM", "F1", "Hit@3", "IoU>=.3", "cover", "sel_acc", "conf_wrong", "AURC"]
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    tex = ["\\begin{tabular}{lccccccccc}", "\\toprule",
           "arm & n & EM & F1 & Hit@3 & IoU$\\geq$.3 & cover & sel.acc & conf.wr & AURC \\\\", "\\midrule"]
    for s in summaries:
        em, f1 = s["answer_em"]["mean"], s["answer_f1"]["mean"]
        row = [s["arm"], str(s["n"]), f"{em:.3f}", f"{f1:.3f}",
               f"{s['retrieval_hit3']:.3f}", f"{s['iou_hit03']:.3f}",
               f"{s['coverage']:.3f}", f"{s['selective_accuracy']:.3f}",
               f"{s['confident_wrong_rate']:.3f}", f"{s['aurc']:.3f}"]
        md.append("| " + " | ".join(row) + " |")
        tex.append(" & ".join(row) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(md), "\n".join(tex)


def write_run(summaries: list[dict], records_by_arm: dict[str, list[PredRecord]],
              run_dir: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    preds = []
    for arm, recs in records_by_arm.items():
        for r in recs:
            d = r.result.to_dict()
            d.update({"arm": arm, "qid": r.item.question_id, "deck": r.item.deck_id,
                      "gold": r.item.gold_answer, "f1": r.correct_f1, "em": r.correct_em,
                      "gold_box": list(r.gold_box) if r.gold_box else None, "iou": r.iou,
                      "hit3": r.hit3})
            preds.append(d)
    (run_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(p) for p in preds), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    md_table, tex_table = main_table_rows(summaries)
    (run_dir / "summary.md").write_text("# Main results\n\n" + md_table + "\n", encoding="utf-8")
    (run_dir / "main_table.tex").write_text(tex_table + "\n", encoding="utf-8")

    # risk-coverage curves, all arms on one plot
    plt.figure()
    for arm, recs in records_by_arm.items():
        sc = np.array([r.result.score for r in recs])
        er = np.array([0 if r.correct_f1 >= 0.5 else 1 for r in recs])
        c, risk = risk_coverage_curve(sc, er)
        plt.plot(c, risk, marker=".", label=arm)
    plt.xlabel("coverage")
    plt.ylabel("selective error")
    plt.legend()
    plt.savefig(fig_dir / "risk_coverage.png", dpi=120)
    plt.close()

    # IoU histogram (full system arm if present)
    full = records_by_arm.get("verified-conformal", next(iter(records_by_arm.values())))
    ious = [r.iou for r in full if r.iou is not None]
    plt.figure()
    plt.hist(ious, bins=20)
    plt.xlabel("IoU")
    plt.savefig(fig_dir / "iou_hist.png", dpi=120)
    plt.close()

    # calibration scatter: score vs correctness
    plt.figure()
    for arm, recs in records_by_arm.items():
        plt.scatter([r.result.score for r in recs],
                    [r.correct_f1 for r in recs], s=8, label=arm, alpha=0.6)
    plt.xlabel("confidence score")
    plt.ylabel("answer F1")
    plt.legend()
    plt.savefig(fig_dir / "calibration.png", dpi=120)
    plt.close()
    return run_dir
