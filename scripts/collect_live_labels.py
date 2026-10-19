#!/usr/bin/env python
"""Collect live correctness labels for gate calibration (Priority 2 / Phase D).

Runs the live pipeline (Gemini reader + ColQwen index) on a gold set and emits
one JSONL row per question with the verifier FEATURES + whether the answer was
correct. calibrate_gate.py --from-jsonl then fits the scorer + tau on these
real labels instead of the synthetic offline stand-in.

Gold JSON format (see scripts/prepare_slidevqa.py or testRun gold lists):
  [{"qid", "question", "answer", "acceptable_answers": [], "doc_id", "page"}]

Usage:
  python scripts/collect_live_labels.py --index data/slidevqa/index \
      --gold data/slidevqa/gold.json --out results/live_labels.jsonl
  python scripts/collect_live_labels.py --index data/index \
      --gold results/demo/testrun4_*/gold.json   # reuse a prior gold list
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from vvrag.eval.metrics import equivalent_correct  # noqa: E402
from vvrag.logging_utils import get_logger  # noqa: E402

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect live gate-calibration labels")
    ap.add_argument("--index", required=True)
    ap.add_argument("--gold", required=True, help="gold JSON list")
    ap.add_argument("--out", default="results/live_labels.jsonl")
    ap.add_argument("--reader-mode", default="live")
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default="")
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--grounding", default="span", choices=["span", "heatmap"])
    ap.add_argument("--gate", default="none", help="gate mode for the recorded run")
    ap.add_argument("--retrieval-mode", default="",
                    help="keyword|visual|hybrid ('' = config default)")
    ap.add_argument("--work-dir", default="results/live_labels_work")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between questions")
    args = ap.parse_args()

    from vvrag.ui.qa import answer_question

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    if args.limit:
        gold = gold[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with out.open("w", encoding="utf-8") as f:
        for i, g in enumerate(gold):
            qid = g.get("qid", str(i))
            golds = [g["answer"], *g.get("acceptable_answers", [])]
            try:
                r = answer_question(
                    g["question"], args.index, f"{args.work_dir}/{qid}",
                    reader_mode=args.reader_mode, provider=args.provider,
                    model=args.model, n_samples=args.n_samples,
                    grounding=args.grounding, gate=args.gate,
                    doc_id=g.get("doc_id"),
                    retrieval_mode=args.retrieval_mode or None)
            except Exception as e:  # noqa: BLE001
                log.warning("%s failed: %s", qid, e)
                continue
            correct = (not r["abstained"]) and equivalent_correct(r["answer"], golds)
            row = {
                "qid": qid, "question": g["question"], "gold": g["answer"],
                "answer": r["answer"], "correct": bool(correct),
                "abstained": bool(r["abstained"]), "score": r.get("score"),
                "features": r.get("features"), "abstain_reason": r.get("abstain_reason"),
                "verified": r.get("verified"), "support": r.get("support"),
                "retrieval": r.get("retrieval_debug"),
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            log.info("[%d/%d] %s correct=%s abstained=%s score=%s",
                     i + 1, len(gold), qid, correct, r["abstained"], r.get("score"))
            if args.sleep:
                time.sleep(args.sleep)

    n = len(rows)
    acc = sum(1 for r in rows if r["correct"]) / max(1, n)
    log.info("wrote %d rows -> %s (answered accuracy %.3f)", n, out, acc)


if __name__ == "__main__":
    main()
