"""Calibrate the confidence gate (suggestion #5).

Trains the LogisticScorer on offline eval records (features -> correctness)
and selects a conformal tau, then writes the artifacts the UI/qa.py loads:

    data/gate/scorer.pkl   (LogisticScorer)
    data/gate/tau.json     ({"tau": ..., "alpha": ..., "delta": ...})

Usage:
    python scripts/calibrate_gate.py --n-docs 16
    python scripts/calibrate_gate.py --scorer hand   # skip fitting, still pick tau

    # P2/D: calibrate on LIVE labels (features + correct) instead of synthetic
    python scripts/collect_live_labels.py --index data/slidevqa/index \
        --gold data/slidevqa/gold.json --out results/live_labels.jsonl
    python scripts/calibrate_gate.py --from-jsonl results/live_labels.jsonl \
        --target-error 0.10     # tau from the coverage/accuracy curve

The offline harness uses the FakePatchEmbedder index + SimulatedReader, so it
runs on CPU with no keys. Real SlideVQA calibration uses --from-jsonl after a
live collect_live_labels run. This only calibrates; it never calls a live reader.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO = Path(__file__).resolve().parent.parent

from vvrag.eval.arms import ArmConfig, OfflineSystem  # noqa: E402
from vvrag.eval.datasets import build_synthetic_set, deck_split  # noqa: E402
from vvrag.gate.scorer import features_to_matrix  # noqa: E402
from vvrag.ingest.run_ingest import run_ingest  # noqa: E402
from vvrag.logging_utils import get_logger  # noqa: E402

log = get_logger(__name__)


def _fit_from_jsonl(args) -> None:
    """Fit scorer + tau on LIVE labels (features + correct) from collect_live_labels."""
    import numpy as np

    from vvrag.gate.conformal import select_tau, select_tau_target_error
    from vvrag.gate.scorer import LogisticScorer, features_to_matrix
    from vvrag.verify.features import FEATURE_ORDER

    rows = [json.loads(ln) for ln in Path(args.from_jsonl).read_text(
        encoding="utf-8").splitlines() if ln.strip()]
    rows = [r for r in rows if r.get("features")]
    if len(rows) < 10:
        raise SystemExit(f"need >=10 labeled rows, got {len(rows)}")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(rows))
    n_tr = int(0.6 * len(rows))
    tr = [rows[i] for i in order[:n_tr]]
    ca = [rows[i] for i in order[n_tr:]]

    X_tr = features_to_matrix([r["features"] for r in tr])
    y_tr = np.array([1 if r["correct"] else 0 for r in tr], dtype=int)
    X_ca = features_to_matrix([r["features"] for r in ca])
    y_ca_err = np.array([0 if r["correct"] else 1 for r in ca], dtype=int)

    scorer = LogisticScorer()
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.scorer == "learned" and len(set(y_tr.tolist())) > 1:
        scorer.fit(X_tr, y_tr)
        scorer.save(out_dir / "scorer.pkl")
        log.info("fitted live scorer on %d rows -> %s", len(tr), out_dir / "scorer.pkl")
        scores = scorer.predict_score(X_ca)
    else:
        log.warning("single-class train or --scorer hand; no scorer.pkl written")
        scores = X_ca[:, FEATURE_ORDER.index("quote_match")]  # crude fallback

    if args.target_error is not None:
        tau, info = select_tau_target_error(scores, y_ca_err, args.target_error,
                                            min_n=args.min_n)
        log.info("tau=%.4f from target-error=%.2f (cov=%.2f)",
                 tau, args.target_error, info.get("selected_cov", 0.0))
    else:
        tau, info = select_tau(scores, y_ca_err, args.alpha, args.delta, min_n=args.min_n)
        log.info("conformal tau=%.4f (alpha=%.2f delta=%.2f)", tau, args.alpha, args.delta)

    (out_dir / "tau.json").write_text(json.dumps({
        "tau": float(tau) if tau != float("inf") else 1.0,
        "alpha": args.alpha, "delta": args.delta,
        "target_error": args.target_error,
        "n_calib": len(ca), "n_train": len(tr), "n_source": "live-jsonl",
        "scorer": args.scorer, "feature_order": list(FEATURE_ORDER),
    }, indent=2), encoding="utf-8")
    log.info("wrote %s", out_dir / "tau.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate the gate scorer + tau")
    ap.add_argument("--from-jsonl", default="",
                    help="fit on LIVE labels from collect_live_labels.py instead "
                         "of the synthetic offline set")
    ap.add_argument("--target-error", type=float, default=None,
                    help="with --from-jsonl: pick tau from the coverage/accuracy "
                         "curve so accepted error <= this (else conformal UCB)")
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--n-docs", type=int, default=16)
    ap.add_argument("--pages-per-doc", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--reader-acc", type=float, default=0.7)
    ap.add_argument("--scorer", choices=["learned", "hand"], default="learned")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--out-dir", default="data/gate")
    args = ap.parse_args()

    if args.from_jsonl:
        _fit_from_jsonl(args)
        return

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = REPO / "results" / "calib"

    eset = build_synthetic_set(data_dir, n_docs=max(args.n_docs, 16),
                               pages_per_doc=args.pages_per_doc, seed=args.seed)
    index_dir = str(data_dir / "index")
    run_ingest(data_dir / "raw", index_dir, dpi=150, embedder="fake",
               skip_ocr=True, qdrant_path=f"{index_dir}/qdrant")

    train, calib, test = deck_split(eset.items, calib_frac=0.5, seed=args.seed)
    log.info("split: train=%d calib=%d test=%d", len(train), len(calib), len(test))

    cfg = ArmConfig(name="calibrate", grounding=True, gate="none",
                    scorer=args.scorer, n_samples=args.n_samples,
                    reader_acc=args.reader_acc, use_real_nli=False)
    sys = OfflineSystem(index_dir, cfg, work_dir=str(data_dir / "work"))
    sys.set_distractor_pool(train + calib + test)

    train_recs = [sys.answer_item(it) for it in train + calib]
    calib_recs = [sys.answer_item(it) for it in calib]
    test_recs = [sys.answer_item(it) for it in test]

    if args.scorer == "learned":
        sys.fit_scorer(train_recs)
        sys.rescore(train_recs)
    sys.rescore(calib_recs)
    sys.rescore(test_recs)

    tau = sys.calibrate(calib_recs)
    log.info("calibrated tau=%.4f (alpha=%.2f delta=%.2f)", tau, args.alpha, args.delta)

    if args.scorer == "learned" and sys.scorer._clf is not None:
        scorer_path = out_dir / "scorer.pkl"
        sys.scorer.save(scorer_path)
        log.info("wrote %s", scorer_path)
    else:
        log.warning("scorer not fitted (single-class train?); no scorer.pkl written")

    tau_path = out_dir / "tau.json"
    tau_path.write_text(json.dumps({
        "tau": float(tau) if tau != float("inf") else 1.0,
        "alpha": args.alpha, "delta": args.delta,
        "n_calib": len(calib_recs), "n_train": len(train_recs),
        "scorer": args.scorer, "feature_order": __import__(
            "vvrag.verify.features", fromlist=["FEATURE_ORDER"]).FEATURE_ORDER,
    }, indent=2), encoding="utf-8")
    log.info("wrote %s", tau_path)

    X = features_to_matrix([r.result.features for r in test_recs])
    log.info("calibration done: test feature matrix %s", X.shape)


if __name__ == "__main__":
    main()
