"""Eval CLI (Section 10): one command reproduces tables + figures.

Definition-of-done command:
    python -m vvrag.eval.run_eval --config config.yaml --all
Runs the 4 arms + core ablations on the (synthetic) debug set with CIs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from vvrag.config import load_config
from vvrag.eval.arms import ARMS, ArmConfig, OfflineSystem
from vvrag.eval.datasets import build_synthetic_set, deck_split
from vvrag.eval.report import summarize, write_run
from vvrag.ingest.run_ingest import run_ingest
from vvrag.logging_utils import get_logger

log = get_logger(__name__)


def arm_configs(args) -> list[ArmConfig]:
    base = dict(n_samples=args.n_samples, heat_mode=args.heat_mode,
                scorer=args.scorer, alpha=args.alpha, delta=args.delta,
                fixed_tau=args.fixed_tau, reader_acc=args.reader_acc,
                prefetch_mult=args.prefetch_mult)
    return [
        ArmConfig(name="baseline", grounding=False, gate="none", **base),
        ArmConfig(name="grounded", grounding=True, gate="none", **base),
        ArmConfig(name="verified-fixed", grounding=True, gate="fixed", **base),
        ArmConfig(name="verified-conformal", grounding=True, gate="conformal", **base),
    ]


def run_arm(cfg: ArmConfig, index_dir: str, items: dict, work_dir: str):
    """items: {split: [QAItem]}. Fit scorer on train, tau on calib, test gated."""
    sys = OfflineSystem(index_dir, cfg, work_dir=f"{work_dir}/{cfg.name}")
    sys.set_distractor_pool(items["train"] + items["calib"] + items["test"])
    out: dict[str, list] = {}
    for split in ("train", "calib", "test"):
        out[split] = [sys.answer_item(it) for it in items[split]]
        log.info("%s/%s: %d done", cfg.name, split, len(out[split]))
    if cfg.scorer == "learned":
        sys.fit_scorer(out["train"] + out["calib"])
        for split in ("train", "calib", "test"):
            sys.rescore(out[split])
    if cfg.gate == "conformal":
        sys.calibrate(out["calib"])
        sys.apply_gate_records(out["test"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="VVRAG eval harness")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "slidevqa"])
    ap.add_argument("--n-docs", type=int, default=10)
    ap.add_argument("--pages-per-doc", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--heat-mode", default="max", choices=["max", "mean"])
    ap.add_argument("--scorer", default="learned", choices=["learned", "hand"])
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--fixed-tau", type=float, default=0.5)
    ap.add_argument("--reader-acc", type=float, default=0.7)
    ap.add_argument("--prefetch-mult", type=int, default=4)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--out", default="")
    ap.add_argument("--all", action="store_true",
                    help="full debug matrix (Definition of Done one-command run)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out or f"results/runs/{stamp}")
    eval_data = run_dir / "eval_data"
    n_docs = max(args.n_docs, 16) if args.all else args.n_docs  # --all needs calib mass
    eset = build_synthetic_set(eval_data, n_docs=n_docs,
                               pages_per_doc=args.pages_per_doc, seed=args.seed)
    index_dir = str(eval_data / "index")
    run_ingest(eval_data / "raw", index_dir, dpi=cfg.render.dpi, embedder="fake",
               skip_ocr=True, qdrant_path=f"{index_dir}/qdrant")

    train, calib, test = deck_split(eset.items, calib_frac=cfg.eval.calib_frac,
                                    seed=cfg.eval.seed)
    log.info("split by deck: train=%d calib=%d test=%d", len(train), len(calib), len(test))
    if args.dataset == "slidevqa":
        log.warning("slidevqa needs GPU index + reader key; running synthetic instead (documented)")

    summaries, records_by_arm = [], {}
    cfgs = [c for c in arm_configs(args) if c.name in args.arms]
    if args.all:
        # core ablations (Section 10.4): heat mean, k=5, alpha grid, hand scorer
        cfgs += [ArmConfig(name="abl-heat-mean", grounding=True, gate="conformal",
                           heat_mode="mean", n_samples=args.n_samples, scorer=args.scorer,
                           alpha=args.alpha, delta=args.delta, reader_acc=args.reader_acc),
                 ArmConfig(name="abl-k5", grounding=True, gate="conformal",
                           heat_mode=args.heat_mode, n_samples=5, scorer=args.scorer,
                           alpha=args.alpha, delta=args.delta, reader_acc=args.reader_acc),
                 ArmConfig(name="abl-alpha05", grounding=True, gate="conformal",
                           heat_mode=args.heat_mode, n_samples=args.n_samples, scorer=args.scorer,
                           alpha=0.05, delta=args.delta, reader_acc=args.reader_acc),
                 ArmConfig(name="abl-hand", grounding=True, gate="conformal",
                           heat_mode=args.heat_mode, n_samples=args.n_samples, scorer="hand",
                           alpha=args.alpha, delta=args.delta, reader_acc=args.reader_acc)]
    for ac in cfgs:
        out = run_arm(ac, index_dir, {"train": train, "calib": calib, "test": test},
                      work_dir=str(run_dir / "work"))
        summaries.append(summarize(out["test"], ac.name))
        records_by_arm[ac.name] = out["test"]
        s = summaries[-1]
        log.info("%s: F1=%.3f cover=%.3f sel_acc=%.3f tau=%s",
                 ac.name, s["answer_f1"]["mean"], s["coverage"],
                 s["selective_accuracy"], f"{s['tau']:.3f}")

    write_run(summaries, records_by_arm, run_dir)
    (run_dir / "meta.json").write_text(json.dumps({
        "dataset": "synthetic-debug", "n_docs": n_docs,
        "split": {"train": len(train), "calib": len(calib), "test": len(test)},
        "note": "Synthetic stand-ins: fake embeddings, gold-patch queries, "
                "simulated reader. Real SlideVQA/GPU/key runs pending (decisions.md).",
    }, indent=2), encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
