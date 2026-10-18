"""Live-label gate calibration (P2/D): --from-jsonl fit + target-error tau.

Builds a tiny synthetic live_labels.jsonl (features + correct) that is separable,
runs scripts/calibrate_gate.py --from-jsonl, and checks the artifacts.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from vvrag.verify.features import build_features

REPO = Path(__file__).resolve().parent.parent


def _make_labels(path: Path, n: int = 40) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        correct = i % 2 == 0
        # correct rows: high agreement + quote_match; wrong rows: low
        agr = rng.uniform(0.8, 1.0) if correct else rng.uniform(0.0, 0.3)
        qm = rng.uniform(0.85, 1.0) if correct else rng.uniform(0.0, 0.3)
        feats = build_features(agreement=agr, entail_min=rng.random(),
                               retrieval_margin=rng.random(),
                               heat_concentration=rng.random(), n_claims=2,
                               crop_has_text=True, localize_score=rng.random(),
                               evidence_valid=correct, quote_match=qm)
        rows.append({"qid": f"q{i}", "correct": bool(correct), "score": None,
                     "features": feats.to_dict()})
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_calibrate_from_jsonl_writes_artifacts(tmp_path):
    labels = tmp_path / "labels.jsonl"
    _make_labels(labels)
    out_dir = tmp_path / "gate"
    res = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "calibrate_gate.py"),
         "--from-jsonl", str(labels), "--out-dir", str(out_dir),
         "--target-error", "0.10"],
        capture_output=True, text=True, cwd=str(REPO))
    assert res.returncode == 0, res.stdout + res.stderr
    tau_path = out_dir / "tau.json"
    assert tau_path.exists()
    tau = json.loads(tau_path.read_text(encoding="utf-8"))
    assert tau["n_source"] == "live-jsonl"
    assert tau["target_error"] == 0.10
    assert 0.0 <= tau["tau"] <= 1.0
    assert (out_dir / "scorer.pkl").exists()
