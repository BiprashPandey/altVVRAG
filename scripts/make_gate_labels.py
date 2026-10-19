"""Convert test-run rows.json into gate-calibration JSONL (Phase 4, v2).

v2 (Run 8 Phase D) fixes two label bugs and widens the population:

1. Correctness is recomputed from the reader's ANSWER TEXT vs the gold(s).
   The old runs stored `equiv_correct = (not abstained) and equivalent(...)`,
   so a gate-vetoed row with a correct answer (Run 7 idx 0/37/38) was labeled
   incorrect. Recomputing makes vetoed-but-correct rows positive labels and
   vetoed-but-wrong rows negative ones.
2. Rows where the reader itself returned UNANSWERABLE are skipped (the gate
   never saw an answer), but vetoed rows (reader answered, gate abstained) are
   now included as gate examples.
3. Features are recomputed as the current 15-column vector via build_features;
   legacy 9-col (runs 5/6) and 12-col (run 7) rows are both accepted.

Usage:
    python scripts/make_gate_labels.py --rows results/demo/testrun5_*/rows.json \
        results/demo/testrun6_*/rows.json results/demo/testrun7_*/rows.json \
        --out results/gate_labels_runs567.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vvrag.eval.metrics import equivalent_correct_norm  # noqa: E402
from vvrag.reader.claims import normalize_answer  # noqa: E402
from vvrag.verify.features import build_features  # noqa: E402
from vvrag.verify.support import answer_type_match, qa_support  # noqa: E402

_STATUS_TO_RESOLVED = {"none": 1.0, "ok": 1.0, "partial": 0.5, "unresolved": 0.0}


def _golds(row: dict) -> list[str]:
    out: list[str] = []
    g = row.get("gold_answer") or row.get("gold") or row.get("answer_gold")
    if g and normalize_answer(str(g)) != "unanswerable":
        out.append(str(g))
    for a in (row.get("acceptable_answers") or row.get("gold_answers") or []):
        if a and str(a) not in out:
            out.append(str(a))
    return out


def _visual_margin(row: dict) -> float:
    vt = ((row.get("retrieval_debug") or {}).get("visual_top5") or [])
    if len(vt) >= 2:
        return float(vt[0].get("score", 0.0)) - float(vt[1].get("score", 0.0))
    return 0.0


def _citation_resolved(row: dict, cited: list[dict]) -> float:
    cit = row.get("citations")
    if isinstance(cit, dict) and cit.get("status") in _STATUS_TO_RESOLVED:
        return _STATUS_TO_RESOLVED[cit["status"]]
    loc = row.get("localize") or {}
    if loc.get("citations_valid") is not None:
        return 1.0 if loc.get("citations_valid") else 0.0
    return 1.0 if cited else 1.0  # nothing cited -> neutral (no hallucinated id)


def convert_row(row: dict) -> dict | None:
    if row.get("error"):
        return None
    answer = str(row.get("answer") or "")
    if not answer or normalize_answer(answer) == "unanswerable":
        return None  # the gate never saw an answer here
    if row.get("unanswerable"):
        # probe rows: an answered probe is a false answer -> negative label
        golds: list[str] = []
    else:
        golds = _golds(row)
        if not golds:
            return None
    correct = equivalent_correct_norm(answer, golds)

    feats_old = row.get("features") or {}
    cited = [c for c in (row.get("cited_sentences") or [])
             if int(c.get("sent_id", -1)) >= 0 or c.get("visual")]
    texts = [c.get("text", "") for c in cited
             if int(c.get("sent_id", -1)) >= 0]
    entail = feats_old.get("entail_prob")
    question = str(row.get("question") or "")
    supported, sup_score, sup_method = qa_support(
        question, answer, texts, entail)

    base = build_features().to_dict()  # 15-col canonical defaults
    for k, v in feats_old.items():
        if k in base and v is not None:
            base[k] = v
    base["answer_supported"] = 1.0 if supported else 0.0
    base["cited_is_top1"] = 1.0 if cited else 0.0
    base["visual_margin"] = _visual_margin(row)
    base["citation_resolved"] = _citation_resolved(row, cited)
    atm = answer_type_match(question, answer)
    base["answer_type_match"] = 1.0 if atm is None else (1.0 if atm else 0.0)
    base["support_type"] = (1.0 if sup_method.endswith("+type")
                            else 0.5 if sup_method == "disabled" else 0.0)

    retrieval = row.get("retrieval") or []
    top1 = ((retrieval[0].get("doc_id"), retrieval[0].get("page"))
            if retrieval else None)
    ev = row.get("evidence") or {}
    cited_pages = set()
    for c in cited:
        d = c.get("doc_id", ev.get("doc_id"))
        p = c.get("page", ev.get("page"))
        if d is not None and p is not None:
            cited_pages.add((d, p))
    if top1 is not None and cited_pages:
        base["cited_is_top1"] = 1.0 if top1 in cited_pages else 0.0

    return {
        "qid": row.get("qid", ""),
        "question": question,
        "gold": golds[0] if golds else "",
        "answer": answer,
        "correct": bool(correct),
        "abstained": bool(row.get("abstained")),
        "accepted": not bool(row.get("abstained")),
        "score": row.get("score"),
        "tier": row.get("tier") or "",
        "support_method": sup_method,
        "support_score": round(float(sup_score), 4),
        "unanswerable": bool(row.get("unanswerable")),
        "features": base,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", nargs="+", required=True,
                    help="one or more rows.json files")
    ap.add_argument("--out", default="results/gate_labels.jsonl")
    args = ap.parse_args()

    rows_out = []
    for path in args.rows:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        added = 0
        for row in data:
            conv = convert_row(row)
            if conv is not None:
                rows_out.append(conv)
                added += 1
        print(f"{path}: {len(data)} rows in, {added} gate labels, "
              f"running total {len(rows_out)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    pos = sum(1 for r in rows_out if r["correct"])
    neg = len(rows_out) - pos
    print(f"WROTE {out} ({len(rows_out)} labeled rows: "
          f"{pos} correct / {neg} incorrect)")
    vetoed_correct = sum(1 for r in rows_out
                         if r["abstained"] and r["correct"])
    print(f"  vetoed-but-correct rows now usable: {vetoed_correct}")


if __name__ == "__main__":
    main()
