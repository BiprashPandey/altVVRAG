"""Run 8 Phase E: pre-registered selection of the hybrid-arm items.

Rule (fixed BEFORE any Run 8 measurement, using only deterministic keyword
rankings + Run 7 failure labels):

  1. retrieval-miss (cap 15): the best acceptable-page keyword rank is > 1 or
     the page is outside the keyword top-5. These are exactly the items where
     the visual retriever can plausibly change the outcome.
  2. visual-abstain (cap 7): Run 7 abstained on this item (reader
     UNANSWERABLE or support veto) and the gold answer is answer_visual_only.
  3. multi-hop fill: remaining slots (to 25 total) go to bridged/multi-hop
     items in gold order.

Output: results/run8_hybrid_items.json  [{qid, question, reason, page_rank}]
The rule is deterministic and documented so the Run 8 hybrid arm has no
selection leakage from hybrid scores.

Usage:
    python scripts/select_hybrid_items.py --out results/run8_hybrid_items.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from vvrag.ui.qa import keyword_rank  # noqa: E402

TARGET_N = 25
MISS_CAP = 15
VISUAL_CAP = 7


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=str(REPO / "data" / "index"))
    ap.add_argument("--gold", default=str(REPO / "data" / "gold.json"))
    ap.add_argument("--run7", default=str(REPO / "results" / "demo"
                                        / "testrun7_20261018_192723" / "rows.json"))
    ap.add_argument("--out", default=str(REPO / "results" / "run8_hybrid_items.json"))
    args = ap.parse_args()

    con = sqlite3.connect(str(Path(args.index) / "meta.db"))
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    run7 = json.loads(Path(args.run7).read_text(encoding="utf-8"))
    r7_by_qid = {r.get("qid"): r for r in run7}

    infos = []
    for item in gold:
        qid = item["qid"]
        ranking = keyword_rank(item["question"], con, limit=None)
        rank_map = {(d, p): i + 1 for i, (d, p, _s, _w) in enumerate(ranking)}
        acceptable = [(pp["doc_id"], pp["page"])
                      for pp in (item.get("acceptable_pages") or [])]
        page_rank = min((rank_map.get(k, 10 ** 6) for k in acceptable),
                        default=rank_map.get((item["doc_id"], int(item["page"])),
                                             10 ** 6))
        r7 = r7_by_qid.get(qid, {})
        infos.append({
            "qid": qid,
            "question": item["question"],
            "page_rank": page_rank,
            "visual_only": bool(item.get("answer_visual_only")),
            "bridged": bool(item.get("bridged_question")),
            "run7_abstained": bool(r7.get("abstained")),
            "run7_reason": r7.get("abstain_reason", ""),
        })
    con.close()

    selected: list[dict] = []
    chosen: set[str] = set()

    def _take(info: dict, reason: str) -> None:
        selected.append({"qid": info["qid"], "question": info["question"],
                         "reason": reason, "page_rank": info["page_rank"]})
        chosen.add(info["qid"])

    for info in infos:
        if len(selected) >= MISS_CAP:
            break
        if info["page_rank"] > 1:
            _take(info, f"retrieval-miss (best acceptable-page keyword rank "
                        f"{info['page_rank'] if info['page_rank'] < 10**6 else '>5'})")

    n_vis = 0
    for info in infos:
        if n_vis >= VISUAL_CAP or len(selected) >= TARGET_N:
            break
        if info["qid"] in chosen:
            continue
        if info["run7_abstained"] and info["visual_only"]:
            _take(info, f"visual-abstain (run7: {info['run7_reason'][:60]})")
            n_vis += 1

    for info in infos:
        if len(selected) >= TARGET_N:
            break
        if info["qid"] in chosen:
            continue
        if info["bridged"]:
            _take(info, "multi-hop fill")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selected, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    from collections import Counter
    print(f"WROTE {out} ({len(selected)} items)")
    print("by reason:", dict(Counter(s["reason"].split(" (")[0] for s in selected)))
    for s in selected:
        print(f"  p{s['page_rank']:>2}  {s['reason'][:64]:<64} {s['qid'][-8:]}")


if __name__ == "__main__":
    main()
