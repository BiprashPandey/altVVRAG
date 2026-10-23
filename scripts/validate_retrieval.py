"""Offline retrieval validation on the gold set (no API, no GPU needed).

Compares, per gold item:
  keyword  -- OCR keyword overlap (current live path)
  visual   -- exact ColQwen MaxSim against every page's stored patch vectors
  hybrid   -- reciprocal-rank fusion of the two (Phase 1 live path)

Metrics: page_hit@1/@3/@5, deck_hit@5, MRR per mode. Visual/hybrid need the
query cache (scripts/precompute_query_vectors.py); without it this script
reports keyword-only and says exactly what is missing.

Usage:
    python scripts/validate_retrieval.py
    python scripts/validate_retrieval.py --gold data/gold.json --index data/index
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vvrag.ingest.index import connect_sqlite  # noqa: E402
from vvrag.retrieve.fusion import (rank_order, rrf_fuse,  # noqa: E402
                                   visual_page_scores)
from vvrag.retrieve.query_vectors import QueryVectorStore  # noqa: E402
from vvrag.ui.qa import keyword_rank  # noqa: E402


def _hit_at(order: list[tuple[str, int]], gold: tuple[str, int], n: int) -> bool:
    return gold in order[:n]


def _rank_of(order: list[tuple[str, int]], gold: tuple[str, int]) -> int | None:
    return order.index(gold) + 1 if gold in order else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--gold", default="data/gold.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--query-cache", default="data/query_cache",
                    help="query-vector cache directory")
    args = ap.parse_args()

    index = Path(args.index)
    gold_path = Path(args.gold)
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    if isinstance(gold, dict):
        gold = gold.get("items", [])

    con = connect_sqlite(index / "meta.db")
    pages = [(r[0], int(r[1])) for r in
             con.execute("SELECT doc_id, page FROM pages ORDER BY doc_id, page").fetchall()]
    store = QueryVectorStore(args.query_cache)
    n_q = len(gold)
    cached, _ = store.coverage([g["question"] for g in gold])
    print(f"index pages={len(pages)}  gold={n_q}  query cache={cached}/{n_q}")

    modes = ["keyword"] + (["visual", "hybrid"] if cached else [])
    if not cached:
        print("NOTE: no cached query vectors -> visual/hybrid skipped. "
              "Run scripts/precompute_query_vectors.py on a GPU box first.")

    rows: list[dict] = []
    for g in gold:
        q = g["question"]
        gold_key = (g["doc_id"], int(g["page"]))
        kw_full = keyword_rank(q, con, limit=None)
        kw_order = [(d, p) for d, p, _s, _w in kw_full]
        row: dict = {"qid": g.get("qid", ""), "question": q, "gold": list(gold_key)}

        orders: dict[str, list[tuple[str, int]]] = {"keyword": kw_order}
        if cached:
            hit = store.get(q)
            if hit is not None:
                vis = visual_page_scores(index, pages, hit[0])
                vis_order = rank_order(vis)
                orders["visual"] = vis_order
                fused = rrf_fuse([kw_order, vis_order], k=60)
                orders["hybrid"] = [(d, p) for d, p, _s in fused]

        for mode, order in orders.items():
            rank = _rank_of(order, gold_key)
            row[mode] = {"rank": rank, "hit@1": _hit_at(order, gold_key, 1),
                         "hit@3": _hit_at(order, gold_key, 3),
                         "hit@5": _hit_at(order, gold_key, 5)}
        rows.append(row)
        ranks = {m: row[m]["rank"] for m in orders}
        top_pages = ", ".join(f"{d}/{p}" for d, p in orders.get("hybrid", kw_order)[:3])
        print(f"{row['qid']:>10} gold={gold_key[0]}/{gold_key[1]} "
              f"ranks={ranks} top3(hybrid)={top_pages}")

    print("\n=== summary ===")
    summary: dict = {}
    for mode in modes:
        ranks = [r[mode]["rank"] for r in rows if r.get(mode)]
        n = len(ranks)
        if not n:
            continue
        h1 = sum(1 for r in ranks if r == 1) / n
        h3 = sum(1 for r in ranks if r and r <= 3) / n
        h5 = sum(1 for r in ranks if r and r <= 5) / n
        mrr = sum(1.0 / r for r in ranks if r) / n
        summary[mode] = {"n": n, "hit@1": round(h1, 3), "hit@3": round(h3, 3),
                         "hit@5": round(h5, 3), "mrr": round(mrr, 3)}
        print(f"{mode:>8}: n={n} page_hit@1={h1:.3f} @3={h3:.3f} @5={h5:.3f} "
              f"MRR={mrr:.3f}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else REPO / "results" / "demo" / f"retrieval_validation_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, default=str),
        encoding="utf-8")
    print(f"\nWROTE {out / 'rows.json'}")


if __name__ == "__main__":
    main()
