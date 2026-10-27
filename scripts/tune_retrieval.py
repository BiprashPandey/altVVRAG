"""Phase 0 retrieval sweep: keyword vs visual vs weighted RRF, with/without
the lexical reranker.

This is an offline, deterministic page-retrieval evaluator. It never calls a
reader, an examiner, or the network. Query vectors must already be cached
(`scripts/precompute_query_vectors.py` on a GPU box).

Examples
--------
    python scripts/tune_retrieval.py --index data/index --gold data/gold.json \
        --query-cache data/query_cache --allow-partial

    python scripts/tune_retrieval.py --index data/index --gold data/gold.json \
        --query-cache data/query_cache \
        --weights 1:1,1:2,1:3,1:5 --rrf-k 30,60,100 \
        --candidate-k 5,10,20 --rerank both

Outputs `rows.json` (meta + summary + per-question rows) and `summary.csv`
under `results/demo/retrieval_sweep_<timestamp>/` by default.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vvrag.eval.metrics import clopper_pearson_interval  # noqa: E402
from vvrag.ingest.index import connect_sqlite  # noqa: E402
from vvrag.retrieve.fusion import (rank_order, rrf_fuse,  # noqa: E402
                                   visual_page_scores)
from vvrag.retrieve.query_vectors import QueryVectorStore  # noqa: E402
from vvrag.retrieve.rerank import lexical_rerank  # noqa: E402
from vvrag.ui.qa import keyword_rank, page_words  # noqa: E402


def _load_items(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("items", [])
    return list(data)


def _acceptable(item: dict) -> list[tuple[str, int]]:
    got: list[tuple[str, int]] = []
    for ap in item.get("acceptable_pages") or []:
        if isinstance(ap, dict) and ap.get("doc_id") is not None:
            got.append((str(ap["doc_id"]), int(ap["page"])))
    if not got and item.get("doc_id") is not None:
        got.append((str(item["doc_id"]), int(item["page"])))
    seen, out = set(), []
    for key in got:
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _rank_of(order: list[tuple[str, int]],
             acceptable: list[tuple[str, int]]) -> int | None:
    best = None
    for key in acceptable:
        try:
            r = order.index(key) + 1
        except ValueError:
            continue
        if best is None or r < best:
            best = r
    return best


def _metrics(rows: list[dict], mode: str) -> dict:
    ranks = [r[mode]["rank"] for r in rows if r.get(mode)]
    n = len(ranks)
    if not n:
        return {"n": 0}
    h1 = sum(1 for r in ranks if r and r <= 1)
    h3 = sum(1 for r in ranks if r and r <= 3)
    h5 = sum(1 for r in ranks if r and r <= 5)
    found = [r for r in ranks if r]
    mrr = sum(1.0 / r for r in found) / n
    mrr_found = sum(1.0 / r for r in found) / len(found) if found else 0.0
    lo, hi = clopper_pearson_interval(h1, n)
    return {
        "n": n,
        "hit@1": round(h1 / n, 4),
        "hit@1_cp95_low": round(lo, 4),
        "hit@1_cp95_high": round(hi, 4),
        "hit@3": round(h3 / n, 4),
        "hit@5": round(h5 / n, 4),
        "mrr": round(mrr, 4),
        "mrr_found": round(mrr_found, 4),
        "found": len(found),
    }


def _parse_weights(raw: str) -> list[tuple[float, float]]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"weight must be keyword:visual, got {part!r}")
        k, v = part.split(":", 1)
        out.append((float(k), float(v)))
    return out or [(1.0, 1.0)]


def _parse_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _candidate_list(order: list[tuple[str, int]], candidate_k: int,
                    words_map: dict) -> list[tuple[str, int, float, list]]:
    out = []
    for i, (d, p) in enumerate(order[:candidate_k]):
        out.append((d, p, 1.0 / (i + 1), words_map.get((d, p)) or []))
    return out


def _rerank(order: list[tuple[str, int]], question: str, candidate_k: int,
            words_map: dict) -> list[tuple[str, int]]:
    cand = _candidate_list(order, candidate_k, words_map)
    if len(cand) <= 1:
        return order
    ranked = lexical_rerank(question, cand, k=candidate_k)
    return [(d, p) for d, p, _s, _w in ranked] + [
        key for key in order[candidate_k:] if key not in {(d, p) for d, p, _s, _w in ranked}
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--gold", default="data/gold.json")
    ap.add_argument("--query-cache", default="data/query_cache")
    ap.add_argument("--out", default="")
    ap.add_argument("--modes", default="keyword,visual,hybrid",
                    help="comma list: keyword,visual,hybrid")
    ap.add_argument("--weights", default="1:1,1:2,1:3,1:5",
                    help="hybrid keyword:visual weights, comma list")
    ap.add_argument("--rrf-k", default="30,60,100")
    ap.add_argument("--candidate-k", default="5,10,20")
    ap.add_argument("--rerank", choices=("off", "lexical", "both"), default="both")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N items")
    ap.add_argument("--allow-partial", action="store_true",
                    help="evaluate only items with cached query vectors")
    args = ap.parse_args()

    index = Path(args.index)
    items = _load_items(args.gold)
    if args.limit:
        items = items[: args.limit]
    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    modes = [m for m in modes if m in {"keyword", "visual", "hybrid"}]
    if not modes:
        raise SystemExit("no valid modes")
    weights = _parse_weights(args.weights)
    rrf_ks = _parse_ints(args.rrf_k)
    candidate_ks = _parse_ints(args.candidate_k)
    reranks = ["off", "lexical"] if args.rerank == "both" else [args.rerank]

    con = connect_sqlite(index / "meta.db")
    pages = [(str(r[0]), int(r[1])) for r in
             con.execute("SELECT doc_id, page FROM pages ORDER BY doc_id, page")]
    pages_set = set(pages)
    store = QueryVectorStore(args.query_cache)
    cached = {it["qid"]: store.get(it["question"]) is not None for it in items}
    missing = [it["qid"] for it in items if not cached[it["qid"]]]
    if missing and not args.allow_partial:
        print(f"ERROR: {len(missing)}/{len(items)} questions lack query vectors in "
              f"{args.query_cache}.", file=sys.stderr)
        print("Run scripts/precompute_query_vectors.py on a GPU box, or pass "
              "--allow-partial for a smaller diagnostic subset.", file=sys.stderr)
        raise SystemExit(2)
    if missing:
        keep = {it["qid"] for it in items if cached[it["qid"]]}
        items = [it for it in items if it["qid"] in keep]
        print(f"WARNING: partial cache; evaluating {len(items)} questions "
              f"({len(missing)} missing).")
    if not items:
        raise SystemExit("no questions with cached query vectors")

    # Build all source rankings once.
    base: dict[str, dict] = {}
    words_map: dict[tuple[str, int], list] = {}
    for i, it in enumerate(items, 1):
        q = it["question"]
        kw_full = keyword_rank(q, con, limit=None)
        kw = [(d, p) for d, p, _s, _w in kw_full]
        for d, p, _s, w in kw_full:
            words_map[(d, p)] = w
        vis = None
        hit = store.get(q)
        if hit is not None:
            vis_scores = visual_page_scores(index, pages, hit[0])
            vis = rank_order(vis_scores)
        # Ensure visual-only pages have words available for the reranker.
        if vis:
            for d, p in vis[:max(candidate_ks or [1])]:
                if (d, p) in pages_set and (d, p) not in words_map:
                    words_map[(d, p)] = page_words(con, d, p)
        base[it["qid"]] = {"item": it, "kw": kw, "vis": vis}
        print(f"[{i}/{len(items)}] {it['qid']}: keyword={len(kw)} "
              f"visual={'yes' if vis else 'no'}")

    configs: list[dict] = []
    if "keyword" in modes:
        for rk in reranks:
            configs.append({"mode": "keyword", "weights": None, "rrf_k": None,
                            "candidate_k": None, "rerank": rk})
    if "visual" in modes:
        for rk in reranks:
            # candidate_k only changes the lexical-reranker input; when the
            # reranker is off the full visual order is already used.
            for ck in (candidate_ks if rk != "off" else candidate_ks[:1]):
                configs.append({"mode": "visual", "weights": None, "rrf_k": None,
                                "candidate_k": ck, "rerank": rk})
    if "hybrid" in modes:
        for w in weights:
            for rk in rrf_ks:
                for rr in reranks:
                    for ck in (candidate_ks if rr != "off" else candidate_ks[:1]):
                        configs.append({"mode": "hybrid", "weights": w,
                                        "rrf_k": rk, "candidate_k": ck,
                                        "rerank": rr})

    rows: list[dict] = []
    for cfg in configs:
        label = _label(cfg)
        for it in items:
            b = base[it["qid"]]
            order = list(b["kw"])
            if cfg["mode"] == "visual":
                if b["vis"] is None:
                    continue
                order = list(b["vis"])
            elif cfg["mode"] == "hybrid":
                if b["vis"] is None:
                    continue
                fused = rrf_fuse([b["kw"], b["vis"]], k=int(cfg["rrf_k"]),
                                 weights=list(cfg["weights"]))
                order = [(d, p) for d, p, _s in fused]
            if cfg["rerank"] == "lexical":
                order = _rerank(order, it["question"], int(cfg["candidate_k"] or 5),
                                words_map)
            rank = _rank_of(order, _acceptable(it))
            rows.append({
                "config": label, "mode": cfg["mode"],
                "keyword_weight": cfg["weights"][0] if cfg["weights"] else None,
                "visual_weight": cfg["weights"][1] if cfg["weights"] else None,
                "rrf_k": cfg["rrf_k"], "candidate_k": cfg["candidate_k"],
                "rerank": cfg["rerank"], "qid": it["qid"],
                "question": it["question"], "rank": rank,
                "top10": [f"{d}/{p}" for d, p in order[:10]],
            })
        print(f"evaluated {label}: {sum(1 for r in rows if r['config'] == label)} rows")

    summary = []
    for cfg in configs:
        label = _label(cfg)
        sub = [r for r in rows if r["config"] == label]
        m = _metrics([{label: r} for r in sub], label)
        summary.append({**cfg, "label": label, **m})
    summary.sort(key=lambda s: (-float(s.get("hit@1", 0)),
                                -float(s.get("mrr", 0)),
                                -float(s.get("hit@3", 0))))

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else (
        REPO / "results" / "demo" / f"retrieval_sweep_{stamp}")
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "index": str(index),
        "gold": str(args.gold),
        "query_cache": str(args.query_cache),
        "n_questions": len(items),
        "missing_query_vectors": len(missing),
        "modes": modes,
        "weights": weights,
        "rrf_k": rrf_ks,
        "candidate_k": candidate_ks,
        "rerank": reranks,
        "pages": len(pages),
    }
    (out / "rows.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "rows": rows}, indent=2),
        encoding="utf-8")
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["label", "mode", "keyword_weight", "visual_weight", "rrf_k",
                  "candidate_k", "rerank", "n", "hit@1", "hit@3", "hit@5",
                  "mrr", "mrr_found"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)
    print("\n=== top configs ===")
    for s in summary[:15]:
        print(f"{s['label']:<55} hit@1={s.get('hit@1', 0):.3f} "
              f"hit@3={s.get('hit@3', 0):.3f} mrr={s.get('mrr', 0):.3f}")
    print(f"\nWROTE {out}")


def _label(cfg: dict) -> str:
    if cfg["mode"] == "keyword":
        return f"keyword|rerank={cfg['rerank']}"
    if cfg["mode"] == "visual":
        return f"visual|k={cfg['candidate_k']}|rerank={cfg['rerank']}"
    kw, vis = cfg["weights"]
    return (f"hybrid|w={kw:g}:{vis:g}|rrf={cfg['rrf_k']}|"
            f"k={cfg['candidate_k']}|rerank={cfg['rerank']}")


if __name__ == "__main__":
    main()
