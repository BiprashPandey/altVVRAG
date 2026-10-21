"""Run 9 Phase 3 - gold audit (offline, guideline s4.8b).

Flags gold items whose answer text is not actually present on the labeled gold
page:

- presence: share of the answer's content tokens found in the page OCR words
  (order-insensitive, normalized);
- page_hit: whether the answer words appear on the labeled page at all;
- best_page: the page (any page of the same doc) with the highest token
  coverage, so "gold page is wrong" can be told apart from "answer is
  image-only / OCR noise";
- verdict: ok | page_mismatch | ocr_typo (>=50% tokens) | image_only (0%);
  also `answer_in_ocr` from the run 8 relabeling for cross-checking.

Writes results/gold_audit.json + prints the summary. No API, no GPU.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from vvrag.ingest.index import connect_sqlite  # noqa: E402
from vvrag.reader.claims import normalize_answer  # noqa: E402

_WORD = re.compile(r"[a-z0-9][a-z0-9%.,'-]*")


def content_tokens(text: str) -> list[str]:
    toks = []
    for t in _WORD.findall(normalize_answer(text)):
        t = t.strip(".,'")
        if not t:
            continue
        if len(t) == 1 and not t.isdigit():
            continue
        toks.append(t)
    return toks


def coverage(answer_tokens: list[str], page_text: str | None) -> float:
    if not answer_tokens:
        return 1.0
    if not page_text:
        return 0.0
    seen = set(_WORD.findall(normalize_answer(page_text)))
    hit = sum(1 for t in answer_tokens if t in seen)
    return hit / len(answer_tokens)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=str(REPO / "data" / "index"))
    ap.add_argument("--gold", default=str(REPO / "data" / "gold.json"))
    ap.add_argument("--out", default=str(REPO / "results" / "gold_audit.json"))
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    con = connect_sqlite(Path(args.index) / "meta.db")
    rows = []
    try:
        for g in gold:
            toks = content_tokens(g["answer"])
            doc_id, page = str(g["doc_id"]), int(g["page"])
            pages = {}
            for (p, txt) in con.execute(
                    "SELECT page, group_concat(text, ' ') FROM word_boxes "
                    "WHERE doc_id=? GROUP BY page", (doc_id,)):
                pages[int(p)] = txt
            cov = coverage(toks, pages.get(page))
            best_p, best_c = None, -1.0
            for p, txt in pages.items():
                c = coverage(toks, txt)
                if c > best_c:
                    best_p, best_c = p, c
            if cov >= 0.999:
                verdict = "ok"
            elif len(toks) == 1:
                # a single token (e.g. "3.6", "12") matches random pages; the
                # best-page signal is not trustworthy -> needs manual review
                verdict = "short_answer"
            elif best_c >= 0.999 and best_p != page:
                verdict = "page_mismatch"
            elif cov >= 0.5:
                verdict = "ocr_typo"
            elif best_c <= 0.0:
                verdict = "image_only"
            else:
                verdict = "partial"
            rows.append({
                "qid": g["qid"], "doc_id": doc_id, "page": page,
                "answer": g["answer"], "n_tokens": len(toks),
                "coverage_gold_page": round(cov, 3),
                "best_page": best_p, "coverage_best_page": round(best_c, 3),
                "verdict": verdict,
                "answer_in_ocr_label": g.get("answer_in_ocr"),
                "category": g.get("category"),
            })
    finally:
        con.close()

    from collections import Counter

    hist = dict(Counter(r["verdict"] for r in rows))
    mismatch = [r["qid"] for r in rows if r["verdict"] == "page_mismatch"]
    image_only = [r["qid"] for r in rows if r["verdict"] == "image_only"]
    label_disagree = [r["qid"] for r in rows
                      if r.get("answer_in_ocr_label") is not None
                      and (r["verdict"] == "ok") != bool(r["answer_in_ocr_label"])]
    out = {"n": len(rows), "verdict_hist": hist,
           "page_mismatch": mismatch, "image_only": image_only,
           "label_disagreements": label_disagree, "rows": rows}
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"gold audit: {len(rows)} items, verdicts={hist}")
    if mismatch:
        print(f"  PAGE MISMATCH (gold page wrong?): {mismatch}")
    if image_only:
        print(f"  IMAGE-ONLY (answer not in any OCR): {len(image_only)} items")
    if label_disagree:
        print(f"  label disagreements vs run-8 relabel: {label_disagree}")
    print(f"WROTE {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
