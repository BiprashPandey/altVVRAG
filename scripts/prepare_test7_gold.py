"""Phase 6: annotate data/gold.json with categories + build unanswerable probes.

Categories (per item, written back into data/gold.json):
  - "image-only": the gold answer is NOT derivable from gold-page OCR words
    (answer visible only inside the slide image / OCR typos).
  - "multi-hop":  question embeds a qualifier clause whose prerequisite fact
    sits elsewhere ("in the year X ...", "the company which ...").
  - "single-hop": everything else.

Unanswerable probes (data/gold_unanswerable.json): 10 questions whose answer
facts are absent from the corpus. Every probe carries `check_absent` tokens
(strings) that are verified ABSENT from the joined OCR text of ALL pages
before the file is written (fail -> probe dropped with a warning).

Usage:
    python scripts/prepare_test7_gold.py [--index data/index]
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vvrag.eval.gold_regions import derive_answer_words  # noqa: E402

MULTIHOP = re.compile(
    r"(in the year|the year (?:nestl|12%|79300)|the company (?:that|which)\b|"
    r"the country where|the brand which|which had won|which was awarded|"
    r"data of which)", re.I)

PROBES = [
    ("How many employees does Zephyrix Analytics have in Zurich?",
     ["zephyrix"]),
    ("Who was the CEO of Intel who replaced Paul Otellini in 2013?",
     ["otellini"]),
    ("What was the total revenue of Zynga in 2011?", ["zynga"]),
    ("How many 5G subscriptions were active worldwide in 2020?", ["5g"]),
    ("How much did Nestle pay to acquire Wyeth Nutrition?", ["wyeth"]),
    ("What was the closing stock price of Alcatel-Lucent on "
     "December 31, 2013?", ["closing price", "stock price"]),
    ("How much Chinese municipal-level government debt had been repaid "
     "as of June 2014?", ["june 2014"]),
    ("What is the second largest mobile operator by revenue mentioned in "
     "the Alcatel-Lucent references?", ["second largest"]),
    ("What was Siemens' organic growth in 2011 according to the Nestle "
     "roadshow?", ["siemens"]),
    ("Which company acquired Alcatel-Lucent in 2016?", ["nokia"]),
]


def _page_text(con: sqlite3.Connection) -> str:
    rows = con.execute("SELECT text FROM word_boxes ORDER BY doc_id, page").fetchall()
    return " ".join((r[0] or "") for r in rows).lower()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=str(REPO / "data" / "index"))
    ap.add_argument("--gold", default=str(REPO / "data" / "gold.json"))
    args = ap.parse_args()
    index = Path(args.index)
    gold_path = Path(args.gold)

    con = sqlite3.connect(str(index / "meta.db"))
    con.row_factory = sqlite3.Row
    gold = json.loads(gold_path.read_text(encoding="utf-8"))

    counts = {"single-hop": 0, "multi-hop": 0, "image-only": 0}
    for item in gold:
        rows = con.execute(
            "SELECT text,x0,y0,x1,y1,block_id,line_id FROM word_boxes "
            "WHERE doc_id=? AND page=?",
            (item["doc_id"], int(item["page"]))).fetchall()
        from vvrag.schemas import WordBox

        words = [WordBox(item["doc_id"], int(item["page"]), r["text"],
                         (r["x0"], r["y0"], r["x1"], r["y1"]),
                         r["block_id"], r["line_id"]) for r in rows]
        found, _ = derive_answer_words(item["answer"], words)
        if not found:
            cat = "image-only"
        elif MULTIHOP.search(item["question"]):
            cat = "multi-hop"
        else:
            cat = "single-hop"
        item["category"] = cat
        counts[cat] += 1
        print(f'{item["qid"][-6:]:>7}  {cat:<11} {item["answer"][:40]!r}')

    gold_path.write_text(json.dumps(gold, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"\nWROTE {gold_path} categories={counts}")

    all_text = _page_text(con)
    con.close()
    probes, dropped = [], []
    for i, (q, checks) in enumerate(PROBES):
        bad = [c for c in checks if c in all_text]
        if bad:
            dropped.append((q, bad))
            print(f"DROP (tokens present): {q!r} -> {bad}")
            continue
        probes.append({
            "qid": f"unans-{i:02d}",
            "question": q,
            "answer": "UNANSWERABLE",
            "acceptable_answers": [],
            "doc_id": None,
            "page": None,
            "category": "unanswerable",
            "check_absent": checks,
        })
    out = REPO / "data" / "gold_unanswerable.json"
    out.write_text(json.dumps(probes, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"WROTE {out} ({len(probes)} verified probes, {len(dropped)} dropped)")


if __name__ == "__main__":
    main()
