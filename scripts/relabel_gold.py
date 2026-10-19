"""Run 8 Phase A: relabel gold.json with OCR-visibility + page annotations.

Adds to every gold item (data/gold.json):
  answer_in_ocr      -- the gold answer is derivable from gold-page OCR words
  answer_visual_only -- inverse: answer only exists inside the slide image
  bridged_question   -- multi-hop qualifier clause ("in the year X ...") or
                        category == "multi-hop"
  acceptable_pages   -- [{doc_id, page, box, sentence_ids}] for the gold and
                        every acceptable variant, gold page first (pages
                        searched within the gold document only).

The old file is backed up once to data/gold_pre_run8_backup.json.

Usage:
    python scripts/relabel_gold.py [--index data/index] [--gold data/gold.json]
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

from vvrag.eval.gold_regions import derive_answer_words, derive_gold_box  # noqa: E402
from vvrag.ingest.sentences import sentence_ids_for_answer  # noqa: E402
from vvrag.schemas import WordBox  # noqa: E402

MULTIHOP = re.compile(
    r"(in the year|the year (?:nestl|12%|79300)|the company (?:that|which)\b|"
    r"the country where|the brand which|which had won|which was awarded|"
    r"data of which)", re.I)


def _page_words(con: sqlite3.Connection, doc_id: str, page: int) -> list[WordBox]:
    rows = con.execute(
        "SELECT text,x0,y0,x1,y1,block_id,line_id FROM word_boxes "
        "WHERE doc_id=? AND page=?", (doc_id, page)).fetchall()
    return [WordBox(doc_id, page, r[0], (r[1], r[2], r[3], r[4]), r[5], r[6])
            for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=str(REPO / "data" / "index"))
    ap.add_argument("--gold", default=str(REPO / "data" / "gold.json"))
    args = ap.parse_args()
    index, gold_path = Path(args.index), Path(args.gold)

    backup = REPO / "data" / "gold_pre_run8_backup.json"
    if not backup.exists():
        backup.write_text(gold_path.read_text(encoding="utf-8"),
                          encoding="utf-8")
        print(f"backup written: {backup}")

    con = sqlite3.connect(str(index / "meta.db"))
    con.row_factory = sqlite3.Row
    gold = json.loads(gold_path.read_text(encoding="utf-8"))

    n_in = n_vis = n_bridge = 0
    for item in gold:
        doc_id, page = item["doc_id"], int(item["page"])
        words = _page_words(con, doc_id, page)
        found, score = derive_answer_words(item["answer"], words)
        item["answer_in_ocr"] = bool(found)
        item["answer_ocr_score"] = round(float(score), 4)
        item["answer_visual_only"] = not found
        item["bridged_question"] = bool(
            item.get("category") == "multi-hop" or MULTIHOP.search(item["question"]))
        n_in += 1 if found else 0
        n_vis += 0 if found else 1
        n_bridge += 1 if item["bridged_question"] else 0

        variants = [item["answer"], *[a for a in item.get("acceptable_answers", [])
                                      if a]]
        pages = []
        seen_pages = set()
        doc_pages = [r[0] for r in con.execute(
            "SELECT DISTINCT page FROM pages WHERE doc_id=? ORDER BY page",
            (doc_id,)).fetchall()]
        doc_pages = [page] + [p for p in doc_pages if p != page]
        for variant in variants:
            hit = False
            for p in doc_pages:
                if (p in seen_pages) and hit:
                    break
                ws = words if p == page else _page_words(con, doc_id, p)
                box = derive_gold_box(variant, ws)
                if box is None:
                    continue
                sents, _ = __import__("vvrag.ingest.sentences",
                                      fromlist=["sentences_from_page"]) \
                    .sentences_from_page(con, doc_id, p)
                sids = sentence_ids_for_answer(variant, sents)
                pages.append({"doc_id": doc_id, "page": p, "box": list(box),
                              "sentence_ids": sids})
                seen_pages.add(p)
                hit = True
                break
        item["acceptable_pages"] = pages
        print(f'{item["qid"][-8:]:>9}  in_ocr={item["answer_in_ocr"]!s:<5} '
              f'visual_only={item["answer_visual_only"]!s:<5} '
              f'bridge={item["bridged_question"]!s:<5} '
              f'pages={[(pp["page"], len(pp["sentence_ids"])) for pp in pages]}')

    con.close()
    gold_path.write_text(json.dumps(gold, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"\nWROTE {gold_path}")
    print(f"summary: answer_in_ocr {n_in}/{len(gold)}, visual_only {n_vis}, "
          f"bridged {n_bridge}")


if __name__ == "__main__":
    main()
