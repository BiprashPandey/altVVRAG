#!/usr/bin/env python
"""Materialize a SlideVQA subset for indexing (Priority 5 / Phase C).

Streams the official SlideVQA mirror, groups rows by deck, and writes:
  data/slidevqa/raw/<deck>.pdf      one multi-page PDF per deck (slide images)
  data/slidevqa/gold.json           QA items (question, answer variants, gold page)

Run on GPU once, then index with:
  python -m vvrag.ingest.run_ingest --input data/slidevqa/raw \
      --out data/slidevqa/index --embedder colqwen
No logic lives here beyond image -> PDF assembly + gold extraction.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pymupdf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

SLIDE_KEYS = [f"page_{i}" for i in range(1, 21)]


def _deck_pages(row) -> list:
    """Non-empty slide images for a deck row, in order."""
    pages = []
    for key in SLIDE_KEYS:
        img = row.get(key)
        if img is None:
            continue
        pages.append(img.convert("RGB") if hasattr(img, "convert") else img)
    return pages


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:120]


def _save_deck_pdf(pages: list, out_path: Path) -> None:
    doc = pymupdf.open()
    for img in pages:
        w, h = img.size
        page = doc.new_page(width=w, height=h)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        page.insert_image(page.rect, stream=buf.getvalue())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    doc.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize a SlideVQA subset")
    ap.add_argument("--out", default="data/slidevqa")
    ap.add_argument("--n-decks", type=int, default=5)
    ap.add_argument("--n-qa", type=int, default=25, help="max QA items to keep")
    ap.add_argument("--max-rows", type=int, default=60,
                    help="streaming scan cap (each row is a full deck of images)")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    from datasets import load_dataset  # local import: heavy

    out = Path(args.out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    repos = ("NTT-hil-insight/SlideVQA", "Ahren09/SlideVQA")
    ds = None
    for cand in repos:
        try:
            ds = load_dataset(cand, split=args.split, streaming=True)
            break
        except Exception as e:  # noqa: BLE001
            print(f"miss {cand}: {type(e).__name__}: {e}")
    if ds is None:
        raise SystemExit("SlideVQA unavailable on the Hub; cannot materialize.")

    golds: list[dict] = []
    decks_seen: dict[str, int] = {}
    n_decks = 0
    for i, row in enumerate(ds):
        if i >= args.max_rows or (n_decks >= args.n_decks and len(golds) >= args.n_qa):
            break
        deck = str(row.get("deck_name") or f"deck_{i}")
        # collect all QA for this deck row (a deck row holds one QA in the mirror)
        ans_raw = row.get("answer", [])
        variants = [str(a).strip() for a in (ans_raw if isinstance(ans_raw, list) else [ans_raw])
                    if str(a).strip()]
        ev = row.get("evidence_pages") or [1]
        ev_pages = [int(p) for p in ev] if isinstance(ev, (list, tuple)) else [int(ev)]
        gold_page = max(0, (ev_pages[0] if ev_pages else 1) - 1)

        if deck not in decks_seen:
            if n_decks >= args.n_decks:
                continue
            pages = _deck_pages(row)
            if not pages:
                continue
            _save_deck_pdf(pages, raw / f"{_safe(deck)}.pdf")
            decks_seen[deck] = len(pages)
            n_decks += 1
            print(f"deck {n_decks}/{args.n_decks}: {deck} -> {len(pages)} pages")

        if len(golds) < args.n_qa and str(row.get("question", "")).strip():
            golds.append({
                "qid": f"{deck}-{row.get('qa_id', i)}",
                "question": str(row["question"]).strip(),
                "answer": variants[0] if variants else "UNANSWERABLE",
                "acceptable_answers": variants[1:],
                "doc_id": _safe(deck),
                "page": gold_page,
            })

    (out / "gold.json").write_text(json.dumps(golds, indent=2), encoding="utf-8")
    print(f"wrote {len(golds)} QA items -> {out / 'gold.json'}")
    print(f"wrote {len(decks_seen)} decks -> {raw}")


if __name__ == "__main__":
    main()
