"""Run 8 Phase A: build the expanded unanswerable probe set (>=30 probes).

Categories (probe_category):
  absent-attribute  -- same subject, attribute for a year/value that the corpus
                       never mentions (verified: the year token is absent from
                       every page's OCR, words AND sentences).
  near-miss         -- question shadows a present fact (adjacent date/number):
                       a careless reader can copy the neighboring value.
  entity-swap       -- subject swapped to an entity that never appears.
  false-bridge      -- multi-hop question whose prerequisite is absent.

Every probe carries `check_absent` strings that are verified ABSENT from the
normalized joined OCR text of ALL pages before the file is written (failing
candidates are dropped with a warning). The Run 7 probe file is backed up to
data/gold_unanswerable_run7.json.

Usage:
    python scripts/prepare_probes.py [--index data/index] [--gold data/gold.json]
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

from vvrag.reader.claims import normalize_answer  # noqa: E402

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
# verified absent from every page's OCR (word boxes + sentences)
ABSENT_YEARS = [2019, 2020, 2021, 2022, 2023, 2025, 1999, 1998, 1997, 1995]

CATEGORY_OF_EXISTING = {
    "unans-00": "entity-swap",
    "unans-01": "false-bridge",
    "unans-02": "entity-swap",
    "unans-03": "entity-swap",
    "unans-04": "entity-swap",
    "unans-05": "absent-attribute",
    "unans-07": "false-bridge",
    "unans-08": "entity-swap",
    "unans-09": "entity-swap",
}

# (question, check strings, why) -- near-miss candidates (verified below)
NEAR_MISS = [
    ("Which are the bigger Debts to be repaid, Provincial-level government "
     "debts or Municipal-level government debts in June 2012?",
     ["june 2012"], "shadows the June 2013 debt table (q39); June 2012 absent"),
    ("Which country had the higher percentage of Tablet Pageviews in July "
     "2014, Russia or Hungary?",
     ["july 2014"], "shadows the June 2014 tablet table (q28)"),
    ("What is the percentage of Nestle's 20-year annual average of the real "
     "internal growth?",
     ["20 year", "20year"], "shadows the 10-year average (q8)"),
    ("What is the total percentage that wait Minutes and Hours for an average "
     "BI request in 2020?",
     ["2020"], "shadows the BI on-demand waiting poll (q12)"),
    ("Which Activity is undertaken by more laptop users, Email services or "
     "Social networks?",
     ["laptop"], "shadows the smartphone activity table (q24)"),
    ("How many points are listed for TAM Notes in 2015?",
     ["2015"], "shadow: 2015 exists only in the debts deck, not TAM Notes"),
    ("Which OS was used more among mobile Facebook users in November 2014, "
     "Android or Windows Mobile?",
     ["november 2014"], "shadows the October 2014 OS table (q30)"),
    ("Which country spend the most in Mobile display ads in 2019, according "
     "to IAB Europe AdEx Benchmark?",
     ["2019"], "shadows the 2013 ad-spend chart (q25/26)"),
    ("Which region of the world accounts for the highest percentage of 2019 "
     "revenues?",
     ["2019"], "shadows the 2010 revenue mix (q19)"),
]

ENTITY_SWAP = [
    ("What was the market share of Xiaomi in the global smartphone market?",
     ["xiaomi"], "Xiaomi never appears"),
    ("How many employees does Samsung Electronics have in Seoul?",
     ["samsung"], "Samsung never appears"),
    ("What was the revenue of Tata Consultancy Services in 2011?",
     ["tata"], "Tata never appears"),
    ("How many users does WeChat have in Europe?",
     ["wechat"], "WeChat never appears"),
    ("What is the market cap of Coca-Cola according to the Alcatel-Lucent "
     "overview?",
     ["coca-cola", "coca cola"], "Coca-Cola never appears"),
    ("How many employees does Microsoft have worldwide?",
     ["microsoft"], "Microsoft never appears"),
    ("What was Sony's revenue in 2011?",
     ["sony"], "Sony never appears"),
    ("How many members does LinkedIn have?",
     ["linkedin"], "LinkedIn never appears"),
    ("What is the estimated cost of the Qatar 2022 World Cup?",
     ["qatar"], "Qatar never appears"),
]

FALSE_BRIDGE = [
    ("In the year Nestle was founded, what was its organic growth?",
     ["was founded"], "founding year absent; qualifier cannot resolve"),
    ("What is the name of the brand which Nestle acquired in the year the "
     "Berlin Wall fell?",
     ["berlin wall"], "event absent from every page"),
    ("In the city where Google has its headquarters, how many employees does "
     "Nestle have?",
     ["google"], "Google never appears"),
    ("What was the revenue of the company that acquired WhatsApp?",
     ["whatsapp"], "acquirer absent"),
    ("Who was the CEO of Nestle when the iPhone was released?",
     ["iphone"], "temporal anchor absent"),
    ("What does the Alcatel-Lucent deck say will happen after Brexit?",
     ["brexit"], "event absent"),
    ("Which of these documents mentions the Amazon rainforest?",
     ["rainforest"], "topic absent"),
    ("What was the Alcatel-Lucent stock price on the day Twitter went "
     "public?",
     ["twitter"], "anchor event absent"),
]

ABSENT_ATTRIBUTE = [
    ("How many kinds of BICC Key Skills are there in 2020?",
     ["2020"], "attribute question scoped to an absent year"),
    ("What is the percentage of Nestle's 10-year annual average of the real "
     "internal growth as reported in 2021?",
     ["2021"], "shadowed by the present 10-year average (q8)"),
    ("Which are the bigger Debts to be repaid as of March 2019, "
     "Provincial-level government debts or Municipal-level government debts?",
     ["march 2019"], "debt table shadow with absent date"),
]


def _all_text(con: sqlite3.Connection) -> str:
    words = con.execute(
        "SELECT text FROM word_boxes ORDER BY doc_id, page").fetchall()
    try:
        sents = con.execute("SELECT text FROM sentences ORDER BY doc_id, page").fetchall()
    except sqlite3.OperationalError:
        sents = []
    text = " ".join((r[0] or "") for r in words + sents)
    return normalize_answer(text)


def _absent(all_norm: str, checks: list[str]) -> list[str]:
    return [c for c in checks if normalize_answer(c) in all_norm]


def _year_swaps(gold: list[dict]) -> list[tuple[str, list[str], str]]:
    """Single-hop questions with years -> same question, absent years.

    Every year occurrence is replaced by a DISTINCT absent year so
    "2003 or 2004?" does not become "2021 or 2021?".
    """
    out = []
    used_years: set[int] = set()
    for item in gold:
        if item.get("category") != "single-hop":
            continue
        q = item["question"]
        if not YEAR_RE.search(q):
            continue
        picks = [y for y in ABSENT_YEARS
                 if y not in used_years and str(y) not in q]
        if not picks:
            continue
        it = iter(picks)
        new_q = YEAR_RE.sub(lambda _m: str(next(it)), q)
        n_repl = len(YEAR_RE.findall(q))
        chosen = picks[:n_repl]
        used_years.update(chosen)
        out.append((new_q, [str(y) for y in chosen],
                    f"qid {item['qid'][-6:]}: years swapped to absent {chosen}"))
        if len(out) >= 10:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=str(REPO / "data" / "index"))
    ap.add_argument("--gold", default=str(REPO / "data" / "gold.json"))
    args = ap.parse_args()

    probe_path = REPO / "data" / "gold_unanswerable.json"
    backup = REPO / "data" / "gold_unanswerable_run7.json"
    if probe_path.exists() and not backup.exists():
        backup.write_text(probe_path.read_text(encoding="utf-8"),
                          encoding="utf-8")
        print(f"backup written: {backup}")

    con = sqlite3.connect(str(Path(args.index) / "meta.db"))
    con.row_factory = sqlite3.Row
    all_norm = _all_text(con)
    con.close()
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))

    existing = []
    if backup.exists():
        for p in json.loads(backup.read_text(encoding="utf-8")):
            existing.append((p["question"], p["check_absent"],
                             CATEGORY_OF_EXISTING.get(p["qid"],
                                                      "false-bridge")))

    candidates: list[tuple[str, str, list[str], str]] = []
    for q, checks, why in NEAR_MISS:
        candidates.append(("near-miss", q, checks, why))
    for q, checks, why in ABSENT_ATTRIBUTE:
        candidates.append(("absent-attribute", q, checks, why))
    for q, checks, why in ENTITY_SWAP:
        candidates.append(("entity-swap", q, checks, why))
    for q, checks, why in FALSE_BRIDGE:
        candidates.append(("false-bridge", q, checks, why))
    for q, checks, why in _year_swaps(gold):
        candidates.append(("absent-attribute", q, checks, why))

    probes: list[dict] = []
    dropped: list[tuple[str, list[str]]] = []

    def _add(cat: str, q: str, checks: list[str], why: str) -> None:
        bad = _absent(all_norm, checks)
        if bad:
            dropped.append((q, bad))
            print(f"DROP [{cat}] (tokens present): {q[:70]!r} -> {bad}")
            return
        probes.append({
            "qid": "",  # assigned after ordering
            "question": q,
            "answer": "UNANSWERABLE",
            "acceptable_answers": [],
            "doc_id": None,
            "page": None,
            "category": "unanswerable",
            "probe_category": cat,
            "check_absent": checks,
            "why": why,
        })

    for q, checks, cat in existing:
        _add(cat, q, checks, "Run 7 verified probe (re-categorized)")
    for cat, q, checks, why in candidates:
        _add(cat, q, checks, why)

    order = {"absent-attribute": 0, "near-miss": 1, "entity-swap": 2,
             "false-bridge": 3}
    probes.sort(key=lambda p: order[p["probe_category"]])
    for i, p in enumerate(probes):
        p["qid"] = f"unans-{i:02d}"

    from collections import Counter
    counts = Counter(p["probe_category"] for p in probes)
    probe_path.write_text(json.dumps(probes, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    print(f"\nWROTE {probe_path} ({len(probes)} probes, {len(dropped)} dropped)")
    print(f"by category: {dict(counts)}")
    if len(probes) < 30:
        raise SystemExit(f"need >=30 probes, got {len(probes)}")


if __name__ == "__main__":
    main()
