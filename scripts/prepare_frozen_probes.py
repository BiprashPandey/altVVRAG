"""Build verified-absent probes for a frozen SlideVQA set.

Unlike the dev probe file, this script is parameterised by --index/--gold/--out
and refuses to emit a probe whose `check_absent` strings occur anywhere in the
combined OCR of the frozen corpus. Probe categories follow the Run 8/9
protocol: absent-attribute, near-miss, entity-swap, false-bridge.

Usage:
  python scripts/prepare_frozen_probes.py --index data/index_frozen3_local \
      --gold data/slidevqa_frozen3/gold.json \
      --out data/slidevqa_frozen3/probes.json --n 60
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vvrag.reader.claims import normalize_answer  # noqa: E402

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
ABSENT_YEARS = [2025, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 1999, 1998]

NEAR_MISS = [
    ("Which are the bigger Debts to be repaid, Provincial-level government "
     "debts or Municipal-level government debts in June 2012?",
     ["june 2012"], "shadows a June 2013 debt table"),
    ("Which country had the higher percentage of Tablet Pageviews in July "
     "2014, Russia or Hungary?",
     ["july 2014"], "shadows a June 2014 tablet table"),
    ("What is the percentage of the company's 20-year annual average real "
     "internal growth?", ["20 year", "20year"], "shadows a 10-year average"),
    ("What is the total percentage that wait Minutes and Hours for an average "
     "BI request in 2020?", ["2020"], "shadows an on-demand waiting poll"),
    ("Which Activity is undertaken by more laptop users, Email services or "
     "Social networks?", ["laptop"], "shadows a smartphone activity table"),
    ("How many points are listed for TAM Notes in 2015?", ["2015"],
     "2015 is absent from the TAM Notes material"),
    ("Which OS was used more among mobile Facebook users in November 2014, "
     "Android or Windows Mobile?", ["november 2014"],
     "shadows an October 2014 OS table"),
    ("Which country spends the most in mobile display ads in 2019, according "
     "to IAB Europe AdEx Benchmark?", ["2019"], "shadows a 2013 ad-spend chart"),
    ("Which region of the world accounts for the highest percentage of 2019 "
     "revenues?", ["2019"], "shadows a 2010 revenue mix"),
    ("What was the global smartphone penetration rate in March 2024?",
     ["march 2024"], "future month outside the corpus"),
    ("How many million users did the service have in the fourth quarter of "
     "2023?", ["2023"], "future year outside the corpus"),
    ("Which category grew fastest between 2018 and 2019?",
     ["2018", "2019"], "both years absent from this deck"),
    ("What was the revenue growth in March 2024?", ["march 2024"],
     "future month outside the corpus"),
    ("How many customers were added in August 2021?", ["august 2021"],
     "month outside the corpus"),
    ("Which segment performed best in July 2019?", ["july 2019"],
     "month outside the corpus"),
    ("What was the margin in December 2025?", ["december 2025"],
     "future month outside the corpus"),
    ("How did the company perform in November 2022?", ["november 2022"],
     "month outside the corpus"),
    ("What was the retention rate in June 2020?", ["june 2020"],
     "month outside the corpus"),
    ("Which product led the market in September 1998?", ["september 1998"],
     "month outside the corpus"),
    ("What was the annual result in October 1999?", ["october 1999"],
     "month outside the corpus"),
    ("How many users were active in January 2024?", ["january 2024"],
     "future month outside the corpus"),
    ("What changed in February 2018?", ["february 2018"],
     "month outside the corpus"),
]

ENTITY_SWAP = [
    ("What was the market share of Xiaomi in the global smartphone market?",
     ["xiaomi"], "Xiaomi never appears"),
    ("How many employees does Samsung Electronics have in Seoul?",
     ["samsung"], "Samsung never appears"),
    ("What was the revenue of Tata Consultancy Services in 2011?",
     ["tata"], "Tata never appears"),
    ("How many users does WeChat have in Europe?", ["wechat"],
     "WeChat never appears"),
    ("What is the market cap of Coca-Cola according to this deck?",
     ["coca-cola", "coca cola"], "Coca-Cola never appears"),
    ("How many employees does Microsoft have worldwide?", ["microsoft"],
     "Microsoft never appears"),
    ("What was Sony's revenue in 2011?", ["sony"], "Sony never appears"),
    ("How many members does LinkedIn have?", ["linkedin"],
     "LinkedIn never appears"),
    ("What is the estimated cost of the Qatar 2022 World Cup?", ["qatar"],
     "Qatar never appears"),
    ("What did the report say about the Tesla Gigafactory?", ["tesla"],
     "Tesla never appears"),
    ("How many offices does Airbnb have in Europe?", ["airbnb"],
     "Airbnb never appears"),
    ("What was the net profit of Netflix in 2015?", ["netflix"],
     "Netflix never appears"),
    ("How many products did Apple launch in 2014?", ["apple"],
     "Apple never appears"),
    ("What is the market share of Huawei in Germany?", ["huawei"],
     "Huawei never appears"),
    ("How many subscribers does Spotify have in Asia?", ["spotify"],
     "Spotify never appears"),
    ("What was the GDP of Brazil in 2012?", ["brazil"],
     "Brazil never appears"),
    ("What did Amazon report for its cloud business?", ["amazon"],
     "Amazon never appears"),
    ("How many employees does Boeing have?", ["boeing"],
     "Boeing never appears"),
    ("What was Nike's annual revenue?", ["nike"],
     "Nike never appears"),
    ("How many stores does IKEA operate?", ["ikea"],
     "IKEA never appears"),
    ("What was Verizon's churn rate?", ["verizon"],
     "Verizon never appears"),
    ("How many mainframes did IBM sell?", ["ibm"],
     "IBM never appears"),
    ("What was Oracle's cloud revenue?", ["oracle"],
     "Oracle never appears"),
    ("How many customers does Salesforce have?", ["salesforce"],
     "Salesforce never appears"),
    ("What was Zoom's daily active user count?", ["zoom"],
     "Zoom never appears"),
    ("How many users does TikTok have in the US?", ["tiktok"],
     "TikTok never appears"),
    ("What was Alibaba's GMV in 2015?", ["alibaba"],
     "Alibaba never appears"),
    ("What was Tencent's operating profit?", ["tencent"],
     "Tencent never appears"),
    ("How many searches does Baidu process?", ["baidu"],
     "Baidu never appears"),
    ("What did Siemens say about its industrial business?", ["siemens"],
     "Siemens never appears"),
    ("What was Nestle's organic growth in 2011?", ["nestle"],
     "Nestle never appears"),
    ("How many brands does Unilever own?", ["unilever"],
     "Unilever never appears"),
    ("What was Pfizer's vaccine revenue?", ["pfizer"],
     "Pfizer never appears"),
    ("How many doses did Moderna produce?", ["moderna"],
     "Moderna never appears"),
    ("What did Exxon report for upstream earnings?", ["exxon"],
     "Exxon never appears"),
    ("How many barrels did Shell produce?", ["shell"],
     "Shell never appears"),
    ("What was Chevron's refining margin?", ["chevron"],
     "Chevron never appears"),
    ("How many cars did Mercedes sell?", ["mercedes"],
     "Mercedes never appears"),
    ("How many stores does Starbucks operate?", ["starbucks"],
     "Starbucks never appears"),
    ("What was McDonald's same-store sales?", ["mcdonalds"],
     "McDonald's never appears"),
    ("What was Coca Cola's volume growth?", ["coca cola"],
     "Coca-Cola never appears"),
]

FALSE_BRIDGE = [
    ("In the year the company was founded, what was its organic growth?",
     ["was founded"], "founding year qualifier absent"),
    ("What is the name of the brand acquired in the year the Berlin Wall "
     "fell?", ["berlin wall"], "event absent from every page"),
    ("In the city where Google has its headquarters, how many employees are "
     "there?", ["google"], "Google never appears"),
    ("What was the revenue of the company that acquired WhatsApp?",
     ["whatsapp"], "acquirer absent"),
    ("Who was the CEO when the iPhone was released?", ["iphone"],
     "temporal anchor absent"),
    ("What does the deck say will happen after Brexit?", ["brexit"],
     "event absent"),
    ("Which of these documents mentions the Amazon rainforest?",
     ["rainforest"], "topic absent"),
    ("What was the stock price on the day Twitter went public?",
     ["twitter"], "anchor event absent"),
    ("Which product replaced the one announced at CES 2016?", ["ces 2016"],
     "anchor event absent"),
    ("What did the CEO say after the merger with Uber?", ["uber"],
     "event/entity absent"),
    ("How did the market react to the 2024 election?", ["2024"],
     "year/event absent"),
    ("What did the report say about the Paris Agreement?", ["paris agreement"],
     "event absent"),
    ("How many medals were won at the 2024 Olympics?", ["2024 olympics"],
     "event absent"),
    ("What was the impact of Hurricane Katrina?", ["hurricane katrina"],
     "event absent"),
    ("How did COVID-19 change the business?", ["covid-19"],
     "event absent"),
    ("What did the company say about ChatGPT?", ["chatgpt"],
     "event/entity absent"),
    ("How did the company use blockchain?", ["blockchain"],
     "topic absent"),
    ("What was the Bitcoin price in 2017?", ["bitcoin"],
     "topic absent"),
    ("What did the CEO say about the metaverse?", ["metaverse"],
     "topic absent"),
    ("How many rockets did SpaceX launch?", ["spacex"],
     "entity absent"),
    ("What did the report say about the moon landing?", ["moon landing"],
     "event absent"),
    ("How did the Cold War affect the market?", ["cold war"],
     "event absent"),
    ("What was the French Revolution's economic impact?", ["french revolution"],
     "event absent"),
    ("How did the Industrial Revolution change the industry?", ["industrial revolution"],
     "event absent"),
    ("What did the report say about artificial intelligence?", ["artificial intelligence"],
     "topic absent"),
    ("How did machine learning change the forecast?", ["machine learning"],
     "topic absent"),
    ("What did the company say about climate change?", ["climate change"],
     "topic absent"),
    ("What was the role of the internet of things?", ["internet of things"],
     "topic absent"),
]

ABSENT_ATTRIBUTE = [
    ("How many kinds of BICC Key Skills are there in 2020?", ["2020"],
     "attribute scoped to an absent year"),
    ("What is the company's 10-year annual average real internal growth as "
     "reported in 2021?", ["2021"], "shadowed by a present 10-year average"),
    ("Which are the bigger Debts to be repaid as of March 2019, "
     "Provincial-level government debts or Municipal-level government debts?",
     ["march 2019"], "debt table shadow with absent date"),
    ("What percentage of the market did the company hold in 2025?",
     ["2025"], "future year outside the corpus"),
    ("How many employees were added in the 2022 financial year?", ["2022"],
     "future year outside the corpus"),
    ("What was the revenue growth in the first quarter of 2021?", ["2021"],
     "year outside the corpus"),
    ("How many stores did the company operate in 2019?", ["2019"],
     "year outside the corpus"),
    ("What was the customer satisfaction score in 2020?", ["2020"],
     "year outside the corpus"),
    ("Which product had the highest margin in 2018?", ["2018"],
     "year outside the corpus"),
    ("What was the advertising spend in March 2024?", ["march 2024"],
     "future date outside the corpus"),
    ("How many patents were filed in 1998?", ["1998"],
     "year outside the corpus"),
    ("What was the annual return in 1999?", ["1999"],
     "year outside the corpus"),
    ("What was the revenue in 1991?", ["1991"], "year outside the corpus"),
    ("How many employees were there in 1992?", ["1992"],
     "year outside the corpus"),
    ("What was the margin in 1996?", ["1996"], "year outside the corpus"),
    ("How many stores existed in 1997?", ["1997"], "year outside the corpus"),
    ("What was the growth rate in 1998?", ["1998"], "year outside the corpus"),
    ("What is the forecast for 2026?", ["2026"], "future year outside the corpus"),
    ("How many users are expected in 2027?", ["2027"], "future year outside the corpus"),
    ("What is the plan for 2028?", ["2028"], "future year outside the corpus"),
    ("What is the target for 2029?", ["2029"], "future year outside the corpus"),
    ("What was the revenue in March 2024?", ["march 2024"],
     "future month outside the corpus"),
    ("How many customers were added in August 2021?", ["august 2021"],
     "month outside the corpus"),
    ("Which segment performed best in July 2019?", ["july 2019"],
     "month outside the corpus"),
    ("What was the margin in December 2025?", ["december 2025"],
     "future month outside the corpus"),
    ("How did the company perform in November 2022?", ["november 2022"],
     "month outside the corpus"),
    ("What was the retention rate in June 2020?", ["june 2020"],
     "month outside the corpus"),
]


def _all_norm(con: sqlite3.Connection) -> str:
    rows = con.execute("SELECT text FROM word_boxes").fetchall()
    try:
        rows += con.execute("SELECT text FROM sentences").fetchall()
    except sqlite3.OperationalError:
        pass
    return normalize_answer(" ".join((r[0] or "") for r in rows))


def _present(all_norm: str, checks: list[str]) -> list[str]:
    return [c for c in checks if normalize_answer(c) in all_norm]


def _year_swaps(gold: list[dict]) -> list[tuple[str, list[str], str]]:
    out, used = [], set()
    for item in gold:
        q = item["question"]
        years = YEAR_RE.findall(q)
        if not years:
            continue
        picks = [y for y in ABSENT_YEARS if str(y) not in q and y not in used]
        if len(picks) < len(years):
            continue
        it = iter(picks)
        new_q = YEAR_RE.sub(lambda _m: str(next(it)), q)
        chosen = picks[:len(years)]
        used.update(chosen)
        out.append((new_q, [str(y) for y in chosen],
                    f"year-swapped from {item.get('qid')} to absent {chosen}"))
        if len(out) >= 30:
            break
    return out


def _year_added(gold: list[dict], start: int = 0) -> list[tuple[str, list[str], str]]:
    out = []
    for item in gold[start:]:
        q = item["question"]
        if YEAR_RE.search(q):
            continue
        year = next((y for y in ABSENT_YEARS if str(y) not in q), None)
        if year is None:
            continue
        out.append((q.rstrip("?") + f" as reported in {year}?",
                    [str(year)],
                    f"absent-year qualifier added to {item.get('qid')}"))
        if len(out) >= 24:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    con = sqlite3.connect(str(Path(args.index) / "meta.db"))
    all_norm = _all_norm(con)
    con.close()
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))

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
    for q, checks, why in _year_added(gold):
        candidates.append(("absent-attribute", q, checks, why))

    probes, dropped, seen = [], [], set()
    for cat, q, checks, why in candidates:
        nq = normalize_answer(q)
        if nq in seen:
            continue
        bad = _present(all_norm, checks)
        if bad:
            dropped.append((cat, q, bad))
            continue
        seen.add(nq)
        probes.append({
            "qid": "", "question": q, "answer": "UNANSWERABLE",
            "acceptable_answers": [], "doc_id": None, "page": None,
            "category": "unanswerable", "probe_category": cat,
            "check_absent": checks, "why": why,
        })

    priority = {"near-miss": 0, "absent-attribute": 1, "entity-swap": 2,
                "false-bridge": 3}
    probes.sort(key=lambda p: (priority[p["probe_category"]], p["question"]))
    chosen, counts = [], Counter()
    quotas = {"near-miss": 15, "absent-attribute": 20, "entity-swap": 15,
              "false-bridge": 10}
    for p in probes:
        if len(chosen) >= args.n:
            break
        if counts[p["probe_category"]] < quotas[p["probe_category"]]:
            chosen.append(p)
            counts[p["probe_category"]] += 1
    for p in probes:
        if len(chosen) >= args.n:
            break
        if p not in chosen:
            chosen.append(p)
    chosen.sort(key=lambda p: (priority[p["probe_category"]], p["question"]))
    for i, p in enumerate(chosen):
        p["qid"] = f"frozen3-unans-{i:02d}"
    Path(args.out).write_text(json.dumps(chosen, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"WROTE {args.out}: {len(chosen)} probes")
    print(f"by category: {dict(Counter(p['probe_category'] for p in chosen))}")
    print(f"dropped because a check string was present: {len(dropped)}")
    if len(chosen) < args.n:
        print(f"WARNING: requested {args.n}, emitted {len(chosen)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
