"""Run 9 Phase 3 - examiner wrong-answer injection proof (offline, guideline s4.8).

For a handful of gold items, the reader is scripted to return three claims:

1. POSITIVE   - the true answer with its real sentence as the quote;
2. HALLUCINATED- a corrupted answer plus a fabricated quote that appears
   nowhere in the document (the "model invents a citation" failure);
3. NO_EVIDENCE - no_evidence=true with UNANSWERABLE (the honest nothing).

Expected states: verified / not VERIFIED (not_confident or nothing_found) /
nothing_found. The script fails (exit 1) if any expectation breaks.

Documented mock limitation: the deterministic MockExaminer confirms once the
CLAIMED QUOTE is present in the OCR text, so it cannot catch an answer that is
wrong while the quote itself is real (a real examiner reads the image and is
exercised live / by tests/test_examiner.py verdict logic). The hallucinated
case is exactly the class the examiner + quote check must catch, and that is
what is asserted here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from vvrag.ingest.index import connect_sqlite  # noqa: E402
from vvrag.ingest.sentences import (sentence_ids_for_answer,  # noqa: E402
                                    sentences_from_page)
from vvrag.reader.claims import normalize_answer  # noqa: E402
from vvrag.schemas import ReaderSample  # noqa: E402


def _corrupt(answer: str, banned: str) -> str:
    """A plausible wrong answer (guideline s4.7 injection)."""
    import re

    ans = str(answer or "")
    cands: list[str] = []
    nums = list(re.finditer(r"\d+(?:\.\d+)?", ans))
    if nums:
        last = nums[-1]
        try:
            val = float(last.group(0))
            cands.append(ans[:last.start()] + f"{val + 7:g}" + ans[last.end():])
            if val >= 2:
                cands.append(ans[:last.start()] + f"{val - 2:g}" + ans[last.end():])
        except ValueError:
            pass
    toks = ans.split()
    if toks:
        cands.append(" ".join(["unrelated"] + toks[1:]))
    cands.append(ans + " (see appendix)")
    banned_n = normalize_answer(banned)
    for c in cands:
        if normalize_answer(c) not in banned_n:
            return c
    return ans + " approximately"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=str(REPO / "data" / "index"))
    ap.add_argument("--gold", default=str(REPO / "data" / "gold.json"))
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--out", default=str(REPO / "results" /
                                         "examiner_injection_report.json"))
    args = ap.parse_args()

    index = Path(args.index)
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    gold = [g for g in gold if g.get("answer_in_ocr")] or gold

    import vvrag.reader.client as rclient
    from vvrag.ui.qa import answer_question

    cases: list[dict] = []
    scripted: dict = {}

    real_run = rclient.run_reader_samples

    def fake_run(reader, images, prompt, n_samples=1, temperature=0.0):
        return [scripted["sample"]], 0

    rclient.run_reader_samples = fake_run
    accepted = 0
    try:
        for item in gold:
            if accepted >= args.limit:
                break
            doc_id, page = str(item["doc_id"]), int(item["page"])
            con = connect_sqlite(index / "meta.db")
            try:
                sents, _ = sentences_from_page(con, doc_id, page)
            finally:
                con.close()
            gold_sids = sentence_ids_for_answer(item["answer"], sents)
            sid = gold_sids[0] if gold_sids else None
            sent = next((s for s in sents if s.sent_id == sid), None)
            if sent is None:
                cases.append({"qid": item["qid"], "status": "skipped",
                              "why": "no derivable gold sentence"})
                continue
            page_text = " ".join(s.text for s in sents)
            wrong = _corrupt(item["answer"], page_text)
            qid = item["qid"]

            def _run(kind, sample):
                wdir = REPO / "results" / "examiner_injection" / f"{qid}_{kind}"
                wdir.mkdir(parents=True, exist_ok=True)
                scripted["sample"] = sample
                t0 = time.time()
                res = answer_question(item["question"], str(index),
                                      str(wdir), reader_mode="mock",
                                      n_samples=1, grounding="span",
                                      gate="calibrated")
                return res, round(time.time() - t0, 1)

            # POSITIVE first; skip the item when retrieval misses the gold
            # page (the scripted citation is page-bound, so the comparison
            # would be vacuous, not a verifier failure).
            try:
                res, secs = _run("positive", ReaderSample(
                    sample_id=0, answer=item["answer"], quote=sent.text,
                    sentence_ids=[sid]))
            except Exception as e:  # noqa: BLE001
                cases.append({"qid": qid, "variant": "positive",
                              "status": "error", "why": str(e)[:200]})
                continue
            ev = res.get("evidence") or {}
            if ev.get("page") != page or str(ev.get("doc_id")) != str(doc_id):
                cases.append({"qid": qid, "variant": "positive",
                              "status": "skipped",
                              "why": f"retrieval miss (top page "
                                     f"{ev.get('page')} != gold {page})"})
                print(f"[{qid}] skipped: retrieval missed the gold page")
                continue
            det = res.get("details") or {}
            if not det.get("ok", True):
                cases.append({"qid": qid, "variant": "positive",
                              "status": "skipped",
                              "why": "question details not in the one cited "
                                     "sentence (multi-hop construction)"})
                print(f"[{qid}] skipped: {res.get('state_reason', '')[:70]}")
                continue
            accepted += 1
            pos = {"qid": qid, "variant": "positive", "expect": "verified",
                   "state": res.get("state") or "?", "seconds": secs,
                   "state_reason": res.get("state_reason", ""),
                   "answer": (res.get("answer") or "")[:80],
                   "examiner_served_by": res.get("examiner_served_by"),
                   "examiner": res.get("examiner")}
            pos["ok"] = pos["state"] == "verified"
            cases.append(pos)
            print(f"[{qid}|positive] state={pos['state']} ok={pos['ok']} "
                  f"({pos['state_reason'][:70]})")

            variants = [
                ("hallucinated", ReaderSample(
                    sample_id=0, answer=wrong,
                    quote="The board approved a restructuring plan in "
                          "September 1987 as described in appendix Z.",
                    sentence_ids=[sid]), "not_confident"),
                ("no_evidence", ReaderSample(
                    sample_id=0, answer="UNANSWERABLE", no_evidence=True),
                 "nothing_found"),
            ]
            for kind, sample, expect in variants:
                try:
                    res, secs = _run(kind, sample)
                except Exception as e:  # noqa: BLE001
                    cases.append({"qid": qid, "variant": kind,
                                  "status": "error", "why": str(e)[:200]})
                    continue
                state = res.get("state") or "?"
                ok = state == expect
                cases.append({
                    "qid": qid, "variant": kind, "expect": expect,
                    "state": state, "ok": bool(ok),
                    "state_reason": res.get("state_reason", ""),
                    "answer": (res.get("answer") or "")[:80],
                    "quote_ok": (res.get("evidence", {}) or {}).get("quote_ok")
                    if isinstance(res.get("evidence"), dict) else None,
                    "examiner_served_by": res.get("examiner_served_by"),
                    "examiner": res.get("examiner"),
                    "seconds": secs,
                })
                print(f"[{qid}|{kind}] state={state} ok={ok} "
                      f"({cases[-1]['state_reason'][:70]})")
    finally:
        rclient.run_reader_samples = real_run

    ran = [c for c in cases if c.get("status") != "skipped"]
    passed = [c for c in ran if c.get("ok")]
    failed = [c for c in ran if c.get("status") == "error" or not c.get("ok")]
    report = {"n_cases": len(cases), "n_ran": len(ran),
              "n_passed": len(passed), "n_failed": len(failed),
              "cases": cases}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWROTE {out}: {len(passed)}/{len(ran)} expectations met "
          f"({len(failed)} failed)")
    return 0 if not failed and ran else 1


if __name__ == "__main__":
    raise SystemExit(main())
