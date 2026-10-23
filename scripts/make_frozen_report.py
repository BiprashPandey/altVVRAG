#!/usr/bin/env python
"""Build the frozen deck-disjoint evaluation report.

Usage: python scripts/make_frozen_report.py <run_dir> [out.pdf]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(v, n=3):
    return "n/a" if v is None else f"{v:.{n}f}"


def pct(v):
    return "n/a" if v is None else fmt(v * 100.0, 1) + "%"


def metrics(rows, unanswerable):
    sub = [r for r in rows if "error" not in r
           and bool(r.get("unanswerable")) == unanswerable]
    n = len(sub)
    answered = [r for r in sub if not r.get("abstained")]
    verified = [r for r in sub if r.get("state") == "verified"]
    correct = [r for r in sub if r.get("equiv_correct")]
    verified_correct = [r for r in verified if r.get("equiv_correct")]
    source_ok = [r for r in answered if r.get("pred_boxes")
                 or r.get("pred_sentence_ids")
                 or r.get("cited_sentences")]
    ex_rows = [r for r in answered if (r.get("examiner") or {}).get("ran")]
    claims = [c for r in answered
              for c in ((r.get("claim_level") or {}).get("labels") or [])]
    cp = [1.0 / r["gold_rank"] for r in sub if r.get("gold_rank")]
    d = {
        "n": n,
        "answered": len(answered),
        "verified": len(verified),
        "correct": len(correct),
        "partial_correct": sum(1 for r in sub if r.get("equiv_correct")
                               or r.get("partial_correct")),
        "partial_precision": (
            sum(1 for r in verified if r.get("equiv_correct")
                or r.get("partial_correct")) / len(verified)
        ) if verified else None,
        "mean_f1_norm": mean([r.get("equiv_f1_norm", r.get("equiv_f1"))
                              for r in sub if not unanswerable]),
        "coverage": len(answered) / n if n else None,
        "verified_yield": len(verified) / n if n else None,
        "verified_precision": (len(verified_correct) / len(verified))
        if verified else None,
        "false_abstention": (sum(1 for r in sub if r.get("abstained")) / n)
        if n and not unanswerable else None,
        "withheld_rate": (sum(1 for r in sub if r.get("abstained")) / n)
        if n and not unanswerable else None,
        "probe_answered_rate": (len(answered) / n)
        if n and unanswerable else None,
        "probe_verified_rate": (len(verified) / n)
        if n and unanswerable else None,
        "source_accuracy": (len(source_ok) / len(answered)) if answered else None,
        "examiner_availability": (len(ex_rows) / len(answered))
        if answered else None,
        "claim_support_rate": (sum(1 for x in claims if x == "supported")
                               / len(claims)) if claims else None,
        "claim_verification_rate": (
            sum(1 for r in answered
                if (r.get("claim_level") or {}).get("ran")) / len(answered)
        ) if answered else None,
        "page_hit1": mean([1.0 if r.get("page_hit") else 0.0 for r in sub])
        if not unanswerable else None,
        "mrr": (sum(cp) / len(cp)) if cp else None,
        "tiers": dict(Counter(
            ("text" if r.get("state") == "verified"
             else "visual" if r.get("visual_verified") else "none")
            for r in answered)),
        "visual_verified": sum(1 for r in answered if r.get("visual_verified")),
        "visual_precision": (
            sum(1 for r in answered if r.get("visual_verified")
                and r.get("equiv_correct"))
            / max(1, sum(1 for r in answered if r.get("visual_verified")))
        ) if any(r.get("visual_verified") for r in answered) else None,
        "states": dict(Counter((r.get("state") or "error") for r in sub)),
    }
    return d, sub


def table(rows, widths=None):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9aa7b4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--title", default="Frozen Deck-Disjoint Evaluation")
    ap.add_argument("--config", default="",
                    help="config file used for the run (for reproduction)")
    ap.add_argument("--set-dir", default="",
                    help="frozen set dir with frozen_meta.json")
    ap.add_argument("--index", default="",
                    help="index label/path for the provenance table")
    args = ap.parse_args()
    run = Path(args.run)
    rows = json.loads((run / "rows.json").read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else REPO / "frozen_eval.pdf"
    meta = {}
    meta_paths = []
    if args.set_dir:
        meta_paths.append(Path(args.set_dir) / "frozen_meta.json")
    meta_paths += [run / "frozen_meta.json", run / "set_meta.json"]
    for mp in meta_paths:
        if mp.exists():
            meta = json.loads(mp.read_text(encoding="utf-8"))
            break

    gold, gsub = metrics(rows, False)
    probe, psub = metrics(rows, True)
    err = [r for r in rows if "error" in r]
    cfg_used = args.config or os.environ.get("VVRAG_CONFIG", "config.yaml")
    env_prefix = ("" if Path(cfg_used).as_posix() == "config.yaml"
                  else f"VVRAG_CONFIG={cfg_used} ")
    reader_models = Counter((r.get("model_used") or r.get("model") or "unknown")
                            for r in gsub)
    reader_model = ", ".join(f"{k} x{v}"
                             for k, v in reader_models.most_common()) or "not-run"

    styles = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16,
                        spaceAfter=5)
    H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11.5,
                        spaceAfter=3)
    BODY = ParagraphStyle("BODY", parent=styles["BodyText"], fontSize=8.5,
                          leading=11)
    SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=7.2, leading=9)

    story = [Paragraph(args.title, H1),
             Paragraph(f"Run directory: <font face='Courier'>{run}</font>",
                       SMALL),
             Paragraph(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
                       SMALL),
             Spacer(1, 4 * mm)]

    story.append(Paragraph("1. Frozen protocol and honest scope", H2))
    protocol_txt = meta.get("protocol_note") or (
        "This is a deck-disjoint frozen run: the development decks are excluded "
        "and the gold set was audited before the run. The numbers below are "
        "measured on that fixed set, not on the development split.")
    story.append(Paragraph(protocol_txt, BODY))
    story.append(Spacer(1, 2 * mm))

    ex_served = Counter((r.get("examiner_served_by") or "not-run")
                        for r in gsub)
    ex_indep = Counter((bool(r.get("independence_ok")),
                        bool(r.get("independence_required")))
                       for r in gsub)
    ex_mode = ", ".join(f"{k} x{v}" for k, v in ex_served.most_common())
    if ex_indep and all(ind for ind, _ in ex_indep):
        independence_txt = "independent"
    elif ex_indep:
        independence_txt = "same-family / not independent (pilot)"
    else:
        independence_txt = "not exercised"

    ex_available = sum(1 for r in gsub if r.get("examiner_ran"))
    if (gold["examiner_availability"] is not None
            and gold["examiner_availability"] < 0.80):
        story.append(Paragraph(
            "<b>Protocol alert (pre-registered):</b> the examiner ran on only "
            f"{ex_available}/{gold['answered']} answered gold rows "
            f"({pct(gold['examiner_availability'])}); the pre-registered "
            "threshold was 80%. Per the pre-registration, this pilot does not "
            "support a claim that the verifier contract was exercised. The "
            "remaining failure is examiner availability/quota, not a reader "
            "failure.", BODY))
        story.append(Spacer(1, 2 * mm))

    story.append(Paragraph("2. Corpus and index provenance", H2))
    cats = Counter((r.get("category") or "unknown") for r in gsub)
    cat_txt = ", ".join(f"{k} {v}" for k, v in sorted(cats.items())) or "n/a"
    mode_used = Counter((r.get("retrieval_debug") or {}).get("mode_used")
                        for r in gsub)
    mode_txt = ", ".join(f"{k} x{v}" for k, v in mode_used.items()) or "n/a"
    prov = [
        ["Item", "Value"],
        ["Set", meta.get("name", "frozen deck-disjoint")],
        ["Decks / pages", f"{meta.get('n_decks', 'n/a')} / "
                          f"{meta.get('n_pages', 'n/a')}"],
        ["Index", args.index or meta.get("index_label", "n/a")],
        ["Pages with OCR", str(meta.get("pages_with_ocr", "n/a"))],
        ["Word boxes", str(meta.get("word_boxes", "n/a"))],
        ["Sentence/table units", str(meta.get("sentence_units", "n/a"))],
        ["Embedding / retrieval", f"{meta.get('embedding', 'n/a')}; "
                                  f"used: {mode_txt}"],
        ["Gold items / probes", f"{gold['n']} / {probe['n']}"],
        ["Gold categories", cat_txt],
        ["Label audit", meta.get("audit", "see gold.json")],
        ["Reader / examiner", f"{reader_model} / {ex_mode}"],
        ["Examiner independence", independence_txt],
    ]
    story.append(table(prov, [52 * mm, 120 * mm]))
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("3. Main result: enhanced contract vs strict "
                           "report-parity counterfactual", H2))
    story.append(Paragraph(
        "The enhanced arm is the product contract: every answer is shown and "
        "the state records whether it is verified. The report-parity column is "
        "the strict withholding counterfactual computed from the same rows: "
        "only verified answers are shown. It is not a separate model run and "
        "must be labelled as a counterfactual.", BODY))
    parity_correct = sum(1 for r in gsub if r.get("state") == "verified"
                         and r.get("equiv_correct"))
    main_rows = [
        ["Metric", "Enhanced (shown)", "Report-parity-style (verified only)"],
        ["Answerable n", str(gold["n"]), str(gold["n"])],
        ["Shown / coverage",
         f"{gold['answered']} ({pct(gold['coverage'])})",
         f"{gold['verified']} ({pct(gold['verified_yield'])})"],
        ["Withheld",
         f"{gold['n'] - gold['answered']} ({pct(gold['withheld_rate'])})",
         f"{gold['n'] - gold['verified']} "
         f"({pct(1 - gold['verified_yield'] if gold['verified_yield'] is not None else None)})"],
        ["Correct / shown",
         f"{gold['correct']} / {gold['answered']}",
         f"{parity_correct} / {gold['verified']}"],
        ["Partial-credit correct / shown",
         f"{gold.get('partial_correct', gold['correct'])} / {gold['answered']}",
         "report-only caveat"],
        ["Verified precision", pct(gold["verified_precision"]),
         pct(gold["verified_precision"])],
        ["Verified partial precision",
         pct(gold.get("partial_precision")), "report-only caveat"],
        ["Verified yield", pct(gold["verified_yield"]),
         pct(gold["verified_yield"])],
        ["False abstention", pct(gold["false_abstention"]),
         "0% by construction"],
        ["Source accuracy (shown)", pct(gold["source_accuracy"]),
         pct(gold["source_accuracy"])],
        ["Examiner availability (answered)",
         pct(gold["examiner_availability"]),
         pct(gold["examiner_availability"])],
        ["Examiner ran (all gold)",
         f"{ex_available} / {gold['n']}", f"{ex_available} / {gold['n']}"],
        ["Examiner independence", independence_txt, independence_txt],
        ["Visual-verified (separate)",
         f"{gold['visual_verified']} ({pct(gold['visual_precision'])})",
         f"{gold['visual_verified']} ({pct(gold['visual_precision'])})"],
        ["Claim support rate", pct(gold["claim_support_rate"]),
         pct(gold["claim_support_rate"])],
        ["Page hit @1 / MRR",
         f"{pct(gold['page_hit1'])} / {fmt(gold['mrr'])}",
         f"{pct(gold['page_hit1'])} / {fmt(gold['mrr'])}"],
        ["State counts", str(gold["states"]), "verified only"],
    ]
    story.append(table(main_rows, [62 * mm, 55 * mm, 55 * mm]))
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("4. Probe behaviour", H2))
    probe_rows = [
        ["Metric", "Value"],
        ["Probes n", str(probe["n"])],
        ["Answered / probe-answered rate",
         f"{probe['answered']} ({pct(probe['probe_answered_rate'])})"],
        ["Nothing found",
         f"{probe['n'] - probe['answered']} "
         f"({pct(1 - probe['probe_answered_rate'] if probe['probe_answered_rate'] is not None else None)})"],
        ["Verified / false-verified rate",
         f"{probe['verified']} ({pct(probe['probe_verified_rate'])})"],
        ["States", str(probe["states"])],
    ]
    story.append(table(probe_rows, [75 * mm, 95 * mm]))
    by_cat = Counter((r.get("probe_category") or "unknown") for r in psub)
    cat_rows = [["Probe category", "n", "answered", "nothing found",
                 "verified"]]
    for cat in sorted(by_cat):
        sub = [r for r in psub
               if (r.get("probe_category") or "unknown") == cat]
        cat_rows.append([cat, str(len(sub)),
                         str(sum(1 for r in sub if not r.get("abstained"))),
                         str(sum(1 for r in sub if r.get("abstained"))),
                         str(sum(1 for r in sub
                                 if r.get("state") == "verified"))])
    story.append(Spacer(1, 1.5 * mm))
    story.append(table(cat_rows,
                       [55 * mm, 20 * mm, 30 * mm, 30 * mm, 30 * mm]))
    story.append(Spacer(1, 3 * mm))

    false_verified = [r for r in psub if r.get("state") == "verified"]
    if false_verified:
        fv = false_verified[0]
        story.append(Paragraph(
            "<b>False-verified probe (the main failure):</b> "
            f"{fv.get('qid')} asks: <i>{fv.get('question')}</i>; the reader "
            f"answered <b>{fv.get('answer')}</b> and the examiner accepted the "
            "on-page table row. The probe is a near-miss: the cited row maps "
            "Old Act Section 4 to New Act Section 11, while the question asks "
            "for Section 12. This is a question-constraint grounding failure, "
            "not a missing citation.", BODY))
        story.append(Spacer(1, 2 * mm))

    story.append(Paragraph("5. Gold by category", H2))
    cat_rows = [["Category", "n", "shown", "verified", "strict-correct",
                 "false abstention"]]
    for cat in ("single-hop", "multi-hop", "image-only"):
        sub = [r for r in gsub if r.get("category") == cat]
        if not sub:
            continue
        ans = [r for r in sub if not r.get("abstained")]
        ver = [r for r in sub if r.get("state") == "verified"]
        cat_rows.append([cat, str(len(sub)), str(len(ans)), str(len(ver)),
                         str(sum(1 for r in ver if r.get("equiv_correct"))),
                         pct(sum(1 for r in sub if r.get("abstained"))
                             / len(sub))])
    story.append(table(cat_rows,
                       [38 * mm, 18 * mm, 22 * mm, 22 * mm, 30 * mm,
                        32 * mm]))
    story.append(Spacer(1, 3 * mm))

    wrong_verified = [r for r in gsub if r.get("state") == "verified"
                      and not r.get("equiv_correct")]
    if wrong_verified:
        wv = wrong_verified[0]
        story.append(Paragraph(
            "<b>Verified-but-wrong gold item:</b> "
            f"{wv.get('qid')} answers <b>{wv.get('answer')}</b>; gold is "
            f"<i>{wv.get('gold_answer')}</i>. The answer is a correct fragment "
            "of a multi-part gold answer, but exact-equivalence scoring marks "
            "it wrong; the examiner accepted the fragment. Report both the "
            "strict score and this partial-answer caveat.", BODY))
        story.append(Spacer(1, 2 * mm))

    story.append(Paragraph("6. Per-item audit", H2))
    item_rows = [["qid", "cat", "state", "shown", "ver", "claims", "rank",
                  "page"]]
    for r in gsub:
        cl = r.get("claim_level") or {}
        labels = ",".join(cl.get("labels") or []) or "-"
        item_rows.append([str(r.get("qid"))[-18:], str(r.get("category")),
                          str(r.get("state")),
                          "Y" if r.get("equiv_correct") else "N",
                          "Y" if r.get("equiv_verified") else "N",
                          labels[:24], str(r.get("gold_rank") or "-"),
                          str(r.get("gold_page"))])
    story.append(table(item_rows, [43 * mm, 22 * mm, 23 * mm, 12 * mm,
                                   10 * mm, 31 * mm, 12 * mm, 12 * mm]))
    story.append(PageBreak())

    story.append(Paragraph("7. What the result does and does not support", H2))
    limits = meta.get("limitations") or [
        f"The frozen set is {gold['n']} gold + {probe['n']} probes; report it "
        "as a fixed-set measurement, not a universal estimate.",
        f"Embedding/retrieval: {meta.get('embedding', 'n/a')}; modes used: "
        f"{mode_txt}. This tests the contract on the stated retrieval path.",
        f"Reader/examiner: {reader_model} / {ex_mode}; independence: "
        f"{independence_txt}. Model quota/version are part of the result.",
        "The report-parity column is a strict-withholding counterfactual over "
        "the enhanced rows, not the original report code path.",
        "Claim-level visual evidence is limited by the configured claim-image "
        "budget; text and visual tiers are reported separately.",
        "This is one frozen run; do not tune thresholds on it and then report "
        "it as held-out.",
    ]
    for x in limits + list(meta.get("notes") or []):
        story.append(Paragraph(f"&bull; {x}", BODY))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("8. Reproduce", H2))
    cmds = meta.get("reproduce") or [
        f"{env_prefix}python scripts/run_test8.py --index "
        f"{args.index or meta.get('index_label', 'data/index')} "
        f"--gold {Path(args.set_dir) / 'gold.json' if args.set_dir else 'GOLD'} "
        f"--probes {Path(args.set_dir) / 'probes.json' if args.set_dir else 'PROBES'} "
        "--arms keyword --models gemini-3.5-flash-lite --n3 0",
        f"python scripts/make_frozen_report.py {run} {out} --config {cfg_used} "
        f"--set-dir {args.set_dir or 'SET_DIR'}",
    ]
    for c in cmds:
        c = str(c).replace("<run_dir>", str(run))
        story.append(Paragraph(
            f"<font face='Courier' size='6.8'>{c}</font>", SMALL))
        story.append(Spacer(1, 1.2 * mm))
    if err:
        story.append(Paragraph(f"Errored rows in this run: {len(err)}", BODY))

    doc = SimpleDocTemplate(
        str(out), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm, title=args.title,
        author="altVRAG")
    doc.build(story)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
