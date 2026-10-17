"""testRun2 matrix: gold Q&A -> reader x grounding runs on the real index.

Reads gold from data/testrun2_gold.json (gitignored):
  [{"qid": "q1", "question": "...", "answer": "..."}]
Gold boxes = OCR word-span union via eval/gold_regions (no hand annotation).
Word-level output: predicted boxes snapped to word boxes
(ground/boxes.snap_box_to_words); metrics = pixel-IoU vs gold union +
word-level precision/recall + EM/F1 vs gold answer.

Writes results/demo/testrun2_{stamp}/{rows.json, overlays..., testrun2_table.md}.
Live readers go through ui/qa.py (disk cache + rate limiter).
"""
import datetime as dt
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

REPO = Path(__file__).resolve().parent.parent
INDEX = REPO / 'data' / 'index'
GOLD = REPO / 'data' / 'testrun2_gold.json'

from vvrag.eval.gold_regions import derive_answer_words, derive_gold_box  # noqa: E402
from vvrag.eval.metrics import (exact_match, f1, iou, center_inside, max_over_golds,  # noqa: E402
                                per_line_iou, union_iou, word_precision_recall_f1)
from vvrag.ground.boxes import snap_box_to_words  # noqa: E402
from vvrag.reader.claims import normalize_answer  # noqa: E402
from vvrag.schemas import WordBox  # noqa: E402
from vvrag.ui.qa import answer_question, page_words  # noqa: E402


def gold_span_words(answer: str, words: list[WordBox]) -> list[WordBox]:
    """Covering WORDS for an answer (shared impl, rec 1+7)."""
    span, _ = derive_answer_words(answer, words)
    return span


def gold_answers(item: dict) -> list[str]:
    """Several acceptable golds (rec 7): q5 short + long both correct."""
    ans = item['answer']
    extra = item.get('gold_answers') or item.get('alt_answers') or []
    seen = []
    for a in [ans, *extra]:
        if a and a not in seen:
            seen.append(a)
    return seen


def word_pr(pred: list[WordBox], gold: list[WordBox]) -> tuple[float, float]:
    """Word-level precision/recall on (text, box) identity (primary metric)."""
    p, r, _ = word_precision_recall_f1(pred, gold)
    return p, r


def main() -> None:
    gold = json.loads(GOLD.read_text(encoding='utf-8'))
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = REPO / 'results' / 'demo' / f'testrun2_{stamp}'
    out.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(INDEX / 'meta.db')

    rows = []
    for item in gold:
        qid, question, answer = item['qid'], item['question'], item['answer']
        words = page_words(con, 'testimg', 0)
        golds = gold_answers(item)
        # primary box/words for display; IoU/word scores take max over variants
        gbox = derive_gold_box(answer, words)
        gwords = gold_span_words(answer, words)
        grow = {'qid': qid, 'question': question, 'gold_answer': answer,
                'gold_answers': golds,
                'gold_box': list(gbox) if gbox else None,
                'gold_words': [w.text for w in gwords]}
        print(f'--- {qid}: {question[:70]}')
        print(f'    gold box={grow["gold_box"]} gold_words={len(gwords)}')
        for reader_mode in ('mock', 'gemini'):
            for grounding in ('span', 'heatmap'):
                if reader_mode == 'mock' and grounding == 'heatmap':
                    pass  # still runs (heatmap is reader-independent); keep full matrix
                wdir = out / f'{qid}_{reader_mode}_{grounding}'
                try:
                    res = answer_question(
                        question, str(INDEX), str(wdir), reader_mode=reader_mode,
                        provider='gemini', n_samples=1, grounding=grounding,
                        gate='fixed', fixed_tau=0.5)
                except Exception as e:  # noqa: BLE001 - record, don't stop the matrix
                    print(f'    {reader_mode}/{grounding} ERROR: {type(e).__name__}: {e}')
                    grow_msg = dict(grow)
                    grow_msg.update({'reader': reader_mode, 'grounding': grounding,
                                     'error': f'{type(e).__name__}: {e}'})
                    rows.append(grow_msg)
                    continue
                ev = res.get('evidence', {})
                pred_boxes = [tuple(b) for b in ev.get('boxes', [])]
                # rec 7: word-F1 primary; boxes are per-line so also report
                # union-IoU + per-line IoU (secondary) and multi-gold max EM/F1.
                loc = ev.get('localize', {}) or {}
                from vvrag.ground.text_in_box import words_in_boxes
                uniq_all = words_in_boxes(words, pred_boxes) if pred_boxes else []
                seen, uniq = set(), []
                for w in uniq_all:
                    if (w.text, w.box) not in seen:
                        seen.add((w.text, w.box))
                        uniq.append(w)
                golds = gold_answers({**item, 'answer': answer})
                em = max_over_golds(res.get('answer') or '', golds, exact_match)
                f1v = max_over_golds(res.get('answer') or '', golds, f1)
                # localization scored against best gold variant (rec 7)
                best_wp, best_wr, best_wf1, best_u, best_pl = 0.0, 0.0, 0.0, 0.0, 0.0
                for g in golds:
                    gw = gold_span_words(g, words)
                    gb = derive_gold_box(g, words)
                    p, r, f1w = word_precision_recall_f1(uniq, gw)
                    u = union_iou(pred_boxes, tuple(gb)) if pred_boxes and gb else 0.0
                    pl = per_line_iou(pred_boxes, tuple(gb)) if pred_boxes and gb else 0.0
                    if (f1w, u) >= (best_wf1, best_u):
                        best_wp, best_wr, best_wf1, best_u, best_pl = p, r, f1w, u, pl
                wp, wr, wf1, u_iou, pl_iou = best_wp, best_wr, best_wf1, best_u, best_pl
                row = dict(grow)
                row.update({
                    'reader': reader_mode, 'grounding': grounding,
                    'answer': res.get('answer'), 'abstained': res.get('abstained'),
                    'abstain_reason': res.get('abstain_reason', ''),
                    'score': res.get('score'),
                    'em': em,
                    'f1': f1v,
                    'pred_boxes': [list(b) for b in pred_boxes],
                    'pred_words': [w.text for w in uniq],
                    'iou': round(u_iou, 4),
                    'per_line_iou': round(pl_iou, 4),
                    'center_inside': bool(center_inside(pred_boxes[0], tuple(gbox)))
                    if pred_boxes and gbox else False,
                    'word_p': round(wp, 3), 'word_r': round(wr, 3),
                    'word_f1': round(wf1, 3),
                    'localize': loc,
                    'claims': res.get('claims'), 'samples': res.get('samples'),
                    'features': res.get('features'), 'api_calls': res.get('api_calls'),
                    'timings': res.get('timings'),
                    'work_dir': str(wdir),
                })
                rows.append(row)
                print(f"    {reader_mode}/{grounding}: ans={row['answer']!r:.60} "
                      f"EM={row['em']} F1={row['f1']:.2f} IoU={row['iou']:.2f} "
                      f"plIoU={row['per_line_iou']:.2f} "
                      f"wP/R/F1={row['word_p']}/{row['word_r']}/{row['word_f1']} "
                      f"score={row['score']} abst={row['abstained']} api={row['api_calls']}")
    (out / 'rows.json').write_text(json.dumps(rows, indent=2, default=str), encoding='utf-8')
    print(f'\nWROTE {out / "rows.json"} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
