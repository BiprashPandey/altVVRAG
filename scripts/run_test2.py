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

from vvrag.eval.gold_regions import derive_gold_box  # noqa: E402
from vvrag.eval.metrics import exact_match, f1, iou, center_inside  # noqa: E402
from vvrag.ground.boxes import snap_box_to_words  # noqa: E402
from vvrag.reader.claims import normalize_answer  # noqa: E402
from vvrag.schemas import WordBox  # noqa: E402
from vvrag.ui.qa import answer_question, page_words  # noqa: E402


def gold_span_words(answer: str, words: list[WordBox]) -> list[WordBox]:
    """Mirror of derive_gold_box, but returns the covering WORDS."""
    import difflib

    target = normalize_answer(answer)
    if not target:
        return []
    norm_words = [normalize_answer(w.text) for w in words]
    joined = ' '.join(norm_words)
    if target in joined:
        s, e = joined.find(target), joined.find(target) + len(target)
        pos, span = 0, []
        for w, nw in zip(words, norm_words):
            w_end = pos + len(nw)
            if w_end > s and pos < e and nw:
                span.append(w)
            pos = w_end + 1
        if span:
            return span
    best, best_r = None, 0.0
    for w, nw in zip(words, norm_words):
        r = difflib.SequenceMatcher(None, target, nw).ratio()
        if r > best_r:
            best, best_r = w, r
    if best is not None and best_r >= 0.8:
        return [w for w in words
                if w.block_id == best.block_id and w.line_id == best.line_id]
    return []


def word_pr(pred: list[WordBox], gold: list[WordBox]) -> tuple[float, float]:
    """Word-level precision/recall on (text, box) identity."""
    if not pred or not gold:
        return 0.0, 0.0
    g = {(w.text, w.box) for w in gold}
    p = [(w.text, w.box) for w in pred]
    hit = sum(1 for t in p if t in g)
    return hit / max(1, len(p)), hit / max(1, len(g))


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
        gbox = derive_gold_box(answer, words)
        gwords = gold_span_words(answer, words)
        grow = {'qid': qid, 'question': question, 'gold_answer': answer,
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
                pred_words: list[WordBox] = []
                for b in pred_boxes:
                    inside, _union = snap_box_to_words(b, words)
                    pred_words.extend(inside)
                seen, uniq = set(), []
                for w in pred_words:
                    if (w.text, w.box) not in seen:
                        seen.add((w.text, w.box))
                        uniq.append(w)
                wp, wr = word_pr(uniq, gwords)
                row = dict(grow)
                row.update({
                    'reader': reader_mode, 'grounding': grounding,
                    'answer': res.get('answer'), 'abstained': res.get('abstained'),
                    'score': res.get('score'),
                    'em': exact_match(res.get('answer') or '', answer),
                    'f1': f1(res.get('answer') or '', answer),
                    'pred_boxes': [list(b) for b in pred_boxes],
                    'pred_words': [w.text for w in uniq],
                    'iou': iou(pred_boxes[0], tuple(gbox)) if pred_boxes and gbox else 0.0,
                    'center_inside': bool(center_inside(pred_boxes[0], tuple(gbox)))
                    if pred_boxes and gbox else False,
                    'word_p': round(wp, 3), 'word_r': round(wr, 3),
                    'claims': res.get('claims'), 'samples': res.get('samples'),
                    'features': res.get('features'), 'api_calls': res.get('api_calls'),
                    'timings': res.get('timings'),
                    'work_dir': str(wdir),
                })
                rows.append(row)
                print(f"    {reader_mode}/{grounding}: ans={row['answer']!r:.60} "
                      f"EM={row['em']} F1={row['f1']:.2f} IoU={row['iou']:.2f} "
                      f"wP/R={row['word_p']}/{row['word_r']} score={row['score']} "
                      f"abst={row['abstained']} api={row['api_calls']}")
    (out / 'rows.json').write_text(json.dumps(rows, indent=2, default=str), encoding='utf-8')
    print(f'\nWROTE {out / "rows.json"} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
