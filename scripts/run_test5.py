"""testRun5 matrix: the 5 questions (with equivalent golds) + an unanswerable probe.

Runs the FULL improved pipeline live (Gemini, n_samples=1) against the existing
index, exercising every Phase A-F change:
  - retrieval never abstains; abstain only at the gate / for UNANSWERABLE;
  - OCR-tolerant quote check (config verify.ocr_normalize) + one UNANSWERABLE
    retry with closest passages (reader.retry_on_unanswerable);
  - final boxes from cited sentences, tightened to the quote (refine_box_to_quote);
  - gate = CALIBRATED (data/gate/scorer.pkl + tau.json).
Metrics per row include EQUIVALENT-correct (any acceptable gold, P3) and the
sentence/word/quote stats. A summary line reports equiv-correct, false-abstention
(answerable), and abstention (unanswerable) rates — Test Run 5's targets.

Usage:
  python scripts/run_test5.py                    # full matrix, paced live calls
  python scripts/run_test5.py --resume <dir> [model]
"""
import argparse
import datetime as dt
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

REPO = Path(__file__).resolve().parent.parent
INDEX = REPO / 'data' / 'index'

from vvrag.eval.gold_regions import derive_answer_words, derive_gold_box  # noqa: E402
from vvrag.eval.metrics import (center_inside, equivalent_correct,  # noqa: E402
                                equivalent_f1, per_line_iou,
                                sentence_precision_recall, union_iou,
                                word_precision_recall_f1)
from vvrag.ground.crop import overlay_heatmap  # noqa: E402
from vvrag.ground.text_in_box import words_in_boxes  # noqa: E402
from vvrag.ingest.sentences import sentence_ids_for_answer, sentences_from_page  # noqa: E402
from vvrag.ui.qa import answer_question, page_words  # noqa: E402

GOLD = [
    {"qid": "q1_title", "question": "What is the title of the document?",
     "answer": "Sample Document for PDF Testing",
     "acceptable_answers": [],
     "why": "heading words, clean in both versions."},
    {"qid": "q2_emph", "question": "What kind of emphasis can italic text give?",
     "answer": "subtle emphasis", "acceptable_answers": [],
     "why": "exact span in both; png nearby text garbled."},
    {"qid": "q3_list", "question": "What is the second entry of the ordered list?",
     "answer": "Second item", "acceptable_answers": [],
     "why": "exact in both; png list lines noisy."},
    {"qid": "q4_quote", "question": "What can a block quote be used to highlight?",
     "answer": "important information",
     "acceptable_answers": ["important information or citations",
                            "important information or citations."],
     "why": "P3 equivalent golds: reader may answer the longer form."},
    {"qid": "q5_table", "question": "What is in Row 1, Column 3 of the table?",
     "answer": "Row 1, Col 3", "acceptable_answers": ["Row 1, Col3"],
     "why": "png OCR reads 'Col3'; equivalent gold covers the typo."},
    # unanswerable probe (suggestion #8): the gate must abstain here.
    {"qid": "q6_absent", "question": "What is the CEO's annual salary in this document?",
     "answer": "UNANSWERABLE", "acceptable_answers": [],
     "is_unanswerable": True,
     "why": "not present in the sample; honest abstention expected."},
]

VERSIONS = [("pdf", "sample"), ("png", "sample_page")]

LIVE_CALL_GAP_S = 25.0  # free-tier ~5 req/min
_last_live_call = [0.0]


def paced_answer_question(*args, **kwargs):
    if kwargs.get("reader_mode", "mock") != "mock":
        wait = LIVE_CALL_GAP_S - (time.time() - _last_live_call[0])
        if wait > 0:
            print(f'    (pacing: sleeping {wait:.0f}s for quota)')
            time.sleep(wait)
        try:
            return answer_question(*args, **kwargs)
        finally:
            _last_live_call[0] = time.time()
    return answer_question(*args, **kwargs)


def golds_of(item):
    return [item['answer'], *[a for a in item.get('acceptable_answers', []) if a]]


def run_single(item, ver, doc_id, grounding, words, gbox, gwords,
               grow, wdir, img_path, model=""):
    try:
        res = paced_answer_question(
            item['question'], str(INDEX), str(wdir), reader_mode='gemini',
            provider='gemini', model=model, n_samples=1, grounding=grounding,
            gate='calibrated', doc_id=doc_id)
    except Exception as e:  # noqa: BLE001
        print(f'    {grounding} ERROR: {type(e).__name__}: {str(e)[:100]}')
        row = dict(grow)
        row.update({'grounding': grounding, 'error': f'{type(e).__name__}: {e}'})
        return row
    ev = res.get('evidence', {})
    pred_boxes = [tuple(b) for b in ev.get('boxes', [])]
    uniq_all = words_in_boxes(words, pred_boxes) if pred_boxes else []
    seen, uniq = set(), []
    for w in uniq_all:
        if (w.text, w.box) not in seen:
            seen.add((w.text, w.box))
            uniq.append(w)
    wp, wr, wf1 = word_precision_recall_f1(uniq, gwords)
    pred_sids = ev.get('sentence_ids', [])
    gold_sids = grow.get('gold_sentence_ids', [])
    s_p, s_r, s_f1 = sentence_precision_recall(pred_sids, gold_sids)
    loc = ev.get('localize', {})
    answer = res.get('answer') or ''
    is_unans = bool(item.get('is_unanswerable'))
    equiv = (not res.get('abstained')) and equivalent_correct(answer, golds_of(item))
    crop_p = wdir / 'crop.png'
    ov_p = wdir / 'overlay.png'
    crops = sorted(wdir.glob('*_crop.png'))
    if crops:
        shutil.copy(crops[0], crop_p)
    if pred_boxes and img_path and Path(img_path).exists():
        import numpy as np
        overlay_heatmap(img_path, np.zeros((8, 6), dtype=np.float32),
                        [list(b) for b in pred_boxes], ov_p)
    else:
        ovs = sorted(wdir.glob('*_overlay.png'))
        if ovs:
            shutil.copy(ovs[0], ov_p)
    row = dict(grow)
    row.update({
        'grounding': grounding,
        'reader_model': model or 'gemini-3.5-flash(config)',
        'answer': answer, 'abstained': res.get('abstained'),
        'abstain_reason': res.get('abstain_reason', ''),
        'score': res.get('score'), 'tau': res.get('tau'),
        'is_unanswerable': is_unans,
        'equiv_correct': bool(equiv),
        'equiv_f1': round(equivalent_f1(answer, golds_of(item)), 4),
        'pred_boxes': [list(b) for b in pred_boxes],
        'pred_words': [w.text for w in uniq],
        'pred_sentence_ids': list(pred_sids),
        'sent_p': round(s_p, 3), 'sent_r': round(s_r, 3), 'sent_f1': round(s_f1, 3),
        'quote_ok': loc.get('quote_ok'), 'quote_score': loc.get('quote_score'),
        'iou': round(union_iou(pred_boxes, tuple(gbox))
                     if pred_boxes and gbox else 0.0, 4),
        'per_line_iou': round(per_line_iou(pred_boxes, tuple(gbox))
                                if pred_boxes and gbox else 0.0, 4),
        'center_inside': bool(center_inside(pred_boxes[0], tuple(gbox)))
        if pred_boxes and gbox else False,
        'word_p': round(wp, 3), 'word_r': round(wr, 3), 'word_f1': round(wf1, 3),
        'localize': loc,
        'cited_sentences': res.get('cited_sentences', []),
        'closest_passages': res.get('closest_passages', []),
        'claims': res.get('claims'), 'samples': res.get('samples'),
        'features': res.get('features'),
        'retrieval': res.get('retrieval'),
        'api_calls': res.get('api_calls'),
        'timings': res.get('timings'),
        'crop': str(crop_p) if crop_p.exists() else None,
        'overlay': str(ov_p) if ov_p.exists() else None,
        'work_dir': str(wdir),
    })
    tag = 'UNANSWERABLE-gold' if is_unans else 'answerable'
    print(f'    {grounding}: ans={answer!r:.50} equiv={equiv} '
          f'abst={row["abstained"]} ({row["abstain_reason"]}) '
          f'quote={row["quote_ok"]} sentF1={row["sent_f1"]} [{tag}]')
    return row


def gold_context(con, item, doc_id):
    words = page_words(con, doc_id, 0)
    is_unans = bool(item.get('is_unanswerable'))
    gbox = None if is_unans else derive_gold_box(item['answer'], words)
    gwords = [] if is_unans else derive_answer_words(item['answer'], words)[0]
    sents, _ = sentences_from_page(con, doc_id, 0)
    gsent_ids = [] if is_unans else sentence_ids_for_answer(item['answer'], sents)
    grow = {'qid': item['qid'], 'version': None, 'doc_id': doc_id,
            'question': item['question'], 'gold_answer': item['answer'],
            'acceptable_answers': item.get('acceptable_answers', []),
            'gold_box': list(gbox) if gbox else None,
            'gold_words': [w.text for w in gwords],
            'gold_sentence_ids': gsent_ids}
    return words, gbox, gwords, grow


def summarize(rows):
    ans = [r for r in rows if not r.get('is_unanswerable') and 'error' not in r]
    una = [r for r in rows if r.get('is_unanswerable') and 'error' not in r]
    errs = [r for r in rows if 'error' in r]
    n_ans = len(ans)
    equiv = sum(1 for r in ans if r.get('equiv_correct')) / max(1, n_ans)
    false_abs = sum(1 for r in ans if r.get('abstained')) / max(1, n_ans)
    quote_ok = sum(1 for r in ans if r.get('quote_ok')) / max(1, n_ans)
    una_abs = (sum(1 for r in una if r.get('abstained')) / max(1, len(una))
               if una else 0.0)
    sent_f1 = (sum(r.get('sent_f1', 0) for r in ans) / max(1, n_ans))
    print('\n=== testRun5 summary ===')
    print(f'rows={len(rows)} errors={len(errs)} '
          f'answerable={n_ans} unanswerable={len(una)}')
    print(f'equiv_correct={equiv:.3f}  false_abstention={false_abs:.3f}  '
          f'quote_ok_rate={quote_ok:.3f}  mean_sent_f1={sent_f1:.3f}')
    if una:
        print(f'unanswerable_abstention_rate={una_abs:.3f}')
    return {'equiv_correct': equiv, 'false_abstention': false_abs,
            'quote_ok_rate': quote_ok, 'mean_sent_f1': sent_f1,
            'unanswerable_abstention_rate': una_abs, 'errors': len(errs)}


def main():
    global INDEX
    ap = argparse.ArgumentParser(description='testRun5 live matrix')
    ap.add_argument('--index', default=str(INDEX))
    ap.add_argument('--out', default='')
    args, resume_argv = ap.parse_known_args(sys.argv[1:])

    if resume_argv and resume_argv[0] == '--resume':
        INDEX = Path(args.index)
        resume(Path(resume_argv[1]),
               resume_argv[2] if len(resume_argv) > 2 else "")
        return

    INDEX = Path(args.index)
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Path(args.out) if args.out else (REPO / 'results' / 'demo' / f'testrun5_{stamp}')
    out.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(INDEX / 'meta.db')

    rows = []
    for item in GOLD:
        for ver, doc_id in VERSIONS:
            words, gbox, gwords, grow = gold_context(con, item, doc_id)
            grow['version'] = ver
            img_path = con.execute(
                'SELECT image_path FROM pages WHERE doc_id=? AND page=0',
                (doc_id,)).fetchone()[0]
            print(f'--- {item["qid"]}/{ver}: gold_sents={grow["gold_sentence_ids"]}')
            for grounding in ('span', 'heatmap'):
                wdir = out / f'{item["qid"]}_{ver}_{grounding}'
                rows.append(run_single(item, ver, doc_id, grounding, words,
                                       gbox, gwords, grow, wdir, img_path))
    (out / 'rows.json').write_text(json.dumps(rows, indent=2, default=str),
                                   encoding='utf-8')
    summarize(rows)
    print(f'\nWROTE {out / "rows.json"} ({len(rows)} rows)')


def resume(out: Path, model: str = "") -> None:
    if model:
        print(f'resume with reader model override: {model}')
    rows = json.loads((out / 'rows.json').read_text(encoding='utf-8'))
    con = sqlite3.connect(INDEX / 'meta.db')
    by_qid = {g['qid']: g for g in GOLD}
    fixed = 0
    for i, row in enumerate(rows):
        if 'error' not in row:
            continue
        item = by_qid[row['qid']]
        ver, doc_id, grounding = row['version'], row['doc_id'], row['grounding']
        words, gbox, gwords, grow = gold_context(con, item, doc_id)
        grow['version'] = ver
        wdir = out / f"{row['qid']}_{ver}_{grounding}"
        img_path = con.execute(
            'SELECT image_path FROM pages WHERE doc_id=? AND page=0',
            (doc_id,)).fetchone()[0]
        print(f'--- resume {row["qid"]}/{ver}/{grounding} (was: {row["error"][:80]})')
        rows[i] = run_single(item, ver, doc_id, grounding, words, gbox,
                             gwords, grow, wdir, img_path, model=model)
        if 'error' not in rows[i]:
            fixed += 1
    (out / 'rows.json').write_text(json.dumps(rows, indent=2, default=str),
                                   encoding='utf-8')
    summarize(rows)
    n_err = sum(1 for r in rows if 'error' in r)
    print(f'\nRESUME DONE: fixed {fixed}, still errored {n_err}/{len(rows)}')


if __name__ == '__main__':
    main()
