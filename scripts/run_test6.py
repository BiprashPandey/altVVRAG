"""testRun6 matrix: SlideVQA — 40 QA items x {span,heatmap} = 80 live rows.

The first BROADER-COVERAGE run (Phase C): real presentation decks (8 decks,
160 OCR'd slide pages, 9.8k word boxes) instead of the 1-page sample docs.
Index: data/index (slidevqa_colab.ipynb export), gold: data/gold.json
(scripts/prepare_slidevqa.py).

Design vs testRun5:
  - FULL-CORPUS retrieval (doc_id=None): matches the real UI path; the reader
    gets the top-ranked page across ALL 160 pages (gold page rank recorded).
  - GOLD page (not page 0): boxes/sentences derived from gold.json's page.
  - Page-aware metrics: geometry/sentence metrics are ZERO when the pipeline
    answered on a different page than gold (coordinates are page-local).
  - MODEL ROTATION on 429: free-tier quota is ~20 req/day/MODEL; on
    RESOURCE_EXHAUSTED the row immediately retries with the next candidate
    model (rotating cursor persists across rows).

Usage:
  python scripts/run_test6.py                       # full matrix, paced
  python scripts/run_test6.py --resume <dir>        # re-run errored rows
  python scripts/run_test6.py --limit 4             # smoke: first 4 rows
"""
import argparse
import datetime as dt
import itertools
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

REPO = Path(__file__).resolve().parent.parent
INDEX = REPO / 'data' / 'index'
GOLD_PATH = REPO / 'data' / 'gold.json'

from vvrag.eval.gold_regions import derive_answer_words, derive_gold_box  # noqa: E402
from vvrag.eval.metrics import (center_inside, equivalent_correct,  # noqa: E402
                                equivalent_f1, per_line_iou,
                                sentence_precision_recall, union_iou,
                                word_precision_recall_f1)
from vvrag.ground.crop import overlay_heatmap  # noqa: E402
from vvrag.ground.text_in_box import words_in_boxes  # noqa: E402
from vvrag.ingest.sentences import sentence_ids_for_answer, sentences_from_page  # noqa: E402
from vvrag.ui.qa import answer_question, page_words  # noqa: E402

# Free-tier candidates in preference order; the cursor advances on 429/503.
# Quota is per-model (~20/day), so rotation buys ~20 calls per bucket.
# Probed 2026-10-18: 2.5-flash-lite is 404 (removed); 3.7-flash was 503
# (transient, kept last); others served.
# Phase 0 (2026-10-18): defaults now come from config.yaml `run:` (single
# source of truth for harnesses); the lists below stay as an offline fallback.
MODEL_CANDIDATES = [
    'gemini-3.5-flash-lite',
    'gemini-3.8-flash',
    'gemini-3.6-flash',
    'gemini-3.1-flash-lite',
    'gemini-3.7-flash',
]

LIVE_CALL_GAP_S = 25.0  # stay under every model's per-minute limit
try:
    from vvrag.config import load_config as _load_config

    _RUN_CFG = _load_config().run
    if _RUN_CFG.model_candidates:
        MODEL_CANDIDATES = list(_RUN_CFG.model_candidates)
    LIVE_CALL_GAP_S = float(_RUN_CFG.live_call_gap_s)
    _PINNED_MODEL = str(_RUN_CFG.reader_model_pinned or '')
except Exception:  # noqa: BLE001 - config is optional for offline mock runs
    _PINNED_MODEL = ''

_last_live_call = [0.0]
_model_cycle = itertools.cycle(MODEL_CANDIDATES)
READER_MODE = 'gemini'  # set to 'mock' by --mock for offline plumbing checks


def _next_model() -> str:
    if _PINNED_MODEL:
        return _PINNED_MODEL
    return next(_model_cycle)


def paced_answer_question(*args, **kwargs):
    if READER_MODE == 'mock':
        return answer_question(*args, **kwargs)
    wait = LIVE_CALL_GAP_S - (time.time() - _last_live_call[0])
    if wait > 0:
        print(f'    (pacing: sleeping {wait:.0f}s for quota)')
        time.sleep(wait)
    try:
        return answer_question(*args, **kwargs)
    finally:
        _last_live_call[0] = time.time()


def _is_quota_error(e: Exception) -> bool:
    s = str(e)
    return ('RESOURCE_EXHAUSTED' in s or '429' in s
            or 'quota' in s.lower() or '503' in s)


def run_single(item, idx, grounding, wdir, model=""):
    """One matrix cell -> row dict. Rotates model on quota errors."""
    doc_id, gpage = item['doc_id'], int(item['page'])
    models_tried = []
    res, last_err, used_model = None, None, model
    n_attempts = len(MODEL_CANDIDATES) + 1 if READER_MODE == 'gemini' else 1
    for _ in range(n_attempts):
        used_model = model if (model and not models_tried) else _next_model()
        if used_model in models_tried:
            models_tried.append(used_model)
            continue
        models_tried.append(used_model)
        try:
            res = paced_answer_question(
                item['question'], str(INDEX), str(wdir), reader_mode=READER_MODE,
                provider='gemini', model=used_model, n_samples=1,
                grounding=grounding, gate='calibrated')  # full corpus: doc_id=None
            last_err = None
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            if _is_quota_error(e):
                print(f'    {grounding} 429 on {used_model}, rotating...')
                continue
            print(f'    {grounding} ERROR {type(e).__name__}: {str(e)[:100]}')
            break
    if last_err is not None or res is None:
        row = _base_row(item, idx, grounding, gpage)
        row.update({'error': f'{type(last_err).__name__}: {last_err}',
                    'models_tried': models_tried})
        return row

    ev = res.get('evidence', {})
    pred_boxes = [tuple(b) for b in ev.get('boxes', [])]
    page_hit = (ev.get('doc_id') == doc_id and ev.get('page') == gpage)

    # gold context is on the GOLD page; pred boxes/sents are on the EVIDENCE
    # page -> geometry/sentence metrics only comparable when pages match.
    gold_words_all = page_words_sql(doc_id, gpage)
    gwords = derive_answer_words(item['answer'], gold_words_all)[0]
    gbox = derive_gold_box(item['answer'], gold_words_all)
    gcon = sqlite3.connect(str(INDEX / 'meta.db'))
    gsents, _ = sentences_from_page(gcon, doc_id, gpage)
    gcon.close()
    gold_sids = sentence_ids_for_answer(item['answer'], gsents)

    # Metric policy (matches gold_regions: "no match -> excluded from
    # LOCALIZATION metrics only"): None = not evaluable, excluded from means.
    #  - wrong page -> geometry/sentence metrics None (page-local coords)
    #  - gold sids empty (answer absent from OCR) -> sentence metrics None
    #  - gold box absent (answer not in OCR words) -> box metrics None
    if page_hit:
        uniq_all = words_in_boxes(gold_words_all, pred_boxes) if pred_boxes else []
        seen, uniq = set(), []
        for w in uniq_all:
            if (w.text, w.box) not in seen:
                seen.add((w.text, w.box))
                uniq.append(w)
        pred_sids = ev.get('sentence_ids', [])
        if gold_sids and pred_sids:
            s_p, s_r, s_f1 = sentence_precision_recall(pred_sids, gold_sids)
        else:
            s_p = s_r = s_f1 = None
        if gwords and pred_boxes:
            wp, wr, wf1 = word_precision_recall_f1(uniq, gwords)
        else:
            wp = wr = wf1 = None
        if gbox and pred_boxes:
            iou = union_iou(pred_boxes, tuple(gbox))
            pl_iou = per_line_iou(pred_boxes, tuple(gbox))
            cin = bool(center_inside(pred_boxes[0], tuple(gbox)))
        else:
            iou = pl_iou = None
            cin = None
    else:
        uniq = []
        wp = wr = wf1 = s_p = s_r = s_f1 = iou = pl_iou = cin = None

    answer = res.get('answer') or ''
    equiv = (not res.get('abstained')) and equivalent_correct(
        answer, [item['answer'], *item.get('acceptable_answers', [])])
    loc = ev.get('localize', {})

    # stable image names for the PDF report
    crop_p, ov_p = wdir / 'crop.png', wdir / 'overlay.png'
    crops = sorted(wdir.glob('*_crop.png'))
    if crops:
        shutil.copy(crops[0], crop_p)
    epage_img = None
    if pred_boxes:
        import sqlite3 as _sq
        con2 = _sq.connect(str(INDEX / 'meta.db'))
        r = con2.execute('SELECT image_path FROM pages WHERE doc_id=? AND page=?',
                         (ev.get('doc_id'), ev.get('page'))).fetchone()
        con2.close()
        epage_img = r[0] if r else None
    if pred_boxes and epage_img and Path(epage_img).exists():
        import numpy as np
        overlay_heatmap(epage_img, np.zeros((8, 6), dtype=np.float32),
                        [list(b) for b in pred_boxes], ov_p)
    else:
        ovs = sorted(wdir.glob('*_overlay.png'))
        if ovs:
            shutil.copy(ovs[0], ov_p)

    # gold-page reference overlay: gold box on the gold slide (for the report)
    gold_ov = wdir / 'gold_overlay.png'
    gold_img = _page_image(doc_id, gpage)
    if gbox and gold_img and Path(gold_img).exists():
        import numpy as np
        overlay_heatmap(gold_img, np.zeros((8, 6), dtype=np.float32),
                        [list(gbox)], gold_ov)

    row = _base_row(item, idx, grounding, gpage)
    row.update({
        'reader_model': used_model, 'models_tried': models_tried,
        'answer': answer, 'abstained': res.get('abstained'),
        'abstain_reason': res.get('abstain_reason', ''),
        'score': res.get('score'), 'tau': res.get('tau'),
        'equiv_correct': bool(equiv),
        'equiv_f1': round(equivalent_f1(
            answer, [item['answer'], *item.get('acceptable_answers', [])]), 4),
        'page_hit': bool(page_hit),
        'ev_doc': ev.get('doc_id'), 'ev_page': ev.get('page'),
        'pred_boxes': [list(b) for b in pred_boxes],
        'pred_words': [w.text for w in uniq],
        'pred_sentence_ids': list(ev.get('sentence_ids', [])),
        'gold_sentence_ids': gold_sids,
        'gold_box': list(gbox) if gbox else None,
        'sent_p': _r(s_p, 3), 'sent_r': _r(s_r, 3), 'sent_f1': _r(s_f1, 3),
        'quote_ok': loc.get('quote_ok'), 'quote_score': loc.get('quote_score'),
        'iou': _r(iou, 4), 'per_line_iou': _r(pl_iou, 4),
        'center_inside': cin,
        'word_p': _r(wp, 3), 'word_r': _r(wr, 3), 'word_f1': _r(wf1, 3),
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
        'gold_overlay': str(gold_ov) if gold_ov.exists() else None,
        'work_dir': str(wdir),
    })
    ranks = [r.get('page') for r in (res.get('retrieval') or [])]
    gold_rank = next((i + 1 for i, p in enumerate(ranks)
                      if res['retrieval'][i].get('doc_id') == doc_id and p == gpage),
                     None)
    row['gold_rank'] = gold_rank
    print(f'    {grounding} [{used_model}]: ans={answer!r:.46} equiv={equiv} '
          f'page_hit={page_hit} (gold_rank={gold_rank}) '
          f'abst={row["abstained"]} quote={row["quote_ok"]} sentF1={row["sent_f1"]}')
    return row


def _base_row(item, idx, grounding, gpage):
    return {'idx': idx, 'qid': item['qid'], 'question': item['question'],
            'gold_answer': item['answer'],
            'acceptable_answers': item.get('acceptable_answers', []),
            'doc_id': item['doc_id'], 'gold_page': gpage,
            'grounding': grounding}


def _r(v, n):
    """round() that passes None through (metric not evaluable)."""
    return None if v is None else round(v, n)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


_page_image_cache: dict = {}


def _page_image(doc_id, page):
    key = (doc_id, page)
    if key not in _page_image_cache:
        con = sqlite3.connect(str(INDEX / 'meta.db'))
        r = con.execute('SELECT image_path FROM pages WHERE doc_id=? AND page=?',
                        (doc_id, page)).fetchone()
        con.close()
        _page_image_cache[key] = r[0] if r else None
    return _page_image_cache[key]


_words_cache: dict = {}


def page_words_sql(doc_id, page):
    key = (doc_id, page)
    if key not in _words_cache:
        con = sqlite3.connect(str(INDEX / 'meta.db'))
        rows = con.execute(
            'SELECT text,x0,y0,x1,y1,block_id,line_id FROM word_boxes '
            'WHERE doc_id=? AND page=?', (doc_id, page)).fetchall()
        con.close()
        from vvrag.schemas import WordBox
        _words_cache[key] = [WordBox(doc_id, page, r[0],
                                     (r[1], r[2], r[3], r[4]), r[5], r[6])
                             for r in rows]
    return _words_cache[key]


def gold_items():
    return json.loads(GOLD_PATH.read_text(encoding='utf-8'))


def summarize(rows):
    ok = [r for r in rows if 'error' not in r]
    errs = [r for r in rows if 'error' in r]
    n = len(ok)
    answered = [r for r in ok if not r.get('abstained')]
    equiv = sum(1 for r in ok if r.get('equiv_correct')) / max(1, n)
    false_abs = sum(1 for r in ok if r.get('abstained')) / max(1, n)
    page_hit = sum(1 for r in ok if r.get('page_hit')) / max(1, n)
    quote_ok = sum(1 for r in answered if r.get('quote_ok')) / max(1, len(answered))
    # None-aware means: only rows where the metric is evaluable count.
    sent_f1 = _mean([r.get('sent_f1') for r in answered])
    iou_m = _mean([r.get('iou') for r in answered])
    word_f1 = _mean([r.get('word_f1') for r in answered])
    n_sent = sum(1 for r in answered if r.get('sent_f1') is not None)
    n_box = sum(1 for r in answered if r.get('iou') is not None)
    sp = [r for r in answered if r.get('page_hit')]
    sp_f1 = _mean([r.get('sent_f1') for r in sp])
    sp_iou = _mean([r.get('iou') for r in sp])
    gold_rank_vals = [r.get('gold_rank') for r in ok if r.get('gold_rank')]
    mrr = _mean([1.0 / r for r in gold_rank_vals]) if gold_rank_vals else None
    print('\n=== testRun6 summary ===')
    print(f'rows={len(rows)} errors={len(errs)} done={n} answered={len(answered)}')
    print(f'equiv_correct={equiv:.3f}  false_abstention={false_abs:.3f}  '
          f'page_hit={page_hit:.3f}  quote_ok(answered)={quote_ok:.3f}')
    print(f'gold_rank MRR={mrr if mrr is None else round(mrr, 3)} '
          f'(gold in top-5: {sum(1 for r in ok if r.get("gold_rank"))}/{n})')
    print(f'sent_f1={_fmt(sent_f1)} over {n_sent} evaluable | '
          f'iou={_fmt(iou_m)} / word_f1={_fmt(word_f1)} over {n_box} evaluable')
    print(f'on page_hit subset: sent_f1={_fmt(sp_f1)} iou={_fmt(sp_iou)} '
          f'(n={len(sp)})')
    return {'rows': len(rows), 'errors': len(errs), 'equiv_correct': equiv,
            'false_abstention': false_abs, 'page_hit': page_hit,
            'quote_ok_rate': quote_ok, 'mean_sent_f1': sent_f1,
            'mean_iou': iou_m, 'mean_word_f1': word_f1,
            'n_sent_eval': n_sent, 'n_box_eval': n_box,
            'mrr': mrr, 'sent_f1_on_page': sp_f1, 'iou_on_page': sp_iou}


def _fmt(v):
    return 'n/a' if v is None else f'{v:.3f}'


def build_rows(gold, groundings, out, limit=0):
    # Pre-seed ALL target rows as error placeholders so a killed run can be
    # completed with --resume (unfinished rows are visible as errors).
    targets = [(i, item, g) for i, item in enumerate(gold)
               if not limit or i < limit for g in groundings]
    rows = [dict(_base_row(item, i, g, int(item['page'])),
                 error='PENDING: not run yet')
            for i, item, g in targets]
    (out / 'rows.json').write_text(
        json.dumps(rows, indent=2, default=str), encoding='utf-8')

    for k, (i, item, grounding) in enumerate(targets):
        print(f'--- [{i + 1}/{len(gold)}] {item["qid"][:60]} '
              f'(gold: {item["doc_id"][:28]} p{item["page"]})')
        wdir = out / f'{i:03d}_{grounding}'
        try:
            rows[k] = run_single(item, i, grounding, wdir)
        except Exception as e:  # noqa: BLE001 - keep the batch alive
            print(f'    {grounding} POST-ERROR {type(e).__name__}: '
                  f'{str(e)[:120]}')
            rows[k] = _base_row(item, i, grounding, int(item['page']))
            rows[k]['error'] = f'{type(e).__name__}: {e}'
        # incremental save: a crash/ctrl-C never loses finished rows
        (out / 'rows.json').write_text(
            json.dumps(rows, indent=2, default=str), encoding='utf-8')
    return rows


def main():
    global INDEX, GOLD_PATH, READER_MODE, _PINNED_MODEL
    ap = argparse.ArgumentParser(description='testRun6 SlideVQA live matrix')
    ap.add_argument('--index', default=str(INDEX))
    ap.add_argument('--gold', default=str(GOLD_PATH))
    ap.add_argument('--out', default='')
    ap.add_argument('--groundings', default='span,heatmap')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--model', default='',
                    help='single model (no rotation); default: rotate on 429')
    ap.add_argument('--mock', action='store_true',
                    help='offline mock reader (plumbing smoke, no API)')
    args, rest = ap.parse_known_args(sys.argv[1:])

    if args.mock:
        READER_MODE = 'mock'

    if rest and rest[0] == '--resume':
        INDEX = Path(args.index)
        resume(Path(rest[1]))
        return

    INDEX = Path(args.index)
    GOLD_PATH = Path(args.gold)
    groundings = [g.strip() for g in args.groundings.split(',') if g.strip()]
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Path(args.out) if args.out else (REPO / 'results' / 'demo' / f'testrun6_{stamp}')
    out.mkdir(parents=True, exist_ok=True)
    gold = gold_items()
    if args.model:
        MODEL_CANDIDATES.clear()
        MODEL_CANDIDATES.append(args.model)
        _PINNED_MODEL = args.model

    rows = build_rows(gold, groundings, out, limit=args.limit)
    (out / 'rows.json').write_text(json.dumps(rows, indent=2, default=str),
                                   encoding='utf-8')
    summarize(rows)
    print(f'\nWROTE {out / "rows.json"} ({len(rows)} rows)')


def resume(out: Path) -> None:
    rows = json.loads((out / 'rows.json').read_text(encoding='utf-8'))
    gold = {g['qid']: g for g in gold_items()}
    fixed = 0
    for i, row in enumerate(rows):
        if 'error' not in row:
            continue
        item = gold[row['qid']]
        wdir = out / f"{row['idx']:03d}_{row['grounding']}"
        print(f'--- resume [{row["idx"]}] {row["qid"][:50]} '
              f'(was: {row["error"][:70]})')
        try:
            rows[i] = run_single(item, row['idx'], row['grounding'], wdir)
        except Exception as e:  # noqa: BLE001 - keep the batch alive
            print(f'    POST-ERROR {type(e).__name__}: {str(e)[:120]}')
            rows[i] = dict(row, error=f'{type(e).__name__}: {e}')
        if 'error' not in rows[i]:
            fixed += 1
        (out / 'rows.json').write_text(
            json.dumps(rows, indent=2, default=str), encoding='utf-8')
    (out / 'rows.json').write_text(json.dumps(rows, indent=2, default=str),
                                   encoding='utf-8')
    summarize(rows)
    n_err = sum(1 for r in rows if 'error' in r)
    print(f'\nRESUME DONE: fixed {fixed}, still errored {n_err}/{len(rows)}')


if __name__ == '__main__':
    main()
