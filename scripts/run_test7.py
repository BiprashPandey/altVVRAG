"""testRun7: SlideVQA — 40 gold items + 9 unanswerable probes, single grounding.

The Phase 6 "honest" run:
  - retrieval: HYBRID (keyword + ColQwen visual RRF) when query vectors are
    cached, else keyword (mode_used recorded per row; visual scoring needs
    data/query_cache vectors from scripts/precompute_query_vectors.py).
  - reader: multi-page context (top-3 pages), sentence refs resolved across
    pages (Phase 2), reader_model pinned (config run.reader_model_pinned or
    --model), quota rotation otherwise.
  - verification: quote check + answer-support veto (Phase 3).
  - gate: 'calibrated' (12-col artifacts in data/gate; regenerate with
    scripts/calibrate_gate.py).
  - metrics: number-normalized equivalence (equivalent_correct_norm), page
    hit@1/@3, category breakdown (single-hop/multi-hop/image-only) with
    bootstrap CIs, unanswerable abstention rate + false-answer rate.
  - n=3 samples on the first `--n3` gold items (default 15), n=1 elsewhere.

Usage:
  python scripts/run_test7.py                     # full run, paced
  python scripts/run_test7.py --limit 2           # smoke (gold only)
  python scripts/run_test7.py --mock              # offline plumbing smoke
  python scripts/run_test7.py --resume <dir>      # re-run errored rows
"""
from __future__ import annotations

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
PROBE_PATH = REPO / 'data' / 'gold_unanswerable.json'

from vvrag.eval.gold_regions import derive_answer_words, derive_gold_box  # noqa: E402
from vvrag.eval.metrics import (bootstrap_ci, center_inside,  # noqa: E402
                                equivalent_correct,
                                equivalent_correct_norm, equivalent_f1,
                                equivalent_f1_norm, per_line_iou,
                                sentence_precision_recall, union_iou,
                                word_precision_recall_f1)
from vvrag.ground.crop import overlay_heatmap  # noqa: E402
from vvrag.ground.text_in_box import words_in_boxes  # noqa: E402
from vvrag.ingest.sentences import sentence_ids_for_answer, sentences_from_page  # noqa: E402
from vvrag.ui.qa import answer_question  # noqa: E402

MODEL_CANDIDATES = [
    'gemini-3.5-flash-lite',
    'gemini-3.8-flash',
    'gemini-3.6-flash',
    'gemini-3.1-flash-lite',
    'gemini-3.7-flash',
]

LIVE_CALL_GAP_S = 25.0
_PINNED_MODEL = ''
try:
    from vvrag.config import load_config as _load_config

    _RUN_CFG = _load_config().run
    if _RUN_CFG.model_candidates:
        MODEL_CANDIDATES = list(_RUN_CFG.model_candidates)
    LIVE_CALL_GAP_S = float(_RUN_CFG.live_call_gap_s)
    _PINNED_MODEL = str(_RUN_CFG.reader_model_pinned or '')
except Exception:  # noqa: BLE001
    pass

_last_live_call = [0.0]
_model_cycle = itertools.cycle(MODEL_CANDIDATES)
READER_MODE = 'gemini'
RETRIEVAL_MODE = 'hybrid'
GROUNDING = 'span'


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
            or 'quota' in s.lower() or '503' in s
            or 'PERMISSION_DENIED' in s or 'denied access' in s.lower()
            or 'getaddrinfo' in s or 'ConnectError' in type(e).__name__)


def run_single(item, idx, wdir, n_samples):
    """One row (gold or probe). Rotates model on quota errors."""
    unanswerable = bool(item.get('category') == 'unanswerable')
    doc_id = item.get('doc_id')
    gpage = None if unanswerable else int(item['page'])
    models_tried = []
    res, last_err, used_model = None, None, ''
    n_attempts = len(MODEL_CANDIDATES) + 1 if READER_MODE == 'gemini' else 1
    for _ in range(n_attempts):
        used_model = (_PINNED_MODEL if _PINNED_MODEL and not models_tried
                      else _next_model())
        if used_model in models_tried:
            models_tried.append(used_model)
            continue
        models_tried.append(used_model)
        try:
            res = paced_answer_question(
                item['question'], str(INDEX), str(wdir), reader_mode=READER_MODE,
                provider='gemini', model=used_model, n_samples=n_samples,
                grounding=GROUNDING, gate='calibrated',
                retrieval_mode=RETRIEVAL_MODE)  # full corpus: doc_id=None
            last_err = None
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            if _is_quota_error(e):
                print(f'    429 on {used_model}, rotating...')
                continue
            print(f'    ERROR {type(e).__name__}: {str(e)[:100]}')
            break
    if last_err is not None or res is None:
        row = _base_row(item, idx, n_samples)
        row.update({'error': f'{type(last_err).__name__}: {last_err}',
                    'models_tried': models_tried})
        return row

    ev = res.get('evidence', {})
    pred_boxes = [tuple(b) for b in ev.get('boxes', [])]
    ret_debug = res.get('retrieval_debug') or {}
    ranks = [(r.get('doc_id'), r.get('page')) for r in (res.get('retrieval') or [])]
    gold_rank = None if unanswerable else next(
        (i + 1 for i, (d, p) in enumerate(ranks)
         if d == doc_id and p == gpage), None)
    page_hit = None if unanswerable else bool(
        ev.get('doc_id') == doc_id and ev.get('page') == gpage)

    gwords = gbox = None
    gold_sids = []
    if not unanswerable:
        gold_words_all = page_words_sql(doc_id, gpage)
        gwords = derive_answer_words(item['answer'], gold_words_all)[0]
        gbox = derive_gold_box(item['answer'], gold_words_all)
        gcon = sqlite3.connect(str(INDEX / 'meta.db'))
        gsents, _ = sentences_from_page(gcon, doc_id, gpage)
        gcon.close()
        gold_sids = sentence_ids_for_answer(item['answer'], gsents)

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
            iou_v = union_iou(pred_boxes, tuple(gbox))
            pl_iou = per_line_iou(pred_boxes, tuple(gbox))
            cin = bool(center_inside(pred_boxes[0], tuple(gbox)))
        else:
            iou_v = pl_iou = None
            cin = None
    else:
        uniq = []
        wp = wr = wf1 = s_p = s_r = s_f1 = iou_v = pl_iou = cin = None

    answer = res.get('answer') or ''
    abstained = bool(res.get('abstained'))
    golds = [item['answer'], *item.get('acceptable_answers', [])]
    if unanswerable:
        equiv = abstained  # correct behavior for probes = abstain
        eq_f1 = None
    else:
        equiv = (not abstained) and equivalent_correct_norm(answer, golds)
        eq_f1 = round(equivalent_f1_norm(answer, golds), 4)

    crop_p, ov_p = wdir / 'crop.png', wdir / 'overlay.png'
    crops = sorted(wdir.glob('*_crop.png'))
    if crops:
        shutil.copy(crops[0], crop_p)
    ovs = sorted(wdir.glob('*_overlay.png'))
    if ovs:
        shutil.copy(ovs[0], ov_p)

    loc = ev.get('localize', {})
    row = _base_row(item, idx, n_samples)
    row.update({
        'reader_model': used_model, 'models_tried': models_tried,
        'answer': answer, 'abstained': abstained,
        'abstain_reason': res.get('abstain_reason', ''),
        'score': res.get('score'), 'tau': res.get('tau'),
        'verified': res.get('verified'), 'support': res.get('support'),
        'equiv_correct': bool(equiv),
        'equiv_correct_raw': (bool(abstained) if unanswerable else
                              (not abstained) and equivalent_correct(answer, golds)),
        'equiv_f1': eq_f1,
        'page_hit': page_hit, 'gold_rank': gold_rank,
        'ev_doc': ev.get('doc_id'), 'ev_page': ev.get('page'),
        'pred_boxes': [list(b) for b in pred_boxes],
        'pred_words': [w.text for w in uniq],
        'pred_sentence_ids': list(ev.get('sentence_ids', [])),
        'gold_sentence_ids': gold_sids,
        'gold_box': list(gbox) if gbox else None,
        'sent_p': _r(s_p, 3), 'sent_r': _r(s_r, 3), 'sent_f1': _r(s_f1, 3),
        'quote_ok': loc.get('quote_ok'), 'quote_score': loc.get('quote_score'),
        'iou': _r(iou_v, 4), 'per_line_iou': _r(pl_iou, 4),
        'center_inside': cin,
        'word_p': _r(wp, 3), 'word_r': _r(wr, 3), 'word_f1': _r(wf1, 3),
        'localize': loc,
        'cited_sentences': res.get('cited_sentences', []),
        'closest_passages': res.get('closest_passages', []),
        'claims': res.get('claims'), 'samples': res.get('samples'),
        'features': res.get('features'),
        'retrieval': res.get('retrieval'),
        'retrieval_debug': ret_debug,
        'api_calls': res.get('api_calls'),
        'timings': res.get('timings'),
        'crop': str(crop_p) if crop_p.exists() else None,
        'overlay': str(ov_p) if ov_p.exists() else None,
        'work_dir': str(wdir),
    })
    print(f'    [{used_model}] ans={answer!r:.46} equiv={equiv} '
          f'page_hit={page_hit} rank={gold_rank} abst={abstained} '
          f'quote={row["quote_ok"]} ver={row["verified"]} '
          f'mode={ret_debug.get("mode_used")}')
    return row


def _base_row(item, idx, n_samples):
    return {'idx': idx, 'qid': item['qid'], 'question': item['question'],
            'gold_answer': item['answer'],
            'acceptable_answers': item.get('acceptable_answers', []),
            'doc_id': item.get('doc_id'), 'gold_page': item.get('page'),
            'category': item.get('category', 'single-hop'),
            'unanswerable': item.get('category') == 'unanswerable',
            'n_samples': n_samples, 'grounding': GROUNDING,
            'retrieval_mode_req': RETRIEVAL_MODE}


def _r(v, n):
    return None if v is None else round(v, n)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _fmt(v):
    return 'n/a' if v is None else f'{v:.3f}'


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


def load_items():
    gold = json.loads(GOLD_PATH.read_text(encoding='utf-8'))
    probes = []
    if PROBE_PATH.exists():
        probes = json.loads(PROBE_PATH.read_text(encoding='utf-8'))
    return gold, probes


def summarize(rows):
    ok = [r for r in rows if 'error' not in r]
    errs = [r for r in rows if 'error' in r]
    gold = [r for r in ok if not r.get('unanswerable')]
    probes = [r for r in ok if r.get('unanswerable')]
    answered = [r for r in gold if not r.get('abstained')]

    equiv = [1.0 if r.get('equiv_correct') else 0.0 for r in gold]
    e_mean, e_lo, e_hi = bootstrap_ci(equiv) if equiv else (0.0, 0.0, 0.0)
    false_abs = _mean([1.0 if r.get('abstained') else 0.0 for r in gold])
    page_hit = _mean([bool(r.get('page_hit')) for r in gold])
    hit3 = _mean([1.0 if (r.get('gold_rank') or 99) <= 3 else 0.0 for r in gold])
    mrrs = [1.0 / r['gold_rank'] for r in gold if r.get('gold_rank')]
    mrr_v = (sum(mrrs) / len(mrrs)) if mrrs else None
    quote_ok = _mean([bool(r.get('quote_ok')) for r in answered])
    verified = _mean([bool(r.get('verified')) for r in answered])
    sent_f1 = _mean([r.get('sent_f1') for r in answered])
    iou_m = _mean([r.get('iou') for r in answered])
    n_sent = sum(1 for r in answered if r.get('sent_f1') is not None)
    n_box = sum(1 for r in answered if r.get('iou') is not None)

    probe_abs = [1.0 if r.get('abstained') else 0.0 for r in probes]
    p_mean, p_lo, p_hi = (bootstrap_ci(probe_abs) if probe_abs
                          else (0.0, 0.0, 0.0))

    print('\n=== testRun7 summary ===')
    print(f'rows={len(rows)} errors={len(errs)} '
          f'gold={len(gold)} probes={len(probes)}')
    print(f'GOLD: equiv_correct(norm)={_fmt(e_mean)} CI[{_fmt(e_lo)},'
          f'{_fmt(e_hi)}]  false_abstention={_fmt(false_abs)}  '
          f'page_hit@1={_fmt(page_hit)} hit@3={_fmt(hit3)}  '
          f'MRR={_fmt(mrr_v)}')
    print(f'GOLD answered={len(answered)}  quote_ok={_fmt(quote_ok)}  '
          f'verified={_fmt(verified)}')
    print(f'GOLD sent_f1={_fmt(sent_f1)} (n={n_sent})  '
          f'iou={_fmt(iou_m)} (n={n_box})')
    print(f'PROBES: abstention={_fmt(p_mean)} CI[{_fmt(p_lo)},{_fmt(p_hi)}] '
          f'({len(probes)} items)')
    for cat in ('single-hop', 'multi-hop', 'image-only'):
        sub = [r for r in gold if r.get('category') == cat]
        if sub:
            acc = _mean([1.0 if r.get('equiv_correct') else 0.0 for r in sub])
            fa = _mean([1.0 if r.get('abstained') else 0.0 for r in sub])
            print(f'  {cat:<11} n={len(sub):<3} equiv={_fmt(acc)} '
                  f'false_abst={_fmt(fa)}')
    modes = {}
    for r in ok:
        m = (r.get('retrieval_debug') or {}).get('mode_used')
        modes[m] = modes.get(m, 0) + 1
    print(f'retrieval modes used: {modes}')
    return {'rows': len(rows), 'errors': len(errs),
            'equiv_correct': e_mean, 'equiv_ci': [e_lo, e_hi],
            'false_abstention': false_abs, 'page_hit1': page_hit,
            'hit3': hit3, 'mrr': mrr_v, 'quote_ok_rate': quote_ok,
            'verified_rate': verified, 'mean_sent_f1': sent_f1,
            'mean_iou': iou_m, 'n_sent_eval': n_sent, 'n_box_eval': n_box,
            'probe_abstention': p_mean, 'probe_ci': [p_lo, p_hi],
            'retrieval_modes': modes}


def build_rows(gold, probes, out, n3=15, limit=0):
    targets = []
    for i, item in enumerate(gold):
        if limit and i >= limit:
            break
        targets.append((i, item, 3 if i < n3 else 1))
    if not limit:
        for j, item in enumerate(probes):
            targets.append((len(gold) + j, item, 1))

    rows = [dict(_base_row(item, i, ns), error='PENDING: not run yet')
            for i, item, ns in targets]
    (out / 'rows.json').write_text(json.dumps(rows, indent=2, default=str),
                                   encoding='utf-8')

    for k, (i, item, ns) in enumerate(targets):
        print(f'--- [{k + 1}/{len(targets)}] idx={i} n={ns} '
              f'{item["qid"][:56]}')
        wdir = out / f'{i:03d}'
        try:
            rows[k] = run_single(item, i, wdir, ns)
        except Exception as e:  # noqa: BLE001
            print(f'    POST-ERROR {type(e).__name__}: {str(e)[:120]}')
            rows[k] = _base_row(item, i, ns)
            rows[k]['error'] = f'{type(e).__name__}: {e}'
        (out / 'rows.json').write_text(
            json.dumps(rows, indent=2, default=str), encoding='utf-8')
    return rows


def main():
    global INDEX, GOLD_PATH, PROBE_PATH, READER_MODE, RETRIEVAL_MODE, GROUNDING
    global _PINNED_MODEL
    ap = argparse.ArgumentParser(description='testRun7 gold+probes live run')
    ap.add_argument('--index', default=str(INDEX))
    ap.add_argument('--gold', default=str(GOLD_PATH))
    ap.add_argument('--probes', default=str(PROBE_PATH))
    ap.add_argument('--out', default='')
    ap.add_argument('--grounding', default='span', choices=['span', 'heatmap'])
    ap.add_argument('--retrieval-mode', default='hybrid',
                    choices=['keyword', 'visual', 'hybrid'])
    ap.add_argument('--n3', type=int, default=15,
                    help='first N gold items use n_samples=3')
    ap.add_argument('--limit', type=int, default=0,
                    help='only first N gold items (probes skipped)')
    ap.add_argument('--model', default='',
                    help='single model (no rotation); default: rotate on 429')
    ap.add_argument('--mock', action='store_true')
    args, rest = ap.parse_known_args(sys.argv[1:])

    if args.mock:
        READER_MODE = 'mock'
    RETRIEVAL_MODE = args.retrieval_mode
    GROUNDING = args.grounding

    if rest and rest[0] == '--resume':
        INDEX = Path(args.index)
        GOLD_PATH = Path(args.gold)
        PROBE_PATH = Path(args.probes)
        resume(Path(rest[1]))
        return

    INDEX = Path(args.index)
    GOLD_PATH = Path(args.gold)
    PROBE_PATH = Path(args.probes)
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Path(args.out) if args.out else (
        REPO / 'results' / 'demo' / f'testrun7_{stamp}')
    out.mkdir(parents=True, exist_ok=True)
    gold, probes = load_items()
    if args.model:
        MODEL_CANDIDATES.clear()
        MODEL_CANDIDATES.append(args.model)
        _PINNED_MODEL = args.model

    rows = build_rows(gold, probes, out, n3=args.n3, limit=args.limit)
    (out / 'rows.json').write_text(json.dumps(rows, indent=2, default=str),
                                   encoding='utf-8')
    summarize(rows)
    print(f'\nWROTE {out / "rows.json"} ({len(rows)} rows)')


def resume(out: Path) -> None:
    rows = json.loads((out / 'rows.json').read_text(encoding='utf-8'))
    gold, probes = load_items()
    by_qid = {g['qid']: g for g in [*gold, *probes]}
    fixed = 0
    for i, row in enumerate(rows):
        if 'error' not in row:
            continue
        item = by_qid[row['qid']]
        wdir = out / f"{row['idx']:03d}"
        print(f'--- resume idx={row["idx"]} {row["qid"][:50]} '
              f'(was: {row["error"][:70]})')
        try:
            rows[i] = run_single(item, row['idx'], wdir, row.get('n_samples', 1))
        except Exception as e:  # noqa: BLE001
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
