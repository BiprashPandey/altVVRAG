"""testRun8: arm x model matrix for the Run 8 plan (Phases E/F).

Matrix (pre-registered):
  model A x keyword  x all items (40 gold + 41 probes)   <- baseline
  model B x keyword  x all items                          <- model variance
  model A x hybrid   x results/run8_hybrid_items.json (24) <- visual retriever
  (probes run in the keyword arms only)

Per row: retrieval mode used, model used (+ quota exceptions), tier
(text|visual|none), citations resolution, visual confirmation, support veto,
quote check, strict (text-verified) vs any-tier equivalence, page_hit,
sent-f1/IoU when evaluable, CP intervals for probe abstention.

Colab query-cache PRECONDITION: the hybrid arm refuses to run (SystemExit)
unless every selected question has a cached ColQwen query vector
(scripts/precompute_query_vectors.py). Use --allow-missing-vectors to fall
back to keyword per row instead (recorded in the row).

Usage:
  python scripts/run_test8.py --mock --limit 6          # offline plumbing smoke
  python scripts/run_test8.py --check-cache             # preflight only
  python scripts/run_test8.py                            # full live matrix
  python scripts/run_test8.py --resume <dir>             # re-run errored rows
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
HYBRID_ITEMS = REPO / 'results' / 'run8_hybrid_items.json'

from vvrag.eval.gold_regions import derive_answer_words, derive_gold_box  # noqa: E402
from vvrag.eval.metrics import (bootstrap_ci, center_inside,  # noqa: E402
                                clopper_pearson_interval,
                                equivalent_correct, equivalent_correct_norm,
                                equivalent_f1, equivalent_f1_norm, per_line_iou,
                                sentence_precision_recall, union_iou,
                                word_precision_recall_f1)
from vvrag.ground.crop import overlay_heatmap  # noqa: E402
from vvrag.ground.text_in_box import words_in_boxes  # noqa: E402
from vvrag.ingest.sentences import sentence_ids_for_answer, sentences_from_page  # noqa: E402
from vvrag.ui.qa import answer_question  # noqa: E402

MODEL_CANDIDATES = [
    'gemini-3.5-flash-lite',
    'gemini-3.1-flash-lite',
    'gemini-3.8-flash',
    'gemini-3.6-flash',
    'gemini-3.7-flash',
]
DEFAULT_MODELS = ['gemini-3.5-flash-lite', 'gemini-3.8-flash']
LIVE_CALL_GAP_S = 25.0

try:
    from vvrag.config import load_config as _load_config

    _RUN_CFG = _load_config().run
    if _RUN_CFG.model_candidates:
        MODEL_CANDIDATES = list(_RUN_CFG.model_candidates)
    if _RUN_CFG.reader_model_pinned:
        DEFAULT_MODELS = [str(_RUN_CFG.reader_model_pinned)] + [
            m for m in MODEL_CANDIDATES if m != str(_RUN_CFG.reader_model_pinned)][:1]
    LIVE_CALL_GAP_S = float(_RUN_CFG.live_call_gap_s)
except Exception:  # noqa: BLE001
    pass

READER_MODE = 'gemini'
GROUNDING = 'span'
ALLOW_MISSING_VECTORS = False
ARMS = ['keyword', 'hybrid']
MODELS = list(DEFAULT_MODELS)

_last_live_call = [0.0]


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


def check_query_cache(questions: list[str]) -> list[str]:
    """Return the questions missing a cached ColQwen query vector."""
    from vvrag.retrieve.query_vectors import QueryVectorStore

    store = QueryVectorStore('data/query_cache')
    return [q for q in questions if store.get(q) is None]


def run_single(item, idx, arm, model, wdir, n_samples):
    """One row: reader pinned to `model`, retrieval `arm`. Quota -> rotate."""
    unanswerable = bool(item.get('category') == 'unanswerable')
    gpage = None if unanswerable else int(item['page'])
    models_tried: list[str] = []
    exceptions: list[dict] = []
    res, last_err, used_model = None, None, model
    order = [model] + [m for m in MODEL_CANDIDATES if m != model]
    for cand in order if READER_MODE == 'gemini' else [model]:
        used_model = cand
        try:
            res = paced_answer_question(
                item['question'], str(INDEX), str(wdir), reader_mode=READER_MODE,
                provider='gemini', model=used_model, n_samples=n_samples,
                grounding=GROUNDING, gate='calibrated',
                retrieval_mode=arm)  # full corpus: doc_id=None
            if used_model != model:
                exceptions.append({'model': used_model, 'reason': 'quota-rotation'})
            last_err = None
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            exceptions.append({'model': used_model,
                               'error': f'{type(e).__name__}: {str(e)[:120]}'})
            if _is_quota_error(e) and READER_MODE == 'gemini':
                print(f'    429/503 on {used_model}, rotating...')
                models_tried.append(used_model)
                continue
            print(f'    ERROR {type(e).__name__}: {str(e)[:100]}')
            break
    if last_err is not None or res is None:
        row = _base_row(item, idx, n_samples)
        row.update({'arm': arm, 'model': model, 'model_used': '',
                    'model_exceptions': exceptions,
                    'error': f'{type(last_err).__name__}: {last_err}'})
        return row

    ev = res.get('evidence', {})
    pred_boxes = [tuple(b) for b in ev.get('boxes', [])]
    ret_debug = res.get('retrieval_debug') or {}
    ranks = [(r.get('doc_id'), r.get('page')) for r in (res.get('retrieval') or [])]
    gold_rank = None if unanswerable else next(
        (i + 1 for i, (d, p) in enumerate(ranks)
         if d == item['doc_id'] and p == gpage), None)
    page_hit = None if unanswerable else bool(
        ev.get('doc_id') == item['doc_id'] and ev.get('page') == gpage)

    gwords = gbox = None
    gold_sids = []
    gold_words_all = []
    if not unanswerable:
        gold_words_all = page_words_sql(item['doc_id'], gpage)
        gwords = derive_answer_words(item['answer'], gold_words_all)[0]
        gbox = derive_gold_box(item['answer'], gold_words_all)
        gcon = sqlite3.connect(str(INDEX / 'meta.db'))
        gsents, _ = sentences_from_page(gcon, item['doc_id'], gpage)
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
    tier = res.get('tier', 'text' if res.get('verified') else 'none')
    verified = bool(res.get('verified'))
    golds = [item['answer'], *item.get('acceptable_answers', [])]
    if unanswerable:
        equiv = abstained  # correct probe behavior = abstain
        eq_f1 = None
        equiv_verified = abstained
    else:
        equiv_shown = (not abstained) and equivalent_correct_norm(answer, golds)
        equiv_verified = bool(equiv_shown and verified)
        equiv = equiv_shown  # legacy name: answer shown and correct (any tier)
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
        'arm': arm, 'model': model, 'model_used': used_model,
        'model_exceptions': exceptions,
        'answer': answer, 'abstained': abstained,
        'abstain_reason': res.get('abstain_reason', ''),
        'score': res.get('score'), 'tau': res.get('tau'),
        'verified': verified, 'tier': tier,
        'citations': res.get('citations'),
        'visual_confirm': res.get('visual_confirm'),
        'support': res.get('support'),
        'equiv_correct': bool(equiv),
        'equiv_verified': bool(equiv_verified),
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
    print(f'    [{arm}|{used_model}] ans={answer!r:.40} equiv={equiv} '
          f'ver_equiv={equiv_verified} tier={tier} page_hit={page_hit} '
          f'abst={abstained} quote={row["quote_ok"]} '
          f'mode={ret_debug.get("mode_used")}')
    return row


def _base_row(item, idx, n_samples):
    return {'idx': idx, 'qid': item['qid'], 'question': item['question'],
            'gold_answer': item['answer'],
            'acceptable_answers': item.get('acceptable_answers', []),
            'doc_id': item.get('doc_id'), 'gold_page': item.get('page'),
            'category': item.get('category', 'single-hop'),
            'probe_category': item.get('probe_category'),
            'unanswerable': item.get('category') == 'unanswerable',
            'n_samples': n_samples, 'grounding': GROUNDING}


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


def build_matrix(limit=0):
    """Pre-registered cell list: [(idx, item, arm, model, n_samples)].

    Keyword cells are built only when 'keyword' is in ARMS (or when --limit is
    used for smoke runs); so `--arms hybrid` runs exactly the hybrid arm.
    """
    gold, probes = load_items()
    hyb = []
    if HYBRID_ITEMS.exists():
        hyb = json.loads(HYBRID_ITEMS.read_text(encoding='utf-8'))
    hyb_ids = {h['qid'] for h in hyb}
    want_kw = 'keyword' in ARMS or bool(limit)
    cells = []
    idx = 0
    for item in gold:
        if limit and idx >= limit:
            break
        if want_kw:
            for model in MODELS:
                cells.append((idx, item, 'keyword', model,
                              3 if idx < 15 else 1))
        idx += 1
    if not limit:
        for item in probes:
            if want_kw:
                for model in MODELS:
                    cells.append((idx, item, 'keyword', model, 1))
            idx += 1
        if 'hybrid' in ARMS:
            for item in gold:
                if item['qid'] in hyb_ids:
                    cells.append((idx, item, 'hybrid', MODELS[0], 1))
                    idx += 1
    return cells


def summarize(rows, label='testRun8'):
    ok = [r for r in rows if 'error' not in r]
    errs = [r for r in rows if 'error' in r]
    print(f'\n=== {label} summary ===')
    print(f'rows={len(rows)} errors={len(errs)}')
    if errs:
        from collections import Counter
        print('  errored rows:',
              dict(Counter(f"{r.get('arm')}|{r.get('model')}" for r in errs)))

    out = {'rows': len(rows), 'errors': len(errs), 'cells': {}}
    for arm in ARMS:
        for model in MODELS:
            sub = [r for r in ok if r.get('arm') == arm
                   and r.get('model') == model]
            sub = [r for r in sub if r.get('model_used')]
            if not sub:
                continue
            gold = [r for r in sub if not r.get('unanswerable')]
            probes = [r for r in sub if r.get('unanswerable')]
            answered = [r for r in gold if not r.get('abstained')]
            cell = {}
            if gold:
                eq_shown = _mean([1.0 if r.get('equiv_correct') else 0.0
                                  for r in gold])
                eq_ver = _mean([1.0 if r.get('equiv_verified') else 0.0
                                for r in gold])
                false_abs = _mean([1.0 if r.get('abstained') else 0.0
                                   for r in gold])
                page_hit = _mean([bool(r.get('page_hit')) for r in gold])
                mrrs = [1.0 / r['gold_rank'] for r in gold if r.get('gold_rank')]
                mrr_v = (sum(mrrs) / len(mrrs)) if mrrs else None
                tiers = {}
                for r in answered:
                    t = r.get('tier') or 'none'
                    tiers[t] = tiers.get(t, 0) + 1
                cell.update(eq_shown=eq_shown, eq_verified=eq_ver,
                            false_abstention=false_abs, page_hit1=page_hit,
                            mrr=mrr_v, n_answered=len(answered), tiers=tiers,
                            quote_ok=_mean([bool(r.get('quote_ok'))
                                            for r in answered]),
                            sent_f1=_mean([r.get('sent_f1') for r in answered]),
                            iou=_mean([r.get('iou') for r in answered]))
            if probes:
                abs_ok = [1.0 if r.get('abstained') else 0.0 for r in probes]
                k = int(sum(abs_ok))
                lo, hi = clopper_pearson_interval(k, len(abs_ok))
                false_ans = 1.0 - (k / len(abs_ok))
                cell.update(probe_n=len(probes), probe_abstention=k / len(probes),
                            probe_cp95=[round(lo, 4), round(hi, 4)],
                            probe_false_answer_rate=false_ans)
                by_cat = {}
                for cat in sorted({r.get('probe_category') for r in probes
                                   if r.get('probe_category')}):
                    sub_p = [r for r in probes if r.get('probe_category') == cat]
                    kk = sum(1 for r in sub_p if r.get('abstained'))
                    l2, h2 = clopper_pearson_interval(kk, len(sub_p))
                    by_cat[cat] = {'n': len(sub_p), 'abstained': kk,
                                   'cp95': [round(l2, 4), round(h2, 4)]}
                cell['probe_by_category'] = by_cat
            out['cells'][f'{arm}|{model}'] = cell

            print(f'\n[{arm}|{model}] gold n={len(gold)} probes n={len(probes)}')
            if gold:
                print(f'  equiv_shown={_fmt(cell["eq_shown"])} '
                      f'equiv_verified(strict)={_fmt(cell["eq_verified"])} '
                      f'false_abstention={_fmt(cell["false_abstention"])} '
                      f'page_hit@1={_fmt(cell["page_hit1"])} '
                      f'MRR={_fmt(cell["mrr"])}')
                print(f'  answered={cell["n_answered"]} '
                      f'tiers={cell["tiers"]} quote_ok={_fmt(cell["quote_ok"])} '
                      f'sent_f1={_fmt(cell["sent_f1"])} iou={_fmt(cell["iou"])}')
                for cat in ('single-hop', 'multi-hop', 'image-only'):
                    s2 = [r for r in gold if r.get('category') == cat]
                    if s2:
                        print(f'    {cat:<11} n={len(s2):<3} '
                              f'shown={_fmt(_mean([1.0 if r.get("equiv_correct") else 0.0 for r in s2]))} '
                              f'strict={_fmt(_mean([1.0 if r.get("equiv_verified") else 0.0 for r in s2]))} '
                              f'false_abst={_fmt(_mean([1.0 if r.get("abstained") else 0.0 for r in s2]))}')
            if probes:
                print(f'  probes: abstention={cell["probe_abstention"]:.3f} '
                      f'CP95={cell["probe_cp95"]} '
                      f'false_answer={cell["probe_false_answer_rate"]:.3f}')
                for cat, d in cell['probe_by_category'].items():
                    print(f'    {cat:<17} abstained {d["abstained"]}/{d["n"]} '
                          f'CP95={d["cp95"]}')
    modes = {}
    for r in ok:
        m = (r.get('retrieval_debug') or {}).get('mode_used')
        modes[m] = modes.get(m, 0) + 1
    out['retrieval_modes_used'] = modes
    print(f'\nretrieval modes used: {modes}')
    return out


def preflight_hybrid():
    """Hard precondition: cached query vectors for every selected question."""
    if 'hybrid' not in ARMS or READER_MODE == 'mock':
        return
    hyb = json.loads(HYBRID_ITEMS.read_text(encoding='utf-8'))
    missing = check_query_cache([h['question'] for h in hyb])
    if missing and not ALLOW_MISSING_VECTORS:
        print(f'PRECONDITION FAILED: {len(missing)}/{len(hyb)} hybrid items have '
              f'no cached query vector in data/query_cache.')
        print('Run scripts/precompute_query_vectors.py on Colab first '
              '(see PIPELINE.md), or pass --allow-missing-vectors to fall back '
              'to keyword per row (recorded in retrieval.note).')
        for q in missing[:5]:
            print('  missing:', q[:90])
        raise SystemExit(2)
    if missing:
        print(f'WARNING: {len(missing)}/{len(hyb)} hybrid items lack cached '
              f'vectors; those rows will fall back to keyword.')


def build_rows(limit=0):
    cells = build_matrix(limit=limit)
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = REPO / 'results' / 'demo' / f'testrun8_{stamp}'
    out.mkdir(parents=True, exist_ok=True)
    rows = [dict(_base_row(item, i, ns), arm=arm, model=model, model_used='',
                 model_exceptions=[], error='PENDING: not run yet')
            for i, item, arm, model, ns in cells]
    (out / 'rows.json').write_text(json.dumps(rows, indent=2, default=str),
                                   encoding='utf-8')
    for k, (i, item, arm, model, ns) in enumerate(cells):
        print(f'--- [{k + 1}/{len(cells)}] {arm}|{model} idx={i} '
              f'n={ns} {item["qid"][:52]}')
        wdir = out / f'{k:03d}_{arm}'
        try:
            rows[k] = run_single(item, i, arm, model, wdir, ns)
        except Exception as e:  # noqa: BLE001
            print(f'    POST-ERROR {type(e).__name__}: {str(e)[:120]}')
            rows[k] = dict(_base_row(item, i, ns), arm=arm, model=model,
                           model_used='', model_exceptions=[],
                           error=f'{type(e).__name__}: {e}')
        (out / 'rows.json').write_text(
            json.dumps(rows, indent=2, default=str), encoding='utf-8')
    return rows, out


def resume(out: Path) -> None:
    rows = json.loads((out / 'rows.json').read_text(encoding='utf-8'))
    gold, probes = load_items()
    by_qid = {g['qid']: g for g in [*gold, *probes]}
    fixed = 0
    for k, row in enumerate(rows):
        if 'error' not in row:
            continue
        item = by_qid[row['qid']]
        wdir = out / f"{k:03d}_{row.get('arm', 'keyword')}"
        print(f'--- resume k={k} {row.get("arm")}|{row.get("model")} '
              f'{row["qid"][:46]} (was: {str(row["error"])[:60]})')
        try:
            rows[k] = run_single(item, row['idx'], row.get('arm', 'keyword'),
                                 row.get('model', MODELS[0]), wdir,
                                 row.get('n_samples', 1))
        except Exception as e:  # noqa: BLE001
            print(f'    POST-ERROR {type(e).__name__}: {str(e)[:120]}')
            rows[k] = dict(row, error=f'{type(e).__name__}: {e}')
        if 'error' not in rows[k]:
            fixed += 1
        (out / 'rows.json').write_text(
            json.dumps(rows, indent=2, default=str), encoding='utf-8')
    summary = summarize(rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=str),
                                      encoding='utf-8')
    n_err = sum(1 for r in rows if 'error' in r)
    print(f'\nRESUME DONE: fixed {fixed}, still errored {n_err}/{len(rows)}')


def main():
    global INDEX, GOLD_PATH, PROBE_PATH, READER_MODE, GROUNDING
    global ARMS, MODELS, ALLOW_MISSING_VECTORS
    ap = argparse.ArgumentParser(description='testRun8 arm x model matrix')
    ap.add_argument('--index', default=str(INDEX))
    ap.add_argument('--gold', default=str(GOLD_PATH))
    ap.add_argument('--probes', default=str(PROBE_PATH))
    ap.add_argument('--grounding', default='span', choices=['span', 'heatmap'])
    ap.add_argument('--arms', default='keyword,hybrid',
                    help='comma list from {keyword,hybrid}')
    ap.add_argument('--models', default=','.join(DEFAULT_MODELS),
                    help='comma list of pinned models (model A first)')
    ap.add_argument('--n3', type=int, default=15,
                    help='first N gold items use n_samples=3 (keyword arms)')
    ap.add_argument('--limit', type=int, default=0,
                    help='only first N gold items (probes/hybrid skipped)')
    ap.add_argument('--allow-missing-vectors', action='store_true',
                    help='hybrid cache miss -> keyword fallback per row')
    ap.add_argument('--check-cache', action='store_true',
                    help='print hybrid cache coverage and exit')
    ap.add_argument('--mock', action='store_true')
    args, rest = ap.parse_known_args(sys.argv[1:])

    if args.mock:
        READER_MODE = 'mock'
    GROUNDING = args.grounding
    ARMS = [a.strip() for a in args.arms.split(',') if a.strip()]
    MODELS = [m.strip() for m in args.models.split(',') if m.strip()]
    ALLOW_MISSING_VECTORS = args.allow_missing_vectors

    if rest and rest[0] == '--resume':
        INDEX = Path(args.index)
        GOLD_PATH = Path(args.gold)
        PROBE_PATH = Path(args.probes)
        resume(Path(rest[1]))
        return

    INDEX = Path(args.index)
    GOLD_PATH = Path(args.gold)
    PROBE_PATH = Path(args.probes)

    if args.check_cache:
        hyb = json.loads(HYBRID_ITEMS.read_text(encoding='utf-8'))
        missing = check_query_cache([h['question'] for h in hyb])
        print(f'hybrid cache coverage: {len(hyb) - len(missing)}/{len(hyb)}')
        for q in missing:
            print('  MISSING:', q[:90])
        raise SystemExit(1 if missing else 0)

    if not args.limit:
        preflight_hybrid()  # hybrid cells are only built in the full matrix
    rows, out = build_rows(limit=args.limit)
    (out / 'rows.json').write_text(json.dumps(rows, indent=2, default=str),
                                   encoding='utf-8')
    summary = summarize(rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=str),
                                      encoding='utf-8')
    print(f'\nWROTE {out / "rows.json"} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
