"""Build testRun8.pdf: arm x model matrix report (Run 8 plan Phases E/F).

testRun8 = SlideVQA index, 40 gold items + 41 verified-absent unanswerable
probes, run as an arm x model matrix:
  - keyword arm x 2 pinned models x all items (40 gold + 41 probes)
  - hybrid arm x model A x 24 selected items (results/run8_hybrid_items.json;
    query-vector cache precondition enforced by run_test8.py)
Per row: retrieval mode used, tier (text|visual|none), citation resolution,
visual confirmation, support veto, quote check, strict (text-verified) vs
any-tier equivalence, CP intervals on probe abstention by probe category.

Usage: python scripts/make_test8_pdf.py [rows_dir] [out_pdf]
Default rows_dir: newest results/demo/testrun8_*/.
"""
import datetime as dt
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_cands = sorted((REPO / 'results' / 'demo').glob('testrun8_*'),
                key=lambda p: p.stat().st_mtime)
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else (_cands[-1] if _cands else None)
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / 'testRun8.pdf'
INDEX = REPO / 'data' / 'index'
GOLD = REPO / 'data' / 'gold.json'
PROBES_JSON = REPO / 'data' / 'gold_unanswerable.json'

sys.path.insert(0, str(REPO / 'src'))

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image as RLImage, Paragraph, Spacer,
                                Table, TableStyle)

PAGE_W, PAGE_H = A4
styles = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=17, spaceAfter=6)
H2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=13, spaceAfter=4)
H3 = ParagraphStyle('H3', parent=styles['Heading3'], fontSize=10.5, spaceAfter=3,
                    spaceBefore=6)
BODY = ParagraphStyle('Body', parent=styles['Normal'], fontSize=8.5, leading=11)
SMALL = ParagraphStyle('Small', parent=styles['Normal'], fontSize=7.2, leading=9.5)
CELL = ParagraphStyle('Cell', parent=styles['Normal'], fontSize=6.3, leading=7.8)
CELLH = ParagraphStyle('CellH', parent=styles['Normal'], fontSize=6.3, leading=7.8,
                       textColor=colors.white)

_img_cache = REPO / 'results' / 'demo' / '_pdfimg'


def img(path, max_w=85 * mm, max_h=78 * mm):
    from PIL import Image as PILImage
    if not path or not Path(path).exists():
        return Paragraph('<i>[image missing]</i>', SMALL)
    p = Path(path)
    try:
        im = PILImage.open(p)
        if im.mode not in ('RGB', 'L'):
            im = im.convert('RGB')
        w, h = im.size
        pt_w, pt_h = w * 0.75, h * 0.75
        s = min(max_w / pt_w, max_h / pt_h, 1.0)
        if p.suffix.lower() != '.jpg' or s < 1.0 or w > 1100:
            _img_cache.mkdir(parents=True, exist_ok=True)
            key = f'{abs(hash((str(p), round(s, 3))))}.jpg'
            jp = _img_cache / key
            if not jp.exists():
                im2 = im.resize((max(1, int(w * s)), max(1, int(h * s))),
                                PILImage.LANCZOS)
                im2.save(jp, 'JPEG', quality=72, optimize=True)
            return RLImage(str(jp), width=pt_w * s, height=pt_h * s)
        return RLImage(str(p), width=pt_w * s, height=pt_h * s)
    except Exception:  # noqa: BLE001
        return Paragraph('<i>[image error]</i>', SMALL)


def table(data, widths=None, header=True):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
             ('VALIGN', (0, 0), (-1, -1), 'TOP'),
             ('FONTSIZE', (0, 0), (-1, -1), 6.5)]
    if header:
        style += [('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b5797')),
                  ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                  ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')]
    t.setStyle(TableStyle(style))
    return t


def P(text, style=BODY):
    return Paragraph(text, style)


def esc(s, n=None):
    s = '' if s is None else str(s)
    s = s.replace('<', '&lt;').replace('>', '&gt;')
    return s if n is None or len(s) <= n else s[:n - 1] + '\u2026'


def mm1(v):
    return 'n/a' if v is None else f'{v:.3f}'


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


if RUN is None or not (RUN / 'rows.json').exists():
    sys.exit('No testrun8 rows.json found — run scripts/run_test8.py first.')

rows = json.loads((RUN / 'rows.json').read_text(encoding='utf-8'))
golds = json.loads(GOLD.read_text(encoding='utf-8'))
probe_specs = (json.loads(PROBES_JSON.read_text(encoding='utf-8'))
               if PROBES_JSON.exists() else [])
try:
    tau_meta = json.loads(
        (REPO / 'data' / 'gate' / 'tau.json').read_text(encoding='utf-8'))
except Exception:  # noqa: BLE001
    tau_meta = {'tau': '?'}

con = sqlite3.connect(str(INDEX / 'meta.db'))
docs = {}
for doc, pg in con.execute(
        'SELECT doc_id, COUNT(*) FROM pages GROUP BY doc_id ORDER BY doc_id'):
    docs[doc] = {'pages': pg, 'words': 0, 'sentences': 0, 'embeds': 0}
for doc, nw in con.execute(
        'SELECT doc_id, COUNT(*) FROM word_boxes GROUP BY doc_id'):
    if doc in docs:
        docs[doc]['words'] = nw
for doc, ns in con.execute('SELECT doc_id, COUNT(*) FROM sentences GROUP BY doc_id'):
    if doc in docs:
        docs[doc]['sentences'] = ns
for doc in docs:
    docs[doc]['embeds'] = len(list((INDEX / 'embeddings').glob(f'{doc}_*.npy')))
n_pages = sum(d['pages'] for d in docs.values())
n_words = sum(d['words'] for d in docs.values())
n_sents = sum(d['sentences'] for d in docs.values())
n_embeds = sum(d['embeds'] for d in docs.values())
con.close()

# ---- cells ----
ok = [r for r in rows if 'error' not in r]
errs = [r for r in rows if 'error' in r]
cells = sorted({(r.get('arm'), r.get('model')) for r in ok})
cell_rows = {c: [r for r in ok if (r.get('arm'), r.get('model')) == c]
             for c in cells}
MODEL_A = cells[0][1] if cells else '?'

from vvrag.eval.metrics import clopper_pearson_interval  # noqa: E402


def cell_stats(rs):
    gold = [r for r in rs if not r.get('unanswerable')]
    probes = [r for r in rs if r.get('unanswerable')]
    ans = [r for r in gold if not r.get('abstained')]
    st = {'n_gold': len(gold), 'n_probes': len(probes), 'n_ans': len(ans)}
    st['shown'] = _mean([1.0 if r.get('equiv_correct') else 0.0 for r in gold])
    st['strict'] = _mean([1.0 if r.get('equiv_verified') else 0.0 for r in gold])
    st['false_abs'] = _mean([1.0 if r.get('abstained') else 0.0 for r in gold])
    st['page_hit'] = _mean([bool(r.get('page_hit')) for r in gold])
    ranks = [r['gold_rank'] for r in gold if r.get('gold_rank')]
    st['mrr'] = (sum(1.0 / r for r in ranks) / len(ranks)) if ranks else None
    st['quote_ok'] = _mean([bool(r.get('quote_ok')) for r in ans])
    st['tiers'] = dict(Counter(r.get('tier') or 'none' for r in ans))
    st['sent_f1'] = _mean([r.get('sent_f1') for r in ans])
    st['n_sent'] = sum(1 for r in ans if r.get('sent_f1') is not None)
    st['iou'] = _mean([r.get('iou') for r in ans])
    st['n_box'] = sum(1 for r in ans if r.get('iou') is not None)
    st['visual_wrong'] = sum(1 for r in ans
                             if r.get('tier') == 'visual'
                             and not r.get('equiv_correct'))
    st['shown_wrong_unverified'] = sum(
        1 for r in ans if not r.get('equiv_correct') and not r.get('equiv_verified'))
    if probes:
        k = sum(1 for r in probes if r.get('abstained'))
        st['probe_abs'] = k / len(probes)
        st['probe_cp'] = list(clopper_pearson_interval(k, len(probes)))
        st['probe_false'] = len(probes) - k
    return st


stats = {c: cell_stats(rs) for c, rs in cell_rows.items()}
arm_model_names = [f'{a}|{m}' for a, m in cells]
modes_used = dict(Counter((r.get('retrieval_debug') or {}).get('mode_used')
                          for r in ok))
model_used_hist = dict(Counter(r.get('model_used') or '?' for r in ok))
n_exceptions = sum(1 for r in ok if r.get('model_exceptions'))

kw_a = [r for r in ok if r.get('arm') == 'keyword' and r.get('model') == MODEL_A]
hyb = [r for r in ok if r.get('arm') == 'hybrid']
n3_rows = [r for r in kw_a if (r.get('n_samples') or 1) > 1]

# citations / support / visual diagnostics (keyword arms)
cit_hist = dict(Counter((r.get('citations') or {}).get('status')
                        for r in ok if not r.get('unanswerable')))
sup_hist = dict(Counter((r.get('support') or {}).get('method')
                        for r in ok if not r.get('abstained')
                        and not r.get('unanswerable')))
vis_attempts = [r for r in ok if (r.get('visual_confirm') or {}).get('attempted')]
vis_verdicts = dict(Counter((r.get('visual_confirm') or {}).get('verdict')
                            for r in vis_attempts))


def gold_overlay(r):
    """Render (once, cached) the gold box on the gold slide for row r."""
    if not r.get('gold_box') or r.get('gold_page') is None:
        return None
    out = _img_cache / f"gold_{RUN.name}_{r['qid']}.jpg"
    if out.exists():
        return str(out)
    try:
        import numpy as np
        from vvrag.ground.crop import overlay_heatmap

        con = sqlite3.connect(str(INDEX / 'meta.db'))
        row = con.execute(
            'SELECT image_path FROM pages WHERE doc_id=? AND page=?',
            (r['doc_id'], r['gold_page'])).fetchone()
        con.close()
        if not row or not Path(row[0]).exists():
            return None
        _img_cache.mkdir(parents=True, exist_ok=True)
        overlay_heatmap(row[0], np.zeros((8, 6), dtype=np.float32),
                        [list(r['gold_box'])], out)
        return str(out)
    except Exception:  # noqa: BLE001
        return None


def row_by(qid, arm, model):
    return next((r for r in ok if r.get('qid') == qid and r.get('arm') == arm
                 and r.get('model') == model), None)


story = []
story.append(P('Test Run 8 — arm x model matrix: strict verification, visual '
               'tier, hybrid retrieval, 41 unanswerable probes', H1))
_n_hyb_sel = len([r for r in hyb if r.get('unanswerable') is False])
_hyb_desc = (f"<b>hybrid arm x model A x {_n_hyb_sel} selected items</b>"
             if hyb else
             "<b>hybrid arm x model A x 24 selected items — NOT RUN</b> "
             "(query-cache precondition: scripts/precompute_query_vectors.py "
             "on Colab, then run_test8.py --arms hybrid)")
story.append(P(
    f"Date: <b>{dt.datetime.now().strftime('%Y-%m-%d %H:%M')}</b>. Index: "
    f"<b>{len(docs)} SlideVQA decks, {n_pages} pages, {n_words:,} OCR word "
    f"boxes, {n_sents:,} sentences, {n_embeds} ColQwen page vectors</b>. "
    f"Matrix: <b>keyword arm x {len([m for a, m in cells if a == 'keyword'])} "
    f"pinned models x {len(kw_a)} items (40 gold + 41 probes)</b> plus "
    f"{_hyb_desc} = <b>{len(rows)} rows</b>, grounding=<b>span</b>, "
    f"gate=<b>calibrated</b> (tau={tau_meta.get('tau')}) with hard support "
    f"veto, retrieval modes used: {modes_used}. Row-level result: "
    f"<b>{RUN.name}/rows.json</b> ({len(rows)} rows, {len(errs)} errors).",
    BODY))
story.append(Spacer(1, 4))

story.append(P('1. What testRun8 adds over testRun7', H2))
story.append(P(
    '1. <b>Citation-resolution honesty</b>: reader refs that do not resolve on '
    'the shown pages are reported and veto the answer (no silent rebinding to '
    'the primary page); per-row <i>citations.status</i> = '
    'none|ok|partial|unresolved.<br/>'
    '2. <b>Question-aware verification</b>: answer type must match the question '
    'kind (money/date/number/count/entity) and comparison questions require '
    'both operands in the cited text, before/independent of NLI '
    '(containment-first; NLI is the fallback).<br/>'
    '3. <b>Visual middle tier</b>: when the reader reports a visual region '
    '(chart / image answer) and text verification fails, a cross-family model '
    '(Groq) checks the region crop against the answer; agreement over n=3 '
    'samples is the fallback. Accepted rows carry tier=<b>visual</b> and are '
    'never counted as text-verified.<br/>'
    '4. <b>Real hybrid retrieval arm</b>: keyword rank RRF-fused with ColQwen '
    'visual page scores on 24 pre-selected items (15 retrieval-miss + 7 '
    'visual-abstain + 2 multi-hop). The arm refuses to start without cached '
    'query vectors (precondition check), so a hybrid row that ran really used '
    'visual scores.<br/>'
    '5. <b>Two pinned reader models</b> x full matrix: every item is answered '
    'by both models, isolating model variance from system behaviour; quota '
    'exceptions are recorded per row.<br/>'
    '6. <b>41 probes in 4 categories</b> (absent-attribute, near-miss, '
    'entity-swap, false-bridge) with Clopper-Pearson 95% intervals per '
    'category, plus relabeled gold (answer_in_ocr / visual-only / bridged) '
    'with acceptable-page tolerance.', BODY))
story.append(Spacer(1, 3))

story.append(P('2. Corpus and index', H2))
story.append(table(
    [[P('<b>deck (doc_id)</b>', CELLH), P('<b>pages</b>', CELLH),
      P('<b>OCR words</b>', CELLH), P('<b>sentences</b>', CELLH),
      P('<b>page vectors</b>', CELLH)]] +
    [[P(esc(d, 52), CELL), P(str(v['pages']), CELL), P(str(v['words']), CELL),
      P(str(v['sentences']), CELL), P(str(v['embeds']), CELL)]
     for d, v in sorted(docs.items())] +
    [[P('<b>TOTAL</b>', CELL), P(f'<b>{n_pages}</b>', CELL),
      P(f'<b>{n_words:,}</b>', CELL), P(f'<b>{n_sents:,}</b>', CELL),
      P(f'<b>{n_embeds}</b>', CELL)]],
    widths=[78 * mm, 16 * mm, 24 * mm, 24 * mm, 24 * mm]))
story.append(P('Retrieval is full-corpus (doc_id=None, the real UI path). The '
               + ('hybrid arm required cached query multivectors for all 24 '
                  'selected questions (scripts/precompute_query_vectors.py); '
                  'any keyword fallback inside the hybrid arm would be '
                  'visible in the mode histogram above. '
                  if hyb else
                  'hybrid arm not run in this pass: it requires cached query '
                  'multivectors for all 24 selected questions '
                  '(scripts/precompute_query_vectors.py on Colab; the '
                  'harness refuses to run the arm without them). ')
               + 'Experiments were run with pinned models '
               '(config run.reader_model_pinned / --models).', SMALL))
story.append(Spacer(1, 3))

story.append(P('3. Gold set (40 items) + unanswerable probes (41)', H2))
kw_models = [m for a, m in cells if a == 'keyword']
hdr = [P('<b>#</b>', CELLH), P('<b>question</b>', CELLH),
       P('<b>gold</b>', CELLH), P('<b>cat</b>', CELLH), P('<b>p</b>', CELLH)]
for i, m in enumerate(kw_models):
    hdr += [P(f'<b>M{i + 1} shown</b>', CELLH), P(f'<b>M{i + 1} strict</b>', CELLH)]
if hyb:
    hdr += [P('<b>hyb shown</b>', CELLH), P('<b>hyb strict</b>', CELLH)]
gdata = [hdr]
for gi, g in enumerate(golds):
    row = [P(str(gi + 1), CELL), P(esc(g['question'], 80), CELL),
           P(f"<b>{esc(g['answer'], 24)}</b>", CELL),
           P(esc(g.get('category', '?'), 10), CELL), P(str(g['page']), CELL)]
    for m in kw_models:
        rr = row_by(g['qid'], 'keyword', m)
        if rr is None or 'error' in rr:
            row += [P('—', CELL), P('—', CELL)]
        else:
            row += [P('ABS' if rr.get('abstained')
                      else ('Y' if rr.get('equiv_correct') else 'N'), CELL),
                    P('ABS' if rr.get('abstained')
                      else ('Y' if rr.get('equiv_verified') else 'N'), CELL)]
    if hyb:
        rr = row_by(g['qid'], 'hybrid', MODEL_A)
        if rr is None:
            row += [P('—', CELL), P('—', CELL)]
        else:
            row += [P('ABS' if rr.get('abstained')
                      else ('Y' if rr.get('equiv_correct') else 'N'), CELL),
                    P('ABS' if rr.get('abstained')
                      else ('Y' if rr.get('equiv_verified') else 'N'), CELL)]
    gdata.append(row)
w = [6 * mm, 62 * mm, 26 * mm, 15 * mm, 6 * mm]
w += [11 * mm, 11 * mm] * len(kw_models)
if hyb:
    w += [11 * mm, 11 * mm]
story.append(table(gdata, widths=w))
story.append(P('shown = answer displayed and correct (any tier); strict = answer '
               'displayed, correct AND text-verified (quote ok + supported + '
               'resolved citations); ABS = abstained. Gold flags: '
               + ', '.join(f"{g['qid'][:14]}:{'OCR' if g.get('answer_in_ocr') else 'visual'}"
                           for g in golds[:6]) + ' …', SMALL))
story.append(Spacer(1, 2))
pdata = [[P('<b>probe</b>', CELLH), P('<b>category</b>', CELLH),
          P('<b>question</b>', CELLH), P('<b>check tokens (absent)</b>', CELLH)]
         + [P(f'<b>M{i + 1}</b>', CELLH) for i, m in enumerate(kw_models)]]
for r in sorted([r for r in ok if r.get('unanswerable')], key=lambda r: r['idx']):
    spec = next((p for p in probe_specs if p['qid'] == r['qid']), {})
    answers = []
    for m in kw_models:
        rr = row_by(r['qid'], 'keyword', m)
        answers.append(P('Y' if rr and rr.get('abstained')
                         else f"<b>N: {esc((rr or {}).get('answer'), 18)}</b>", CELL))
    pdata.append([P(esc(r['qid'], 10), CELL),
                  P(esc(spec.get('probe_category', r.get('probe_category')), 17), CELL),
                  P(esc(r['question'], 64), CELL),
                  P(esc(', '.join(spec.get('check_absent', [])), 30), CELL)]
                 + answers)
story.append(table(pdata, widths=[10 * mm, 22 * mm, 70 * mm, 32 * mm]
                   + [18 * mm] * len(kw_models)))
story.append(Spacer(1, 3))

story.append(P('4. Headline metrics by cell', H2))
hdata = [[P('<b>cell (arm|model)</b>', CELLH), P('<b>gold n</b>', CELLH),
          P('<b>equiv shown</b>', CELLH), P('<b>equiv strict</b>', CELLH),
          P('<b>false abst.</b>', CELLH), P('<b>page_hit@1</b>', CELLH),
          P('<b>MRR</b>', CELLH), P('<b>answered</b>', CELLH),
          P('<b>tiers</b>', CELLH), P('<b>quote_ok</b>', CELLH),
          P('<b>sF1/IoU (n)</b>', CELLH), P('<b>probe abs.</b>', CELLH),
          P('<b>CP95</b>', CELLH), P('<b>vis-wrong</b>', CELLH)]]
for c in cells:
    st = stats[c]
    cp = st.get('probe_cp')
    hdata.append([
        P(f'{c[0]}|{esc(c[1], 24)}', CELL), P(str(st['n_gold']), CELL),
        P(f"<b>{mm1(st['shown'])}</b>", CELL),
        P(f"<b>{mm1(st['strict'])}</b>", CELL),
        P(mm1(st['false_abs']), CELL), P(mm1(st['page_hit']), CELL),
        P(mm1(st['mrr']), CELL), P(str(st['n_ans']), CELL),
        P(esc(str(st['tiers']), 34), CELL), P(mm1(st['quote_ok']), CELL),
        P(f"{mm1(st['sent_f1'])}/{mm1(st['iou'])} "
          f"({st['n_sent']}/{st['n_box']})", CELL),
        P(mm1(st.get('probe_abs')), CELL),
        P(f"[{mm1(cp[0])}, {mm1(cp[1])}]" if cp else '—', CELL),
        P(str(st['visual_wrong']), CELL)])
story.append(table(hdata, widths=[30 * mm, 10 * mm, 14 * mm, 14 * mm, 13 * mm,
                                  13 * mm, 10 * mm, 12 * mm, 26 * mm, 12 * mm,
                                  22 * mm, 13 * mm, 22 * mm, 12 * mm]))
story.append(P('strict = headline metric (text-verified only): '
               'equiv_verified = shown AND quote_ok AND supported AND citations '
               'resolved. visual-wrong = rows accepted via the visual tier that '
               'are nonetheless incorrect; shown-wrong-unverified (any cell) = '
               + str({f'{a}|{m}': stats[(a, m)]['shown_wrong_unverified']
                      for a, m in cells}) + '.', SMALL))
story.append(Spacer(1, 2))
story.append(P('By category and probe type:', H3))
cdata = [[P('<b>category</b>', CELLH)] + [P(f'<b>{a}|{m}: n / shown / strict</b>', CELLH)
                                          for a, m in cells]]
for cat in ('single-hop', 'multi-hop', 'image-only'):
    row = [P(cat, CELL)]
    for c in cells:
        sub = [r for r in cell_rows[c] if r.get('category') == cat]
        ans = [r for r in sub if not r.get('abstained')]
        row.append(P(f"{len(sub)} / "
                     f"{sum(1 for r in sub if r.get('equiv_correct'))} / "
                     f"{sum(1 for r in sub if r.get('equiv_verified'))} "
                     f"(abs {len(sub) - len(ans)})", CELL))
    cdata.append(row)
story.append(table(cdata, widths=[26 * mm] + [54 * mm] * len(cells)))
pcats = {}
for r in ok:
    if r.get('unanswerable'):
        pcats.setdefault(r.get('probe_category') or '?', []).append(r)
pdata2 = [[P('<b>probe category</b>', CELLH)]
          + [P(f'<b>{a}|{m}: abstained/n</b>', CELLH) for a, m in cells]
          + [P('<b>CP95 (model A)</b>', CELLH)]]
for cat in sorted(pcats):
    row = [P(cat, CELL)]
    for c in cells:
        sub = [r for r in pcats[cat]
               if (r.get('arm'), r.get('model')) == c and r.get('model_used')]
        k = sum(1 for r in sub if r.get('abstained'))
        row.append(P(f'{k}/{len(sub)}', CELL))
    sub_a = [r for r in pcats[cat]
             if (r.get('arm'), r.get('model')) == cells[0]]
    k = sum(1 for r in sub_a if r.get('abstained'))
    lo, hi = clopper_pearson_interval(k, len(sub_a)) if sub_a else (0.0, 0.0)
    row.append(P(f'[{lo:.2f}, {hi:.2f}]', CELL))
    pdata2.append(row)
story.append(table(pdata2, widths=[34 * mm] + [44 * mm] * len(cells) + [30 * mm]))
story.append(Spacer(1, 3))

story.append(P('5. Verification diagnostics (keyword arm, gold rows)', H2))
story.append(P(f"Citations status histogram: <b>{cit_hist}</b> "
               f"(none = reader cited nothing; unresolved/partial = vetoed "
               f"unless visual tier rescued). Support methods on accepted "
               f"answers: <b>{sup_hist}</b>. Visual tier attempts: "
               f"<b>{len(vis_attempts)}</b> verdicts={vis_verdicts}.", BODY))
for r0 in [r for r in ok if r.get('visual_confirm', {}).get('attempted')][:14]:
    vc = r0['visual_confirm']
    story.append(P(f"[{r0['arm']}|{esc(r0.get('model'), 20)}] {esc(r0['question'], 90)}"
                   f" — reader: {esc(r0.get('answer'), 30)} | region "
                   f"{'yes' if (r0.get('samples') or [{}])[0].get('region') else 'no'}"
                   f" | verdict <b>{esc(vc.get('verdict'), 12)}</b> "
                   f"({esc(vc.get('method'), 16)}) | detail {esc(vc.get('detail'), 80)}",
                   SMALL))
story.append(Spacer(1, 3))

if hyb:
    story.append(P('6. Keyword vs hybrid (24 selected items, model A)', H2))
    hdata = [[P('<b>#</b>', CELLH), P('<b>question</b>', CELLH),
              P('<b>reason</b>', CELLH), P('<b>kw rank</b>', CELLH),
              P('<b>hyb rank</b>', CELLH), P('<b>kw shown/strict</b>', CELLH),
              P('<b>hyb shown/strict</b>', CELLH), P('<b>hyb tier</b>', CELLH),
              P('<b>hyb mode</b>', CELLH)]]
    kw_ph = hy_ph = 0
    for gi, g in enumerate(golds):
        rk = row_by(g['qid'], 'keyword', MODEL_A)
        rh = row_by(g['qid'], 'hybrid', MODEL_A)
        if rh is None:
            continue
        hyb_item = json.loads((REPO / 'results' / 'run8_hybrid_items.json')
                              .read_text(encoding='utf-8'))
        reason = next((h.get('reason', '') for h in hyb_item
                       if h['qid'] == g['qid']), '')
        kw_ph += 1 if rk and rk.get('page_hit') else 0
        hy_ph += 1 if rh.get('page_hit') else 0
        hdata.append([
            P(str(gi + 1), CELL), P(esc(g['question'], 66), CELL),
            P(esc(reason, 16), CELL),
            P(str((rk or {}).get('gold_rank') or '>5'), CELL),
            P(str(rh.get('gold_rank') or '>5'), CELL),
            P(('ABS' if (rk or {}).get('abstained') else
               ('Y' if (rk or {}).get('equiv_correct') else 'N')) + ' / ' +
              ('ABS' if (rk or {}).get('abstained') else
               ('Y' if (rk or {}).get('equiv_verified') else 'N')), CELL),
            P(('ABS' if rh.get('abstained') else
               ('Y' if rh.get('equiv_correct') else 'N')) + ' / ' +
              ('ABS' if rh.get('abstained') else
               ('Y' if rh.get('equiv_verified') else 'N')), CELL),
            P(esc(rh.get('tier'), 8), CELL),
            P(esc((rh.get('retrieval_debug') or {}).get('mode_used'), 10), CELL)])
    story.append(table(hdata, widths=[6 * mm, 58 * mm, 16 * mm, 12 * mm, 12 * mm,
                                      18 * mm, 18 * mm, 12 * mm, 14 * mm]))
    story.append(P(f'page_hit on the 24 selected items: keyword <b>{kw_ph}/24</b> '
                   f'vs hybrid <b>{hy_ph}/24</b>. Selection rule '
                   f'(scripts/select_hybrid_items.py): 15 retrieval-miss + 7 '
                   f'Run-7 visual-abstain + 2 multi-hop fill.', BODY))
    story.append(Spacer(1, 3))
else:
    story.append(P('6. Keyword vs hybrid (24 selected items, model A)', H2))
    story.append(P('The hybrid arm was <b>not run</b> in this pass: the harness '
                   'refuses to start it without cached ColQwen query '
                   'multivectors for all 24 selected questions '
                   '(scripts/precompute_query_vectors.py, Colab). The table '
                   'below shows the pre-registered selection and the keyword '
                   'baseline for each selected item; re-run '
                   '<i>python scripts/run_test8.py --arms hybrid</i> after '
                   'caching the vectors to fill the hybrid columns.', BODY))
    _sel = json.loads((REPO / 'results' / 'run8_hybrid_items.json')
                      .read_text(encoding='utf-8'))
    _sel_ids = {h['qid'] for h in _sel}
    hdata = [[P('<b>#</b>', CELLH), P('<b>question</b>', CELLH),
              P('<b>reason</b>', CELLH), P('<b>kw rank</b>', CELLH),
              P('<b>kw shown/strict</b>', CELLH)]]
    for gi, g in enumerate(golds):
        if g['qid'] not in _sel_ids:
            continue
        rk = row_by(g['qid'], 'keyword', MODEL_A)
        reason = next((h.get('reason', '') for h in _sel
                       if h['qid'] == g['qid']), '')
        hdata.append([
            P(str(gi + 1), CELL), P(esc(g['question'], 70), CELL),
            P(esc(reason, 16), CELL),
            P(str((rk or {}).get('gold_rank') or '>5'), CELL),
            P(('ABS' if (rk or {}).get('abstained') else
               ('Y' if (rk or {}).get('equiv_correct') else 'N')) + ' / ' +
              ('ABS' if (rk or {}).get('abstained') else
               ('Y' if (rk or {}).get('equiv_verified') else 'N')), CELL)])
    story.append(table(hdata, widths=[6 * mm, 92 * mm, 20 * mm, 14 * mm,
                                      24 * mm]))
    story.append(Spacer(1, 3))

story.append(P('7. Reader-model variance (keyword arm)', H2))
disc = []
for g in golds:
    ra = row_by(g['qid'], 'keyword', kw_models[0])
    rb = row_by(g['qid'], 'keyword', kw_models[1]) if len(kw_models) > 1 else None
    if not ra or not rb:
        continue
    va = 'ABS' if ra.get('abstained') else f"{'OK' if ra.get('equiv_verified') else ('Y' if ra.get('equiv_correct') else 'N')}:{ra.get('answer')}"
    vb = 'ABS' if rb.get('abstained') else f"{'OK' if rb.get('equiv_verified') else ('Y' if rb.get('equiv_correct') else 'N')}:{rb.get('answer')}"
    if (ra.get('abstained') != rb.get('abstained')
            or (not ra.get('abstained') and str(ra.get('answer')).strip().lower()
                != str(rb.get('answer')).strip().lower())):
        disc.append((g, ra, rb, va, vb))
story.append(P(f'Discordant gold items (one model answers / the other abstains, '
               f'or different answers): <b>{len(disc)}/{len(golds)}</b> of the '
               f'40. Rows below are evidence that single-model comparisons mix '
               f'model and system behaviour.', BODY))
if disc:
    ddata = [[P('<b>#</b>', CELLH), P('<b>question</b>', CELLH),
              P(f'<b>{esc(kw_models[0], 20)}</b>', CELLH),
              P(f'<b>{esc(kw_models[1], 20)}</b>', CELLH),
              P('<b>gold</b>', CELLH)]]
    for g, ra, rb, va, vb in disc:
        ddata.append([P(esc(g['qid'][-6:], 8), CELL), P(esc(g['question'], 70), CELL),
                      P(esc(va, 40), CELL), P(esc(vb, 40), CELL),
                      P(esc(g['answer'], 18), CELL)])
    story.append(table(ddata, widths=[12 * mm, 62 * mm, 38 * mm, 38 * mm, 18 * mm]))
story.append(Spacer(1, 3))

story.append(P('8. Per-item results (keyword arm, model A)', H2))
story.append(P('gold overlay = derived gold box on the gold slide; crop = '
               'reader input on the evidence page; overlay = final evidence '
               'boxes. A compact table lists the other cells for the same item.',
               BODY))
for r0 in sorted([r for r in kw_a if not r.get('unanswerable')],
                 key=lambda r: r['idx']):
    gi = next((i for i, g in enumerate(golds) if g['qid'] == r0['qid']), None)
    ci = (gi + 1) if gi is not None else '?'
    story.append(P(f"[{ci}] {esc(r0['question'], 150)}", H3))
    story.append(P(f"Gold: <b>{esc(r0['gold_answer'], 40)}</b> "
                   f"(deck {esc(str(r0['doc_id']).split('-')[0][:22])}, page "
                   f"{r0['gold_page']}, cat {esc(r0.get('category'), 11)}, "
                   f"n={r0.get('n_samples')})", SMALL))
    loc = r0.get('localize', {}) or {}
    samples = r0.get('samples') or [{}]
    quote = samples[0].get('quote', '') or ''
    cited = r0.get('cited_sentences') or []
    cited_txt = '; '.join(
        f"#{c.get('sent_id')} p{c.get('page', '?')} "
        f"\u201c{esc(c.get('text'), 55)}\u201d" for c in cited) if cited else '(none)'
    sup = r0.get('support') or {}
    story.append(P(
        f"ans: <b>{esc(r0.get('answer'), 34)}</b>"
        f"{' (ABSTAIN: ' + esc(r0.get('abstain_reason', ''), 50) + ')' if r0.get('abstained') else ''}"
        f" | shown {r0.get('equiv_correct')} strict {r0.get('equiv_verified')}"
        f" | tier {r0.get('tier')}"
        f" | page_hit {r0.get('page_hit')} rank {r0.get('gold_rank') or '>5'}"
        f" | quote_ok {r0.get('quote_ok')} (q={loc.get('quote_score', '')})"
        f" | verified {r0.get('verified')} (support {esc(sup.get('method'), 12)})"
        f" | cit {(r0.get('citations') or {}).get('status')}"
        f" | sF1 {mm1(r0.get('sent_f1'))} IoU {mm1(r0.get('iou'))}"
        f" | score {r0.get('score')} / tau {r0.get('tau')}", SMALL))
    story.append(P(f"cited: {cited_txt} | quote: \u201c{esc(quote, 90)}\u201d", SMALL))
    if r0.get('abstained') and r0.get('closest_passages'):
        story.append(P('closest passages: '
                       + '; '.join(f"\u201c{esc(p.get('text'), 70)}\u201d"
                                   for p in (r0['closest_passages'] or [])[:3]),
                       SMALL))
    others = [r for r in ok if r.get('qid') == r0['qid']
              and not (r.get('arm') == 'keyword' and r.get('model') == MODEL_A)]
    if others:
        odata = [[P('<b>cell</b>', CELLH), P('<b>answer</b>', CELLH),
                  P('<b>shown/strict</b>', CELLH), P('<b>tier</b>', CELLH),
                  P('<b>abstain reason / support</b>', CELLH)]]
        for r_ in others:
            tag = 'ABS' if r_.get('abstained') else (
                f"{'Y' if r_.get('equiv_correct') else 'N'}/"
                f"{'Y' if r_.get('equiv_verified') else 'N'}")
            note = esc(r_.get('abstain_reason') or
                       f"support {(r_.get('support') or {}).get('method')}", 60)
            odata.append([P(f"{r_.get('arm')}|{esc(r_.get('model'), 20)}", CELL),
                          P(esc(r_.get('answer'), 26), CELL), P(tag, CELL),
                          P(esc(r_.get('tier'), 8), CELL), P(note, CELL)])
        story.append(table(odata, widths=[36 * mm, 40 * mm, 18 * mm, 12 * mm,
                                          60 * mm]))
    story.append(table([
        [P('<b>gold overlay (gold box)</b>', CELLH),
         P('<b>crop (reader input)</b>', CELLH),
         P('<b>overlay (evidence boxes)</b>', CELLH)],
        [img(gold_overlay(r0), 58 * mm, 56 * mm),
         img(r0.get('crop'), 58 * mm, 56 * mm),
         img(r0.get('overlay'), 58 * mm, 56 * mm)],
    ], widths=[58.5 * mm, 58.5 * mm, 58.5 * mm]))
    story.append(Spacer(1, 3))

if hyb:
    story.append(P('9. Hybrid arm detail (24 selected items, model A)', H2))
    for r0 in sorted(hyb, key=lambda r: r['idx']):
        story.append(P(f"[{esc(r0['qid'][-10:], 12)}] {esc(r0['question'], 140)}", H3))
        loc = r0.get('localize', {}) or {}
        story.append(P(
            f"Gold: <b>{esc(r0['gold_answer'], 30)}</b> | ans: "
            f"<b>{esc(r0.get('answer'), 30)}</b>"
            f"{' (ABSTAIN: ' + esc(r0.get('abstain_reason', ''), 44) + ')' if r0.get('abstained') else ''}"
            f" | shown {r0.get('equiv_correct')} strict {r0.get('equiv_verified')}"
            f" | page_hit {r0.get('page_hit')} rank {r0.get('gold_rank') or '>5'}"
            f" | tier {r0.get('tier')} | quote_ok {r0.get('quote_ok')}"
            f" | mode {(r0.get('retrieval_debug') or {}).get('mode_used')}"
            f" | note {esc((r0.get('retrieval_debug') or {}).get('note'), 40)}",
            SMALL))
        story.append(table([
            [P('<b>crop</b>', CELLH), P('<b>overlay</b>', CELLH)],
            [img(r0.get('crop'), 88 * mm, 60 * mm),
             img(r0.get('overlay'), 88 * mm, 60 * mm)],
        ], widths=[88.5 * mm, 88.5 * mm]))
        story.append(Spacer(1, 2))

_sn = 10 if hyb else 9
story.append(P(f'{_sn}. Probes (detail, keyword arm model A)', H2))
for r0 in sorted([r for r in kw_a if r.get('unanswerable')],
                 key=lambda r: r['idx']):
    spec = next((p for p in probe_specs if p['qid'] == r0['qid']), {})
    verdicts = []
    for m in kw_models:
        rr = row_by(r0['qid'], 'keyword', m)
        verdicts.append(f"{m}: "
                        + ('ABSTAIN' if rr and rr.get('abstained')
                           else f"ANSWER({esc((rr or {}).get('answer'), 18)})"))
    story.append(P(f"[{esc(r0['qid'], 12)}] {esc(r0['question'], 130)}", H3))
    story.append(P(f"category {esc(spec.get('probe_category'), 17)} | check "
                   f"tokens: {esc(', '.join(spec.get('check_absent', [])), 40)}"
                   f" | reason {esc(r0.get('abstain_reason'), 55)} | score "
                   f"{r0.get('score')} / tau {r0.get('tau')} | "
                   + ' | '.join(verdicts), SMALL))
    if r0.get('crop') or r0.get('overlay'):
        story.append(table([
            [P('<b>crop (top-ranked page)</b>', CELLH),
             P('<b>overlay</b>', CELLH)],
            [img(r0.get('crop'), 88 * mm, 60 * mm),
             img(r0.get('overlay'), 88 * mm, 60 * mm)],
        ], widths=[88.5 * mm, 88.5 * mm]))
    story.append(Spacer(1, 2))

story.append(P(f'{_sn + 1}. Full result table', H2))
hdr = ['k', 'qid', 'arm', 'model', 'cat', 'answer', 'shown', 'strict', 'tier',
       'pg1', 'rank', 'qok', 'ver', 'cit', 'score', 'verdict']
tdata = [[P(f'<b>{h}</b>', CELLH) for h in hdr]]
for k, r in enumerate(rows):
    if 'error' in r:
        tdata.append([P(str(k), CELL), P(esc(r.get('qid'), 14), CELL),
                      P(esc(r.get('arm'), 8), CELL),
                      P(esc(r.get('model'), 20), CELL)]
                     + [P('ERROR: ' + esc(r.get('error'), 50), CELL)]
                     + [P('', CELL) for _ in range(11)])
        continue
    tdata.append([
        P(str(k), CELL), P(esc(r.get('qid'), 14), CELL),
        P(esc(r.get('arm'), 8), CELL), P(esc(r.get('model'), 20), CELL),
        P(esc(r.get('category') or r.get('probe_category'), 11), CELL),
        P(esc(r.get('answer'), 16), CELL),
        P('Y' if r.get('equiv_correct') else ('N' if not r.get('unanswerable') else '<b>N</b>'), CELL),
        P('Y' if r.get('equiv_verified') else 'N', CELL),
        P(esc(r.get('tier'), 6), CELL),
        P('Y' if r.get('page_hit') else ('N' if r.get('page_hit') is not None else '—'), CELL),
        P(str(r.get('gold_rank') or ('—' if r.get('unanswerable') else '>5')), CELL),
        P('Y' if r.get('quote_ok') else ('N' if r.get('quote_ok') is not None else '—'), CELL),
        P('Y' if r.get('verified') else 'N', CELL),
        P(esc((r.get('citations') or {}).get('status'), 5), CELL),
        P(str(r.get('score')), CELL),
        P('ABS' if r.get('abstained') else esc(r.get('abstain_reason'), 20), CELL)])
story.append(table(tdata, widths=[6 * mm, 16 * mm, 9 * mm, 22 * mm, 12 * mm,
                                  18 * mm, 7 * mm, 7 * mm, 7 * mm, 6 * mm,
                                  7 * mm, 6 * mm, 6 * mm, 6 * mm, 8 * mm,
                                  22 * mm]))
story.append(Spacer(1, 3))
for c in cells:
    st = stats[c]
    story.append(P(f"{c[0]}|{c[1]}: shown <b>{mm1(st['shown'])}</b>, strict "
                   f"<b>{mm1(st['strict'])}</b>, false abstention "
                   f"{mm1(st['false_abs'])}, answered {st['n_ans']} "
                   f"(tiers {st['tiers']}), quote_ok {mm1(st['quote_ok'])}, "
                   f"probe abstention {mm1(st.get('probe_abs'))} "
                   f"CP95 {[round(x, 3) for x in (st.get('probe_cp') or [])]}.", SMALL))
story.append(P(f'Model usage histogram: {model_used_hist}. Rows with quota '
               f'rotation exceptions: {n_exceptions}. Gate: '
               f'<b>data/gate/scorer.pkl + tau.json</b> '
               f'(tau={tau_meta.get("tau")}, trained on runs 5-7 live labels; '
               f'small negative set -> generous threshold, the hard support veto '
               f'is what enforces strictness).', BODY))

story.append(P(f'{_sn + 2}. Honest notes and limits', H2))
story.append(P(
    '1. <b>Strict vs shown:</b> the headline rate is strict (text-verified). '
    'Shown counts visual-tier and otherwise-accepted-but-unverified rows; the '
    'gap between the two IS the verifier\u2019s effect and is reported per cell.<br/>'
    '2. <b>Visual tier:</b> cross-family checks add API calls (Groq) and can be '
    'wrong; accepted visual rows are flagged (tier=visual) in every table never '
    'merged into text-verified counts. The agreement fallback (n=3) is weaker '
    'than a cross-family call.<br/>'
    '3. <b>Probes:</b> corpus-absence tests (check tokens verified absent from '
    'all OCR). An answer could exist in image-only regions OCR missed, so an '
    'answered probe is a risk to audit, not automatically a factual error. CP '
    'intervals quantify binomial uncertainty at these small n.<br/>'
    '4. <b>Hybrid arm:</b> only 24 of 40 gold items were selected (the ones '
    'where keyword retrieval or reading actually failed); hybrid metrics on '
    'this subset are NOT comparable to the full-set keyword numbers, only to '
    'the same-subset keyword rows printed beside them.<br/>'
    '5. <b>Reader-model variance:</b> pinned models for comparability, but '
    'quota rotation still caused X rows to be served by a fallback model '
    '(recorded per row); n=1 on most rows means no agreement signal (neutral '
    '0.5 imputed).', BODY))

story.append(P(f'{_sn + 3}. Reproduce', H2))
story.append(P(
    'python scripts/relabel_gold.py -> python scripts/prepare_probes.py -> '
    'python scripts/prepare_test7_gold.py -> python '
    'scripts/select_hybrid_items.py -> python '
    'scripts/precompute_query_vectors.py (Colab; REQUIRED for the hybrid arm; '
    'scripts/run_test8.py --check-cache verifies) -> python '
    'scripts/make_gate_labels.py (rows 5-7) + python scripts/calibrate_gate.py '
    '--from-jsonl -> <b>python scripts/run_test8.py [--arms keyword,hybrid] '
    '[--mock] [--limit N] [--resume dir]</b> (25 s pacing, pinned models, '
    'quota rotation recorded) -> <b>python scripts/make_test8_pdf.py [rows_dir] '
    '[out]</b>.', SMALL))

from reportlab.platypus import SimpleDocTemplate
doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                        leftMargin=12 * mm, rightMargin=12 * mm,
                        topMargin=12 * mm, bottomMargin=12 * mm,
                        title='Test Run 8', author='altVRAG')
doc.build(story)
print('WROTE', OUT, OUT.stat().st_size, 'bytes')
