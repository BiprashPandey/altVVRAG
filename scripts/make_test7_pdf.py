"""Build testRun7.pdf: honest-evaluation report (gold + unanswerable probes).

testRun7 = SlideVQA index (8 decks / 160 pages), 40 gold items (categories:
single-hop/multi-hop/image-only) + 9 verified-absent unanswerable probes,
single grounding (span), hybrid retrieval (falls back to keyword without
cached query vectors; mode recorded per row), multi-page reader (top-3) with
next-page retry, quote + answer-support verification, calibrated gate,
number-normalized equivalence, bootstrap CIs.

Usage: python scripts/make_test7_pdf.py [rows_dir] [out_pdf]
Default rows_dir: newest results/demo/testrun7_*/.
"""
import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_cands = sorted((REPO / 'results' / 'demo').glob('testrun7_*'),
                key=lambda p: p.stat().st_mtime)
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else (_cands[-1] if _cands else None)
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / 'testRun7.pdf'
INDEX = REPO / 'data' / 'index'
GOLD = REPO / 'data' / 'gold.json'

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
    sys.exit('No testrun7 rows.json found — run scripts/run_test7.py first.')

rows = json.loads((RUN / 'rows.json').read_text(encoding='utf-8'))
golds = json.loads(GOLD.read_text(encoding='utf-8'))
try:
    tau_meta = json.loads(
        (REPO / 'data' / 'gate' / 'tau.json').read_text(encoding='utf-8'))
except Exception:  # noqa: BLE001
    tau_meta = {'tau': '?'}

con = sqlite3.connect(str(INDEX / 'meta.db'))
manifest = json.loads((INDEX / 'manifest.json').read_text(encoding='utf-8')) \
    if (INDEX / 'manifest.json').exists() else {}

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

# ---- metrics ----
ok = [r for r in rows if 'error' not in r]
errs = [r for r in rows if 'error' in r]
gold_rows = [r for r in ok if not r.get('unanswerable')]
probes = [r for r in ok if r.get('unanswerable')]
ans = [r for r in gold_rows if not r.get('abstained')]
n_equiv = sum(1 for r in gold_rows if r.get('equiv_correct'))
n_equiv_raw = sum(1 for r in gold_rows if r.get('equiv_correct_raw'))
n_abs = sum(1 for r in gold_rows if r.get('abstained'))
n_ph1 = sum(1 for r in gold_rows if r.get('page_hit'))
n_ph3 = sum(1 for r in gold_rows if (r.get('gold_rank') or 99) <= 3)
ranks = [r.get('gold_rank') for r in gold_rows if r.get('gold_rank')]
mrr = sum(1.0 / r for r in ranks) / len(ranks) if ranks else None
n_quote = sum(1 for r in ans if r.get('quote_ok'))
n_ver = sum(1 for r in ans if r.get('verified'))
cond_acc = n_equiv / len(ans) if ans else None
probe_abs = sum(1 for r in probes if r.get('abstained'))
probe_fa = sum(1 for r in probes if not r.get('abstained'))
ev_sent = [r for r in ans if r.get('sent_f1') is not None]
ev_box = [r for r in ans if r.get('iou') is not None]
m_sf1 = _mean([r['sent_f1'] for r in ev_sent])
m_iou = _mean([r['iou'] for r in ev_box])
# should-answer subset (same policy as run 6)
sa = [r for r in gold_rows if r.get('page_hit')
      and (r.get('gold_box') or r.get('gold_sentence_ids'))]
sa_ans = [r for r in sa if not r.get('abstained')]
sa_equiv = sum(1 for r in sa_ans if r.get('equiv_correct'))
model_use = {}
for r in ok:
    model_use[r.get('reader_model', '?')] = model_use.get(r.get('reader_model', '?'), 0) + 1
modes_used = {}
for r in ok:
    m = (r.get('retrieval_debug') or {}).get('mode_used')
    modes_used[m] = modes_used.get(m, 0) + 1
n3_rows = [r for r in gold_rows if r.get('n_samples', 1) > 1]

try:
    from vvrag.eval.metrics import bootstrap_ci
    import numpy as np

    e_mean, e_lo, e_hi = bootstrap_ci(
        np.array([1.0 if r.get('equiv_correct') else 0.0 for r in gold_rows]))
    p_mean, p_lo, p_hi = bootstrap_ci(
        np.array([1.0 if r.get('abstained') else 0.0 for r in probes])) \
        if probes else (0.0, 0.0, 0.0)
except Exception:  # noqa: BLE001
    e_mean = e_lo = e_hi = p_mean = p_lo = p_hi = None

cats = {}
for cat in ('single-hop', 'multi-hop', 'image-only'):
    sub = [r for r in gold_rows if r.get('category') == cat]
    sub_ans = [r for r in sub if not r.get('abstained')]
    cats[cat] = {
        'n': len(sub),
        'answered': len(sub_ans),
        'equiv': sum(1 for r in sub if r.get('equiv_correct')),
        'false_abs': len(sub) - len(sub_ans),
        'quote_ok': sum(1 for r in sub_ans if r.get('quote_ok')),
    }

def gold_overlay(r):
    """Render (once, cached) the gold box on the gold slide for row r."""
    if not r.get('gold_box') or r.get('gold_page') is None:
        return None
    out = _img_cache / f"gold_{RUN.name}_{r['idx']}.jpg"
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


story = []
story.append(P('Test Run 7 — honest evaluation: gold items + unanswerable '
               'probes, hybrid retrieval, multi-page reader, support veto', H1))
story.append(P(
    f"Date: <b>{dt.datetime.now().strftime('%Y-%m-%d %H:%M')}</b>. Index: "
    f"<b>{len(docs)} SlideVQA decks, {n_pages} pages, {n_words:,} OCR word "
    f"boxes, {n_sents:,} sentences, {n_embeds} ColQwen page vectors</b>. "
    f"Matrix: <b>{len(gold_rows)} gold items + {len(probes)} unanswerable "
    f"probes = {len(rows)} rows</b>, single grounding (<b>span</b>), "
    f"n_samples=1 ({len(n3_rows)} items re-run with n=3 for agreement), "
    f"retrieval=<b>hybrid</b> (keyword fallback without cached query vectors; "
    f"modes used: {modes_used}), multi-page reader context (top-3 with "
    f"next-page retry), quote + answer-support verification, gate="
    f"<b>calibrated</b> (tau={tau_meta.get('tau')}). Row-level result: "
    f"<b>{RUN.name}/rows.json</b> ({len(rows)} rows, {len(errs)} errors).", BODY))
story.append(Spacer(1, 4))

story.append(P('1. What testRun7 adds over testRun6', H2))
story.append(P(
    '1. <b>Unanswerable probes</b>: 9 questions whose answer facts are '
    'verified ABSENT from the entire corpus OCR (scripts/prepare_test7_gold.py; '
    'each probe carries check tokens that must not occur anywhere). A correct '
    'system abstains on all of them; answering any is a false-positive risk '
    'report.<br/>'
    '2. <b>Hybrid retrieval</b>: keyword BM25-ish rank fused (RRF) with '
    'ColQwen visual page scores from cached query vectors. Without a cache the '
    'pipeline records a keyword fallback in the row (mode_used), so the report '
    'never overstates what ran.<br/>'
    '3. <b>Multi-page reader context + next-page retry</b>: the reader sees up '
    'to 3 ranked pages and can cite sentences on any of them ("p1.s12"); a '
    'one-shot retry re-prompts with the second page when the first attempt '
    'abstains.<br/>'
    '4. <b>Answer-support veto</b>: a shown answer is accepted only when its '
    'text is contained in (or NLI-entailed by) the cited sentences; '
    'quote_ok AND supported -> verified. Unverified answers cannot reach the '
    'displayed output via the gate.<br/>'
    '5. <b>Number-normalized equivalence + CIs + categories</b>: "12.5 bn" '
    'counts as "12.5 billion"; every headline rate carries a bootstrap 95% CI; '
    'items are split single-hop / multi-hop / image-only.<br/>'
    '6. <b>n=3 subset</b>: the first 15 gold items run with 3 samples so '
    'agreement is a real signal there (n=1 rows impute the neutral 0.5).', BODY))
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
story.append(P('Retrieval is full-corpus (doc_id=None, the real UI path). '
               'Visual page scores need query multivectors in data/query_cache '
               '(scripts/precompute_query_vectors.py, GPU/Colab); a cache miss '
               'falls back to keyword rank for that question and says so in '
               'the row.', SMALL))
story.append(Spacer(1, 3))

story.append(P('3. Gold set (40 items, categorized) + unanswerable probes', H2))
gdata = [[P('<b>#</b>', CELLH), P('<b>question</b>', CELLH),
          P('<b>gold</b>', CELLH), P('<b>cat</b>', CELLH),
          P('<b>p</b>', CELLH), P('<b>res</b>', CELLH), P('<b>qok</b>', CELLH)]]
for gi, g in enumerate(golds):
    rr = next((x for x in gold_rows if x['qid'] == g['qid']), None)
    res = '—'
    if rr:
        res = ('ABS' if rr.get('abstained')
               else ('Y' if rr.get('equiv_correct') else 'N'))
    gdata.append([
        P(str(gi + 1), CELL), P(esc(g['question'], 92), CELL),
        P(f"<b>{esc(g['answer'], 28)}</b>", CELL),
        P(esc(g.get('category', '?'), 11), CELL), P(str(g['page']), CELL),
        P(res, CELL),
        P('Y' if rr and rr.get('quote_ok') else ('N' if rr and not rr.get('abstained') else '—'), CELL)])
story.append(table(gdata, widths=[6 * mm, 76 * mm, 32 * mm, 19 * mm, 7 * mm,
                                  9 * mm, 8 * mm]))
story.append(Spacer(1, 2))
pdata = [[P('<b>probe</b>', CELLH), P('<b>question</b>', CELLH),
          P('<b>check tokens (verified absent)</b>', CELLH),
          P('<b>abstained</b>', CELLH), P('<b>answer if not</b>', CELLH)]]
for r in probes:
    q = r['qid']
    item = next((g for g in json.loads(Path(REPO / 'data' / 'gold_unanswerable.json')
                                       .read_text(encoding='utf-8'))
                 if g['qid'] == q), {}) if (REPO / 'data' / 'gold_unanswerable.json').exists() else {}
    pdata.append([
        P(esc(q, 12), CELL), P(esc(r['question'], 80), CELL),
        P(esc(', '.join(item.get('check_absent', [])), 40), CELL),
        P('Y' if r.get('abstained') else '<b>N</b>', CELL),
        P(esc(r.get('answer'), 30), CELL)])
story.append(table(pdata, widths=[14 * mm, 78 * mm, 34 * mm, 14 * mm, 20 * mm]))
story.append(Spacer(1, 3))

story.append(P('4. Headline metrics', H2))
story.append(table([
    [P('<b>metric</b>', CELLH), P('<b>value</b>', CELLH),
     P('<b>n</b>', CELLH), P('<b>note</b>', CELLH)],
    [P('equiv_correct (number-normalized)', CELL),
     P(f'<b>{mm1(e_mean)}</b>  CI [{mm1(e_lo)}, {mm1(e_hi)}]', CELL),
     P(str(len(gold_rows)), CELL), P('bootstrap 95%', CELL)],
    [P('equiv_correct (legacy matcher)', CELL), P(mm1(n_equiv_raw / max(1, len(gold_rows))), CELL),
     P(str(len(gold_rows)), CELL), P('for continuity with run 6', CELL)],
    [P('accuracy when answering', CELL), P(f'<b>{mm1(cond_acc)}</b>', CELL),
     P(str(len(ans)), CELL), P('conditional on not abstaining', CELL)],
    [P('false abstention (gold items)', CELL), P(f'<b>{mm1(n_abs / max(1, len(gold_rows)))}</b>', CELL),
     P(str(len(gold_rows)), CELL), P('', CELL)],
    [P('page_hit @1 / @3', CELL), P(f'{mm1(n_ph1 / max(1, len(gold_rows)))} / {mm1(n_ph3 / max(1, len(gold_rows)))}', CELL),
     P(str(len(gold_rows)), CELL), P(f'MRR {mm1(mrr)}', CELL)],
    [P('quote_ok / verified (answered)', CELL),
     P(f'{n_quote}/{len(ans)} / <b>{n_ver}/{len(ans)}</b>', CELL),
     P(str(len(ans)), CELL), P('deterministic quote check + support veto', CELL)],
    [P('sent-F1 / IoU (evaluable)', CELL), P(f'{mm1(m_sf1)} / {mm1(m_iou)}', CELL),
     P(f'{len(ev_sent)}/{len(ev_box)}', CELL), P('page-aware policy: None excluded', CELL)],
    [P('unanswerable abstention', CELL),
     P(f'<b>{mm1(p_mean)}</b>  CI [{mm1(p_lo)}, {mm1(p_hi)}]', CELL),
     P(str(len(probes)), CELL), P(f'{probe_abs} abstained, {probe_fa} answered', CELL)],
], widths=[58 * mm, 52 * mm, 16 * mm, 50 * mm]))
story.append(Spacer(1, 2))
story.append(P('By category:', H3))
cdata = [[P('<b>category</b>', CELLH), P('<b>n</b>', CELLH),
          P('<b>equiv</b>', CELLH), P('<b>false abst.</b>', CELLH),
          P('<b>quote_ok/answered</b>', CELLH)]]
for cat, v in cats.items():
    cdata.append([P(cat, CELL), P(str(v['n']), CELL),
                  P(f"{v['equiv']}/{v['n']}" if v['n'] else '—', CELL),
                  P(f"{v['false_abs']}/{v['n']}" if v['n'] else '—', CELL),
                  P(f"{v['quote_ok']}/{v['answered']}" if v['answered'] else '—', CELL)])
story.append(table(cdata, widths=[40 * mm, 16 * mm, 30 * mm, 30 * mm, 60 * mm]))
story.append(P(f'Should-answer subset (page_hit AND gold context derivable): '
               f'n={len(sa)}, answered {len(sa_ans)}, equiv on it '
               f'<b>{sa_equiv}/{len(sa_ans)}</b> '
               f'({mm1(sa_equiv / len(sa_ans) if sa_ans else None)}).', BODY))
story.append(P(f'Reader models: '
               + ', '.join(f'{k} x{v}' for k, v in sorted(model_use.items(),
                                                           key=lambda kv: -kv[1]))
               + f'. Retrieval modes used: {modes_used}.', SMALL))
story.append(Spacer(1, 3))

story.append(P('5. Per-item results', H2))
story.append(P('Gold items: gold overlay = derived gold box on the gold slide; '
               'crop = reader input on the evidence page; overlay = final '
               'evidence boxes. Probes: crop/overlay of the top-ranked page '
               'plus the closest passages offered to the reader.', BODY))
for r0 in sorted(gold_rows, key=lambda r: r['idx']):
    if 'error' in r0:
        story.append(P(f"[{r0['idx']}] {esc(r0['question'], 130)} — ERROR: "
                       f"{esc(r0['error'], 130)}", SMALL))
        continue
    story.append(P(f"[{r0['idx']}] {esc(r0['question'], 150)}", H3))
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
        f"[{esc(r0.get('reader_model'), 22)}] ans: <b>{esc(r0.get('answer'), 34)}</b>"
        f"{' (ABSTAIN: ' + esc(r0.get('abstain_reason', ''), 44) + ')' if r0.get('abstained') else ''}"
        f" | equiv {r0.get('equiv_correct')} (raw {r0.get('equiv_correct_raw')})"
        f" | page_hit {r0.get('page_hit')} rank {r0.get('gold_rank') or '>5'}"
        f" | quote_ok {r0.get('quote_ok')} (q={loc.get('quote_score', '')})"
        f" | verified {r0.get('verified')} (support {esc(sup.get('method'), 10)})"
        f" | sF1 {mm1(r0.get('sent_f1'))} IoU {mm1(r0.get('iou'))}"
        f" | score {r0.get('score')} / tau {r0.get('tau')}", SMALL))
    story.append(P(f"cited: {cited_txt} | quote: \u201c{esc(quote, 90)}\u201d", SMALL))
    if r0.get('abstained') and r0.get('closest_passages'):
        story.append(P('closest passages: '
                       + '; '.join(f"\u201c{esc(p.get('text'), 70)}\u201d"
                                   for p in (r0['closest_passages'] or [])[:3]),
                       SMALL))
    story.append(table([
        [P('<b>gold overlay (gold box)</b>', CELLH),
         P('<b>crop (reader input)</b>', CELLH),
         P('<b>overlay (evidence boxes)</b>', CELLH)],
        [img(gold_overlay(r0), 58 * mm, 56 * mm),
         img(r0.get('crop'), 58 * mm, 56 * mm),
         img(r0.get('overlay'), 58 * mm, 56 * mm)],
    ], widths=[58.5 * mm, 58.5 * mm, 58.5 * mm]))
    story.append(Spacer(1, 3))

story.append(P('6. Unanswerable probes (detail)', H2))
for r0 in sorted(probes, key=lambda r: r['idx']):
    story.append(P(f"[{r0['idx']}] {esc(r0['question'], 140)}", H3))
    verdict = ('<b>ABSTAINED (correct)</b>' if r0.get('abstained')
               else f"<b>ANSWERED (false-positive risk): "
                    f"{esc(r0.get('answer'), 40)}</b>")
    story.append(P(f"{verdict} | reason {esc(r0.get('abstain_reason'), 60)}"
                   f" | mode {(r0.get('retrieval_debug') or {}).get('mode_used')}"
                   f" | score {r0.get('score')} / tau {r0.get('tau')}", SMALL))
    if r0.get('closest_passages'):
        story.append(P('closest passages: '
                       + '; '.join(f"\u201c{esc(p.get('text'), 80)}\u201d"
                                   for p in (r0['closest_passages'] or [])[:3]),
                       SMALL))
    if 'error' in r0:
        story.append(P(f"ERROR: {esc(r0.get('error'), 120)}", SMALL))
    elif r0.get('crop') or r0.get('overlay'):
        story.append(table([
            [P('<b>crop (top-ranked page)</b>', CELLH),
             P('<b>overlay</b>', CELLH)],
            [img(r0.get('crop'), 88 * mm, 60 * mm),
             img(r0.get('overlay'), 88 * mm, 60 * mm)],
        ], widths=[88.5 * mm, 88.5 * mm]))
    story.append(Spacer(1, 3))

story.append(P('7. Full result table', H2))
hdr = ['idx', 'qid', 'cat', 'n', 'answer', 'eql', 'raw', 'pg1', 'rank', 'qok',
       'ver', 'sup', 'score', 'tau', 'model', 'verdict']
tdata = [[P(f'<b>{h}</b>', CELLH) for h in hdr]]
for r in rows:
    short_qid = r['qid'].split('-', 1)[-1] if '-' in r['qid'] else r['qid']
    if 'error' in r:
        tdata.append([P(str(r.get('idx', '?')), CELL), P(esc(short_qid, 22), CELL),
                      P(esc(r.get('category'), 10), CELL)] +
                     [P('ERROR: ' + esc(r.get('error'), 40), CELL)] +
                     [P('', CELL) for _ in range(12)])
        continue
    sup = (r.get('support') or {}).get('method', '')
    tdata.append([
        P(str(r['idx']), CELL), P(esc(short_qid, 22), CELL),
        P(esc(r.get('category'), 10), CELL), P(str(r.get('n_samples')), CELL),
        P(esc(r.get('answer'), 20), CELL),
        P('Y' if r.get('equiv_correct') else ('N' if not r.get('unanswerable') else '<b>N</b>'), CELL),
        P('Y' if r.get('equiv_correct_raw') else 'N', CELL),
        P('Y' if r.get('page_hit') else ('N' if r.get('page_hit') is not None else '—'), CELL),
        P(str(r.get('gold_rank') or ('—' if r.get('unanswerable') else '>5')), CELL),
        P('Y' if r.get('quote_ok') else ('N' if r.get('quote_ok') is not None else '—'), CELL),
        P('Y' if r.get('verified') else 'N', CELL),
        P(esc(sup, 11), CELL),
        P(str(r.get('score')), CELL), P(str(r.get('tau')), CELL),
        P(esc(r.get('reader_model'), 18), CELL),
        P('ABS' if r.get('abstained') else esc(r.get('abstain_reason'), 16), CELL)])
story.append(table(tdata, widths=[6 * mm, 22 * mm, 11 * mm, 5 * mm, 20 * mm,
                                  6 * mm, 6 * mm, 7 * mm, 8 * mm, 6 * mm,
                                  6 * mm, 12 * mm, 8 * mm, 7 * mm, 20 * mm,
                                  14 * mm]))
story.append(Spacer(1, 3))
story.append(P(f'Summary: <b>{n_equiv}/{len(gold_rows)} gold items '
               f'equiv-correct (number-normalized)</b>, <b>{n_abs} false '
               f'abstentions</b>, <b>{n_quote} quote-ok / {n_ver} verified of '
               f'{len(ans)} answered</b>, page_hit@1 {n_ph1}/{len(gold_rows)} '
               f'(@3 {n_ph3}), <b>{probe_abs}/{len(probes)} probes abstained</b>, '
               f'{len(errs)} errors. Artifacts: <b>{RUN.name}/</b>; gate: '
               f'<b>data/gate/scorer.pkl + tau.json</b> '
               f'(tau={tau_meta.get("tau")}).', BODY))

story.append(P('8. Honest notes and limits', H2))
story.append(P(
    '1. <b>Retrieval modes:</b> visual/hybrid scoring needs cached ColQwen '
    'query vectors; rows answered from keyword fallback are visible in the '
    'mode histogram. Hybrid ranking is only as good as the cache coverage.<br/>'
    '2. <b>n=1 majority:</b> most rows ran one sample (no agreement signal; '
    'imputed neutral 0.5). The n=3 subset (first 15 gold items) shows the real '
    'agreement behaviour.<br/>'
    '3. <b>Reader-model variance:</b> rotation on quota may serve different '
    'models across rows; same-question disagreement mixes model and system '
    'behaviour (listed per row).<br/>'
    '4. <b>Probes are corpus-absence tests, not falsehood tests:</b> check '
    'tokens were verified absent from all OCR; an answer could still exist in '
    'an image region OCR missed, so probe "answered" rows are risks to audit, '
    'not automatically wrong beyond the corpus.<br/>'
    '5. <b>Grounding metric exclusions</b> follow the run-6 page-aware policy '
    '(wrong page or no gold context -> None, excluded from means).', BODY))

story.append(P('9. Reproduce', H2))
story.append(P(
    'python scripts/prepare_test7_gold.py (categories + verified probes) -> '
    'python scripts/precompute_query_vectors.py (optional, Colab/GPU; fills '
    'data/query_cache) -> <b>python scripts/run_test7.py [--limit N] '
    '[--resume dir] [--mock] [--model m]</b> (25 s pacing, rotation on 429) '
    '-> <b>python scripts/make_test7_pdf.py [rows_dir] [out]</b>. Gate '
    'artifacts: python scripts/calibrate_gate.py (synthetic) or '
    '--from-jsonl for live labels (scripts/collect_live_labels.py).', SMALL))

from reportlab.platypus import SimpleDocTemplate
doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                        leftMargin=12 * mm, rightMargin=12 * mm,
                        topMargin=12 * mm, bottomMargin=12 * mm,
                        title='Test Run 7', author='altVRAG')
doc.build(story)
print('WROTE', OUT, OUT.stat().st_size, 'bytes')
