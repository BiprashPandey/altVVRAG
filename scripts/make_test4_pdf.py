"""Build testRun4.pdf: full narrative + per-run images + result table.

testRun4 = same matrix as testRun3 (5 questions x sample.pdf/sample.png x
span/heatmap) but on the sentence-citing + verified-gate pipeline, with the
gate running CALIBRATED (data/gate artifacts from scripts/calibrate_gate.py).

Usage: python scripts/make_test4_pdf.py [rows_dir] [out_pdf]
Defaults: results/demo/testrun4_20261017_195724 -> testRun4.pdf (repo root).
"""
import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / 'results' / 'demo' / 'testrun4_20261017_195724'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / 'testRun4.pdf'
INDEX = REPO / 'data' / 'index'

sys.path.insert(0, str(REPO / 'src'))

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image as RLImage, PageBreak, Paragraph, Spacer,
                                Table, TableStyle)

from vvrag.ground.text_in_box import join_words
from vvrag.schemas import WordBox
from vvrag.ui.qa import page_words

PAGE_W, PAGE_H = A4
styles = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=18, spaceAfter=6)
H2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=14, spaceAfter=4)
H3 = ParagraphStyle('H3', parent=styles['Heading3'], fontSize=12, spaceAfter=4)
BODY = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9, leading=12)
SMALL = ParagraphStyle('Small', parent=styles['Normal'], fontSize=7.5, leading=10)
CELL = ParagraphStyle('Cell', parent=styles['Normal'], fontSize=6.5, leading=8)
CELLH = ParagraphStyle('CellH', parent=styles['Normal'], fontSize=6.5, leading=8,
                       textColor=colors.white)
CAP = ParagraphStyle('Cap', parent=styles['Normal'], fontSize=8, leading=10,
                     textColor=colors.HexColor('#333333'))


def img(path, max_w=150 * mm, max_h=110 * mm):
    from PIL import Image as PILImage
    if not path or not Path(path).exists():
        return Paragraph('<i>[image missing]</i>', SMALL)
    w, h = PILImage.open(path).size
    s = min(max_w / w, max_h / h, 1.0)
    return RLImage(str(path), width=w * s, height=h * s)


def table(data, widths=None, header=True):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
             ('VALIGN', (0, 0), (-1, -1), 'TOP'),
             ('FONTSIZE', (0, 0), (-1, -1), 7)]
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
    return s if n is None or len(s) <= n else s[:n - 1] + '…'


rows = json.loads((RUN / 'rows.json').read_text(encoding='utf-8'))
manifest = json.loads((INDEX / 'manifest.json').read_text(encoding='utf-8'))
tau_meta = json.loads((REPO / 'data' / 'gate' / 'tau.json').read_text(encoding='utf-8'))
con = sqlite3.connect(INDEX / 'meta.db')
page_text, page_nwords = {}, {}
for doc in ('sample', 'sample_page'):
    words = page_words(con, doc, 0)
    page_text[doc] = join_words(words)
    page_nwords[doc] = len(words)

n_err = sum(1 for r in rows if 'error' in r)
n_em = sum(1 for r in rows if r.get('em') == 1.0)
n_abs = sum(1 for r in rows if r.get('abstained'))
n_part = sum(1 for r in rows if r.get('em') != 1.0 and not r.get('abstained') and 'error' not in r)
answered = [r for r in rows if not r.get('abstained') and 'error' not in r]
avg_sf1 = sum(r.get('sent_f1') or 0 for r in answered) / max(1, len(answered))
avg_wr = sum(r.get('word_r') or 0 for r in answered) / max(1, len(answered))
n_quote_ok = sum(1 for r in answered if r.get('quote_ok'))

story = []
story.append(P('Test Run 4 — same matrix, new pipeline: sentence citations + calibrated gate', H1))
story.append(P(f"Date: {dt.datetime.now().strftime('%Y-%m-%d')}. Same Colab index "
               f"<b>data/index</b> (real 128-dim ColQwen vectors) and same 5 gold "
               f"questions as Test Run 3, but run end-to-end on the refactored "
               f"pipeline (context.md): sentence-level citations with verbatim "
               f"quotes, deterministic quote verification, statement NLI on the "
               f"cited sentences, retrieval that never abstains, and a "
               f"<b>calibrated gate</b> (scorer.pkl + tau.json, tau={tau_meta['tau']} "
               f"from scripts/calibrate_gate.py). Reader: live Gemini "
               f"<b>gemini-3.5-flash</b> (n_samples=1), paced 25 s/call with "
               f"--resume after quota errors. Runner: <b>scripts/run_test4.py</b>; "
               f"report: <b>scripts/make_test4_pdf.py</b>. Rows: "
               f"<b>{RUN.name}/rows.json</b> ({len(rows)} rows, {n_err} errors).", BODY))
story.append(Spacer(1, 4))

# --- 1. corpus ---
story.append(P('1. Corpus and index', H2))
story.append(P('Unchanged from testRun3: index.zip (Colab GPU export) over '
               '<b>data/index</b>; the DB gained the sentence migration '
               '(word_boxes.sent_id + sentences table) and sentences are built '
               'on the fly for pages stored before it. Manifest:', BODY))
m = manifest
story.append(table([
    [P('<b>field</b>', CELLH), P('<b>value</b>', CELLH)],
    [P('timestamp', CELL), P(m['timestamp'], CELL)],
    [P('docs / pages / skipped / ocr_pages', CELL),
     P(f"{m['stats']['docs']} / {m['stats']['pages']} / {m['stats']['skipped']} / {m['stats']['ocr_pages']}", CELL)],
    [P('word_boxes / sqlite_pages', CELL),
     P(f"{m['word_boxes']} / {m['sqlite_pages']}", CELL)],
    [P('qdrant_full / qdrant_pooled', CELL),
     P(f"{m['qdrant_full']} / {m['qdrant_pooled']}", CELL)],
], widths=[70 * mm, 90 * mm]))
story.append(Spacer(1, 3))
meta = {r[0]: r for r in con.execute(
    'SELECT doc_id, page, width, height, grid_h, grid_w FROM pages')}
story.append(table(
    [[P('<b>doc</b>', CELLH), P('<b>page</b>', CELLH), P('<b>size px</b>', CELLH),
      P('<b>grid</b>', CELLH), P('<b>dim</b>', CELLH), P('<b>words</b>', CELLH)]] +
    [[P(d, CELL), P('0', CELL), P(f"{meta[d][2]}x{meta[d][3]}", CELL),
      P(f"{meta[d][4]}x{meta[d][5]}", CELL), P('128', CELL),
      P(str(page_nwords[d]), CELL)] for d in ('sample', 'sample_page')],
    widths=[30 * mm, 15 * mm, 30 * mm, 25 * mm, 15 * mm, 20 * mm]))
story.append(Spacer(1, 3))
story.append(P('<b>sample</b> = sample.pdf text layer (clean, 166 words). '
               '<b>sample_page</b> = sample.png via Tesseract OCR (noisy, 147 words: '
               '"POF"/"PDF", "tex!"/"text", "impertant"/"important"). Same 5 gold '
               'answers apply to both. Embeddings (128-dim ColQwen) untouched '
               'since testRun3.', BODY))
for doc, title in (('sample', 'sample.pdf rendered page'),
                   ('sample_page', 'sample.png rendered page')):
    story.append(P(f'Page image: {title} ({page_nwords[doc]} OCR words)', H3))
    story.append(img(INDEX / 'pages' / f'{doc}_p0.png'))
    story.append(P(f'Figure: {title} as ingested. In the per-run figures below, '
                   '<b>crop</b> is the region fed to the reader and '
                   '<b>overlay</b> is the full page with the FINAL evidence '
                   'boxes (green) drawn from the cited sentences.', CAP))
    story.append(Spacer(1, 4))

# --- 2. method ---
story.append(P('2. Method — what changed since testRun3', H2))
story.append(P('Same 5 questions x 2 versions x 2 groundings = 20 rows, but each '
               'row now runs the refactored pipeline:', BODY))
story.append(P('1. <b>Retrieval never abstains</b> (keyword rank; zero-overlap '
               'pages score 0.0 and are still returned in reading order) — the '
               'only abstention points are the reader and the gate.<br/>'
               '2. <b>Sentence units</b>: the page is segmented into numbered '
               'sentence spans (reading order via block/line ids; list items, '
               'table-cell gaps and paragraph jumps start new units), each with '
               'ONE union box.<br/>'
               '3. <b>Sentence-citing reader</b>: Gemini is shown the numbered '
               'sentences and must answer with a <b>verbatim quote</b> plus '
               '<b>sentence_ids</b>; a failed quote match triggers ONE retry '
               'with a corrective prompt, then abstention.<br/>'
               '4. <b>Deterministic quote check</b> (no model): fuzzy match of '
               "the quote against the page text (threshold 0.8).<br/>"
               '5. <b>Statement NLI</b> (DeBERTa, CUDA) on the CITED sentences '
               'as premise (not the raw crop).<br/>'
               '6. <b>Final boxes come only from cited sentences</b> (fuzzy '
               'answer-text localization as fallback, else the region prior).<br/>'
               '7. <b>Calibrated gate</b>: 9 features (incl. quote_match) &rarr; '
               f"learned LogisticScorer (data/gate/scorer.pkl) &rarr; tau="
               f"{tau_meta['tau']} (tau.json, alpha={tau_meta['alpha']} "
               f"delta={tau_meta['delta']}, fit on the offline synthetic eval "
               f"harness with {tau_meta['n_calib']} calibration items); "
               'abstains below tau with verifier rescue (evidence_valid + '
               'quote &ge; 0.9). If the reader itself returns UNANSWERABLE the '
               'row abstains immediately with that reason, and closest '
               'passages are returned for the viewer.', BODY))
story.append(Spacer(1, 3))
story.append(P('Gold set (unchanged from testRun3; png gold <i>sentence ids</i> '
               'for q4/q5 are empty because the OCR typo breaks the exact '
               'substring match — the pdf ids are the reference):', H3))
story.append(table(
    [[P('<b>qid</b>', CELLH), P('<b>question</b>', CELLH),
      P('<b>gold answer</b>', CELLH), P('<b>gold sids (pdf/png)</b>', CELLH)]] +
    [[P(g['qid'], CELL), P(g['question'], CELL), P(f"<b>{g['answer']}</b>", CELL),
      P(f"{g['sids']}", CELL)] for g in
     [{'qid': 'q1_title', 'question': 'What is the title of the document?',
       'answer': 'Sample Document for PDF Testing', 'sids': '0 / 0'},
      {'qid': 'q2_emph', 'question': 'What kind of emphasis can italic text give?',
       'answer': 'subtle emphasis', 'sids': '7 / 6'},
      {'qid': 'q3_list', 'question': 'What is the second entry of the ordered list?',
       'answer': 'Second item', 'sids': '16 / 11'},
      {'qid': 'q4_quote', 'question': 'What can a block quote be used to highlight?',
       'answer': 'important information', 'sids': '20 / — (OCR typo)'},
      {'qid': 'q5_table', 'question': 'What is in Row 1, Column 3 of the table?',
       'answer': 'Row 1, Col 3', 'sids': '24 / — (OCR "Col3")'}]],
    widths=[20 * mm, 70 * mm, 40 * mm, 40 * mm]))
story.append(Spacer(1, 2))
story.append(P('Quota note: first pass hit 503 high-demand on 3/20 rows; one '
               'paced --resume (25 s gaps) recovered all 3 — 0 errors final. '
               'Unlike testRun3, no model fallback was needed '
               '(gemini-3.5-flash served every call).', SMALL))
story.append(Spacer(1, 2))

# --- 3. per-question results ---
story.append(P('3. Per-question results (crop + final-overlay per run)', H2))
story.append(P('For every run: <b>crop</b> = region fed to the reader; '
               '<b>overlay</b> = full page with the FINAL sentence-derived '
               'evidence boxes in green. Captions give answer, metrics, the '
               'gate score/tau and the verbatim quote the reader was checked '
               'against. sP/sR/sF1 = sentence-citation precision/recall/F1 '
               'against gold sentence ids; wP/wR/wF1 = word-level '
               '(text, box) identity vs the gold answer span.', BODY))
by_q = {}
for r in rows:
    by_q.setdefault(r['qid'], []).append(r)
for qid in ['q1_title', 'q2_emph', 'q3_list', 'q4_quote', 'q5_table']:
    qrows = sorted(by_q[qid], key=lambda r: (r['version'], r['grounding']))
    story.append(P(f"{qid}: {qrows[0]['question']}", H3))
    story.append(P(f"Gold: <b>{qrows[0]['gold_answer']}</b> "
                   f"(pdf gold words: {qrows[0]['gold_words']}; "
                   f"png gold words: {[r for r in qrows if r['version'] == 'png'][0]['gold_words']})", BODY))
    for r in qrows:
        if 'error' in r:
            story.append(P(f"<b>{r['version']}/{r['grounding']}</b> — ERROR: "
                           f"{esc(r['error'], 120)}", SMALL))
            continue
        loc = r.get('localize', {}) or {}
        samples = r.get('samples') or [{}]
        quote = samples[0].get('quote', '') or ''
        cited = r.get('cited_sentences') or []
        cited_txt = '; '.join(f"#{c['sent_id']} “{esc(c['text'], 70)}”"
                              for c in cited) if cited else '(none)'
        story.append(P(
            f"<b>{r['version']}/{r['grounding']}</b> (reader {esc(r.get('reader_model'), 30)}) — "
            f"answer: <b>{esc(r.get('answer'))}</b> "
            f"{'(ABSTAINED: ' + esc(r.get('abstain_reason', '')) + ') ' if r.get('abstained') else ''}"
            f"| EM {r.get('em')} F1 {(r.get('f1') or 0):.2f} "
            f"| sP/R/F1 {r.get('sent_p')}/{r.get('sent_r')}/{r.get('sent_f1')} "
            f"wP/R/F1 {r.get('word_p')}/{r.get('word_r')}/{r.get('word_f1')} "
            f"IoU {r.get('iou')} "
            f"| score {r.get('score')} vs tau {r.get('tau')} "
            f"| quote_ok {r.get('quote_ok')} ({loc.get('score', '')}) "
            f"| quote: “{esc(quote, 80)}”", SMALL))
        story.append(P(f"cited sentences: {cited_txt}", SMALL))
        if r.get('abstained') and r.get('closest_passages'):
            story.append(P('closest passages: '
                           + '; '.join(f"“{esc(p['text'], 80)}”"
                                       for p in r['closest_passages'][:3]), SMALL))
        story.append(table([
            [P('<b>crop (reader input)</b>', CELLH),
             P('<b>overlay (page + final evidence boxes)</b>', CELLH)],
            [img(r.get('crop'), 85 * mm, 80 * mm),
             img(r.get('overlay'), 85 * mm, 80 * mm)],
        ], widths=[85 * mm, 85 * mm]))
        story.append(Spacer(1, 3))

# --- 4. analysis ---
story.append(P('4. Analysis', H2))
story.append(P(f'<b>Headline vs testRun3: {n_em}/20 exact (was 10), '
               f'{n_part}/20 partial (was 4), {n_abs}/20 abstained (was 6), '
               f'0 errors.</b> All {len(answered)} answered rows verified: '
               f'quote_ok {n_quote_ok}/{len(answered)}, average sentence-F1 '
               f'{avg_sf1:.2f}, average word recall {avg_wr:.2f}.', BODY))
story.append(P('<b>q1 answers now — retrieval no longer abstains.</b> In '
               'testRun3 all four q1 cells died at retrieval ("document" is a '
               'stopword; "title" never appears on the page &rarr; zero keyword '
               'overlap). The refactored keyword rank scores zero-overlap pages '
               '0.0 instead of dropping them, so all four cells reached the '
               'reader and returned EM=1 / IoU=1.0 citing sentence 0 (the '
               'heading) with a verbatim quote. The blind spot became an '
               'honest last-resort path instead of a hard failure.', BODY))
story.append(P('<b>Grounding is now verbatim and whole-sentence.</b> 16/16 '
               'answered rows passed the deterministic quote check, and every '
               'one localized via sentence_ids (loc=source sentence_ids, no '
               'fuzzy fallback). The word-id line-shifting that plagued '
               'testRun3 (cited words one line off the answer) cannot happen '
               'anymore: the reader cites a sentence id and a quote that must '
               'match the page text, and the box is that sentence\'s union '
               'box. The metric trade-off is visible in the IoU column: '
               'sentence boxes are whole-sentence unions, so union-IoU vs the '
               'tight gold answer box is lower than testRun3\'s word-line '
               'boxes (e.g. q5 pdf IoU 0.16) even though word recall is 1.0 '
               'and the quote pins the exact answer string. q4 pdf cites '
               'sentence 20 + 21 (quote "important information or '
               'citations"), F1 0.67 vs the shorter gold — same verbosity '
               'artifact as testRun3, now bounded by the quoted text.', BODY))
story.append(P('<b>The 4 abstentions are honest reader abstentions, not gate '
               'or retrieval failures.</b> q3/pdf-heatmap, q4/png (both '
               'groundings) and q5/png-heatmap each returned UNANSWERABLE '
               'directly from the reader: three are the OCR-noise cells '
               '("impertant", "Col3" — the model declines to guess when the '
               'page text is garbled) and one is a heatmap region miss (q3 '
               'list under heatmap grounding, while its span twin scored '
               'EM=1). All four rows scored &ge; 0.93 — the calibrated tau '
               '(0.4) never had to block anything, and verifier rescue was '
               'never needed. Each abstained row carries 3 closest passages '
               'for the viewer\'s "Not found in this document" message. '
               'Contrast with testRun3: its q4/png cells answered through '
               'fuzzy word matching; the sentence pipeline prefers to abstain '
               'when the quote cannot be verified against noisy OCR — fewer '
               'answers on those two cells (q4 png 0/2 vs 2/2 in testRun3) '
               'but zero unverifiable claims.', BODY))
story.append(P('<b>Gate/calibration honesty.</b> The scorer+tau come from the '
               'offline synthetic harness (SimulatedReader, fake embedder, '
               f"{tau_meta['n_calib']} calibration items), not from live "
               'Gemini data — scores cluster near 1.0 on this easy 5-question '
               'set, so the calibrated gate agrees with "answer" on all 16 '
               'answerable rows and would not have changed any testRun3 '
               'decision at tau=0.4. The per-row score-vs-tau columns make '
               'any future miscalibration visible. All abstentions in this '
               'run came from the reader\'s own UNANSWERABLE verdict, which '
               'is the intended order of last resorts: reader (with one '
               'quote-retry), then gate, never retrieval.', BODY))
story.append(P('<b>Sentence-id metric caveat.</b> q5/png/span cites sentence '
               '17 with the verbatim OCR quote "Row 1, Col3" and EM=1, but '
               'reports sF1=0.0 because the gold png sentence list is empty '
               '(the OCR typo breaks the gold derivator\'s exact match). This '
               'is an eval-harness artifact, not a grounding miss; the pdf '
               'columns are the sentence-metric reference.', BODY))
story.append(Spacer(1, 2))

# --- 5. result table ---
story.append(P('5. Result table (20 rows)', H2))
hdr = ['qid', 'ver', 'grnd', 'answer', 'EM', 'F1', 'sF1', 'quote',
       'score', 'tau', 'abstain reason']
tdata = [[P(f'<b>{h}</b>', CELLH) for h in hdr]]
for r in rows:
    if 'error' in r:
        tdata.append([P(r['qid'].replace('_', '<br/>'), CELL), P(r['version'], CELL),
                      P(r['grounding'], CELL), P('ERROR', CELL)] +
                     [P('', CELL) for _ in range(7)])
        continue
    tdata.append([
        P(r['qid'].replace('_', '<br/>'), CELL), P(r['version'], CELL),
        P(r['grounding'], CELL), P(esc(r.get('answer'), 24), CELL),
        P(str(r.get('em')), CELL), P(f"{(r.get('f1') or 0):.2f}", CELL),
        P(str(r.get('sent_f1')), CELL),
        P('Y' if r.get('quote_ok') else ('N' if r.get('quote_ok') is not None else '-'), CELL),
        P(str(r.get('score')), CELL), P(str(r.get('tau')), CELL),
        P(esc(r.get('abstain_reason'), 22), CELL)])
story.append(table(tdata, widths=[22 * mm, 10 * mm, 14 * mm, 30 * mm, 8 * mm,
                                  10 * mm, 10 * mm, 12 * mm, 14 * mm, 10 * mm,
                                  24 * mm]))
story.append(Spacer(1, 3))
story.append(P(f'Summary: <b>{n_em}/20 exact</b>, <b>{n_part}/20 partial</b> '
               f'(q4 pdf x2, F1 0.67), <b>{n_abs}/20 abstained</b> '
               f'(reader UNANSWERABLE: q3 pdf-heat, q4 png x2, q5 png-heat), '
               f'{n_err} errors after one paced resume. 16/16 answered rows '
               f'quote-verified and localized from cited sentences. '
               f'Artifacts: <b>{RUN.name}/</b> (rows.json + per-run '
               f'crop/overlay PNGs); gate: <b>data/gate/scorer.pkl + '
               f'tau.json</b>.', BODY))

from reportlab.platypus import SimpleDocTemplate
doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                        leftMargin=15 * mm, rightMargin=15 * mm,
                        topMargin=15 * mm, bottomMargin=15 * mm,
                        title='Test Run 4', author='altVRAG')
doc.build(story)
print('WROTE', OUT, OUT.stat().st_size, 'bytes')
