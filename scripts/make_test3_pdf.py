"""Build testRun3.pdf: full narrative + per-run images + result table.

Usage: python scripts/make_test3_pdf.py [rows_dir] [out_pdf]
Defaults: results/demo/testrun3_20261016_124649 -> testRun3.pdf (repo root).
"""
import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / 'results' / 'demo' / 'testrun3_20261016_124649'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / 'testRun3.pdf'
INDEX = REPO / 'data' / 'index'

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


rows = json.loads((RUN / 'rows.json').read_text(encoding='utf-8'))
manifest = json.loads((INDEX / 'manifest.json').read_text(encoding='utf-8'))
con = sqlite3.connect(INDEX / 'meta.db')
page_text, page_nwords = {}, {}
for doc in ('sample', 'sample_page', 'testimg'):
    words = page_words(con, doc, 0)
    page_text[doc] = join_words(words)
    page_nwords[doc] = len(words)

story = []
story.append(P('Test Run 3 — 5 questions x pdf/png versions x span/heatmap grounding', H1))
story.append(P(f"Date: {dt.datetime.now().strftime('%Y-%m-%d')}. Index: Colab-built "
               f"<b>data/index</b> (real 128-dim ColQwen vectors). "
               f"Runner: <b>scripts/run_test3.py</b> (+ paced resume after quota/503 errors). "
               f"Rows: <b>{RUN.name}/rows.json</b> (20 rows, 0 errors after resume). "
               f"Reader: live Gemini (<b>gemini-3.5-flash</b> then fallback "
               f"<b>gemini-3.5-flash-lite</b>), n_samples=1, gate fixed @ 0.5. "
               f"Report: <b>scripts/make_test3_pdf.py</b>.", BODY))
story.append(Spacer(1, 4))

# --- 1. corpus ---
story.append(P('1. Corpus and index', H2))
story.append(P('index.zip (Colab GPU export) was extracted over <b>data/index</b> '
               '(previous testRun1/2 index backed up). Manifest:', BODY))
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
      P(str(page_nwords[d]), CELL)] for d in ('sample', 'sample_page', 'testimg')],
    widths=[30 * mm, 15 * mm, 30 * mm, 25 * mm, 15 * mm, 20 * mm]))
story.append(Spacer(1, 3))
story.append(P('<b>sample</b> = sample.pdf text layer (clean, 166 words). '
               '<b>sample_page</b> = sample.png via Tesseract OCR (noisy, 147 words: '
               '"POF"/"PDF", "tex!"/"text", "impertant"/"important"). Both hold the same '
               'document, so the same 5 gold answers apply to both versions. '
               '<b>testimg</b> (Nepal Rastra Bank page, testRun1/2) ships in the index but '
               'is excluded from this matrix via a per-question doc filter.', BODY))
for doc, title in (('sample', 'sample.pdf rendered page'),
                   ('sample_page', 'sample.png rendered page')):
    story.append(P(f'Page image: {title} ({page_nwords[doc]} OCR words)', H3))
    story.append(img(INDEX / 'pages' / f'{doc}_p0.png'))
    story.append(P(f'Figure: {title} as ingested (full-page evidence context). '
                   'Green boxes in the overlay figures below mark the final per-line '
                   'evidence boxes for each run.', CAP))
    story.append(Spacer(1, 4))

# --- 2. method ---
story.append(P('2. Method', H2))
story.append(P('Each of the 5 questions runs against <b>both versions</b> (pdf doc '
               '<b>sample</b>, png doc <b>sample_page</b>) under <b>both groundings</b> '
               '(keyword <b>span</b>, <b>heatmap</b> pseudo-query over the real 128-dim '
               'vectors, snapped to word lines) = 20 rows. Pipeline per row: keyword '
               'retrieval (doc-filtered) -&gt; region boxes -&gt; crop + OCR text -&gt; '
               'Gemini reader (grounded prompt with numbered OCR words, '
               'evidence_word_ids) -&gt; post-reader answer localization '
               '(derive_answer_words, fuzzy) -&gt; NLI entailment on crop text '
               '(DeBERTa, CUDA) -&gt; hand-weighted gate (localization is a gate '
               'signal; UNANSWERABLE auto-abstains). Metrics: EM / token-F1 vs gold; '
               'word-level P/R/F1 on (text, box) identity (primary localization '
               'metric); union-IoU + per-line IoU (secondary).', BODY))
story.append(Spacer(1, 3))
story.append(P('Gold set (spans verified present in both OCR texts):', H3))
story.append(table(
    [[P('<b>qid</b>', CELLH), P('<b>question</b>', CELLH),
      P('<b>gold answer</b>', CELLH), P('<b>why</b>', CELLH)]] +
    [[P(g['qid'], CELL), P(g['question'], CELL), P(f"<b>{g['answer']}</b>", CELL),
      P(g['why'], CELL)] for g in
     [{'qid': 'q1_title', 'question': 'What is the title of the document?',
       'answer': 'Sample Document for PDF Testing',
       'why': 'heading words, clean in both (exact match).'},
      {'qid': 'q2_emph', 'question': 'What kind of emphasis can italic text give?',
       'answer': 'subtle emphasis',
       'why': 'exact span in both; nearby png text garbled.'},
      {'qid': 'q3_list', 'question': 'What is the second entry of the ordered list?',
       'answer': 'Second item', 'why': 'exact in both.'},
      {'qid': 'q4_quote', 'question': 'What can a block quote be used to highlight?',
       'answer': 'important information',
       'why': "pdf exact; png typo 'impertant' (fuzzy)."},
      {'qid': 'q5_table', 'question': 'What is in Row 1, Column 3 of the table?',
       'answer': 'Row 1, Col 3', 'why': "pdf exact; png 'Row 1, Col3' (fuzzy)."}]],
    widths=[20 * mm, 60 * mm, 40 * mm, 50 * mm]))
story.append(P('Quota note (repro honesty): the first pass hit Gemini 503 high-demand '
               'and 429 quota errors on 14/20 rows; a paced <b>--resume</b> (25 s gaps) '
               'recovered 3, then the per-model bucket was exhausted. Probing the model '
               'list showed <b>gemini-2.5-flash-lite retired (404)</b>; Google\'s message '
               'points at <b>gemini-3.5-flash-lite</b>, whose fresh bucket completed the '
               'remaining 10 rows. Reader model per row is recorded in rows.json '
               '(reader_model) and the result table.', SMALL))
story.append(Spacer(1, 2))

# --- 3. per-question results ---
story.append(P('3. Per-question results (crop + overlay per run)', H2))
story.append(P('For every run: <b>crop</b> = evidence region fed to the reader; '
               '<b>overlay</b> = full page with the final per-line evidence boxes in '
               'green. Captions give answer, metrics and the localization source '
               '(word_ids = reader-cited OCR ids; fuzzy = post-reader text match). '
               'q1 cells show <i>[image missing]</i>: retrieval abstained before any '
               'crop was produced (see analysis).', BODY))
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
        loc = r.get('localize', {}) or {}
        story.append(P(
            f"<b>{r['version']}/{r['grounding']}</b> (reader {r.get('reader_model', '?')}) — "
            f"answer: <b>{r.get('answer')}</b> "
            f"{'(ABSTAINED: ' + r.get('abstain_reason', '') + ') ' if r.get('abstained') else ''}"
            f"| EM {r.get('em')} F1 {(r.get('f1') or 0):.2f} "
            f"IoU {r.get('iou')} wP/R/F1 {r.get('word_p')}/{r.get('word_r')}/{r.get('word_f1')} "
            f"| score {r.get('score')} "
            f"| localize: {loc.get('source', 'none')}/{loc.get('score', 0)} "
            f"words={loc.get('words', [])}", SMALL))
        story.append(table([
            [P('<b>crop (reader input)</b>', CELLH),
             P('<b>overlay (page + evidence boxes)</b>', CELLH)],
            [img(r.get('crop'), 85 * mm, 80 * mm),
             img(r.get('overlay'), 85 * mm, 80 * mm)],
        ], widths=[85 * mm, 85 * mm]))
        story.append(Spacer(1, 3))

# --- 4. analysis ---
story.append(P('4. Analysis', H2))
story.append(P('<b>Reader: 10 exact + 4 partial + 4 honest UNANSWERABLEs.</b> q2/q3/q5 '
               'answer EM=1 on both versions and both groundings (q2 heat/png excepted, '
               'see below). q4 answers "important information or citations" on all 4 '
               'cells (F1 0.67 vs the shorter gold — gold verbosity artifact, the extra '
               'words are correct). q1 abstains 4/4 <b>at retrieval, before any '
               'grounding</b>: "document" is on the keyword stopword list, so the only '
               'content token is "title" — which never occurs on the page — giving '
               'zero keyword overlap and the no-keyword-hit abstain path (score 0.0, '
               'no crop/overlay produced). Correct abstention, but it exposes a '
               'retrieval blind spot: a question whose only distinctive term is absent '
               'from the page can never be grounded, even though the title is the most '
               'prominent text on it.', BODY))
story.append(P('<b>Grounding gap: right answer, shifted evidence.</b> The grounded '
               'prompt returns consecutive, in-range word ids — but they are often '
               'shifted by a line: q3/pdf cites [item, 3.] instead of [Second, item]; '
               'q5/pdf cites [1,, Col, 3, Row] running into the next table row; '
               'q4/pdf-heat cites [information, or, citations., Table]. Only 3/16 '
               'answered rows reach IoU 1.0. The id/text cross-check (token-F1 >= 0.3) '
               'lets these through because cited words overlap the answer partially. '
               'Follow-up: raise the threshold (~0.6) or require the cited span text '
               'to contain the normalized answer.', BODY))
story.append(P('<b>Fuzzy localization wins on OCR noise.</b> q4/png gold match 0.952 '
               '("impertant") and q5/png gold match 0.952 ("Col3") localize without '
               'ids; q5/png-heat reaches IoU 1.0 / word-F1 1.0 via the fuzzy path. '
               'q2/png-heat is the complementarity case from testRun2: span grounding '
               'misses (UNANSWERABLE) while heatmap succeeds (EM=1).', BODY))
story.append(P('<b>Gate.</b> No correct answer was dropped at tau=0.5 this run; both '
               'UNANSWERABLE rows score 0.0-0.45 and abstain. Localization scores feed '
               'the gate (evidence_valid=True on all 16 answered rows).', BODY))
story.append(Spacer(1, 2))

# --- 5. result table ---
story.append(P('5. Result table (20 rows)', H2))


def short(a, n=26):
    a = '' if a is None else str(a)
    a = a.replace('<', '&lt;').replace('>', '&gt;')
    return a if len(a) <= n else a[:n - 1] + '…'


hdr = ['qid', 'ver', 'grnd', 'answer', 'EM', 'F1', 'IoU',
       'wP', 'wR', 'wF1', 'score', 'abst']
tdata = [[P(f'<b>{h}</b>', CELLH) for h in hdr]]
for r in rows:
    tdata.append([
        P(r['qid'].replace('_', '<br/>'), CELL), P(r['version'], CELL),
        P(r['grounding'], CELL), P(short(r.get('answer')), CELL),
        P(str(r.get('em')), CELL), P(f"{(r.get('f1') or 0):.2f}", CELL),
        P(str(r.get('iou')), CELL), P(str(r.get('word_p')), CELL),
        P(str(r.get('word_r')), CELL), P(str(r.get('word_f1')), CELL),
        P(str(r.get('score')), CELL),
        P('Y' if r.get('abstained') else '', CELL)])
story.append(table(tdata, widths=[24 * mm, 10 * mm, 14 * mm, 32 * mm, 8 * mm,
                                 10 * mm, 12 * mm, 10 * mm, 10 * mm, 10 * mm,
                                 14 * mm, 10 * mm]))
story.append(Spacer(1, 3))
n_em = sum(1 for r in rows if r.get('em') == 1.0)
n_f1 = sum(1 for r in rows if (r.get('f1') or 0) >= 0.5 and r.get('em') != 1.0)
n_abs = sum(1 for r in rows if r.get('abstained'))
story.append(P(f'Summary: <b>{n_em}/20 exact</b>, <b>{n_f1}/20 partial</b> '
               f'(q4 x4, F1 0.67), <b>{n_abs}/20 abstained</b> (q1 x4 honest + '
               f'q2 heat/pdf, q2 span/png honest). 0 errors after resume. '
               f'Artifacts: <b>{RUN.name}/</b> (rows.json + per-run crop/overlay PNGs).',
               BODY))

from reportlab.platypus import SimpleDocTemplate
doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                        leftMargin=15 * mm, rightMargin=15 * mm,
                        topMargin=15 * mm, bottomMargin=15 * mm,
                        title='Test Run 3', author='altVRAG')
doc.build(story)
print('WROTE', OUT, OUT.stat().st_size, 'bytes')
