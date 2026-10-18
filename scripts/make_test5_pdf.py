"""Build testRun5.pdf: narrative + per-run images + result table.

testRun5 = testRun4 matrix PLUS an unanswerable probe (q6), equivalent golds
for q4/q5 (P3), and every Phase A-F improvement live: OCR-tolerant quote check,
one UNANSWERABLE retry with closest passages, boxes tightened to the quote
(refine_box_to_quote), viewer quote tooltips, calibrated gate.

Usage: python scripts/make_test5_pdf.py [rows_dir] [out_pdf]
Default rows_dir: newest results/demo/testrun5_*/ (fall back to --resume dirs).
"""
import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_cands = sorted((REPO / 'results' / 'demo').glob('testrun5_*'))
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else (_cands[-1] if _cands else None)
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / 'testRun5.pdf'
INDEX = REPO / 'data' / 'index'

sys.path.insert(0, str(REPO / 'src'))

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image as RLImage, Paragraph, Spacer,
                                Table, TableStyle)

from vvrag.ground.text_in_box import join_words
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


if RUN is None or not (RUN / 'rows.json').exists():
    sys.exit('No testrun5 rows.json found — run scripts/run_test5.py first.')

rows = json.loads((RUN / 'rows.json').read_text(encoding='utf-8'))
tau_meta = json.loads((REPO / 'data' / 'gate' / 'tau.json').read_text(encoding='utf-8'))
con = sqlite3.connect(INDEX / 'meta.db')
page_text, page_nwords = {}, {}
for doc in ('sample', 'sample_page'):
    words = page_words(con, doc, 0)
    page_text[doc] = join_words(words)
    page_nwords[doc] = len(words)

n_err = sum(1 for r in rows if 'error' in r)
ans = [r for r in rows if not r.get('is_unanswerable') and 'error' not in r]
una = [r for r in rows if r.get('is_unanswerable') and 'error' not in r]
n_equiv = sum(1 for r in ans if r.get('equiv_correct'))
n_abs_ans = sum(1 for r in ans if r.get('abstained'))
n_abs_una = sum(1 for r in una if r.get('abstained'))
answered = [r for r in ans if not r.get('abstained')]
avg_sf1 = sum(r.get('sent_f1') or 0 for r in answered) / max(1, len(answered))
avg_wr = sum(r.get('word_r') or 0 for r in answered) / max(1, len(answered))
n_quote_ok = sum(1 for r in answered if r.get('quote_ok'))
equiv_rate = n_equiv / max(1, len(ans))
false_abs = n_abs_ans / max(1, len(ans))
una_rate = n_abs_una / max(1, len(una))

story = []
story.append(P('Test Run 5 — full improvement stack: equivalent golds, '
               'OCR-tolerant quotes, unanswerable probe, quote-refined boxes', H1))
story.append(P(f"Date: {dt.datetime.now().strftime('%Y-%m-%d')}. Same Colab index "
               f"<b>data/index</b> and the 5 gold questions from testRun4, extended "
               f"with (a) <b>equivalent golds</b> for q4/q5 (P3: any acceptable "
               f"variant counts as correct), (b) an <b>unanswerable probe</b> "
               f"(q6: CEO salary — must abstain), (c) the Phase A-F stack running "
               f"live: OCR-tolerant quote check (verify.ocr_normalize), one "
               f"<b>UNANSWERABLE retry</b> with closest passages "
               f"(reader.retry_on_unanswerable), final boxes tightened to the "
               f"quote via <b>refine_box_to_quote</b>, and the "
               f"<b>calibrated gate</b> (tau={tau_meta['tau']}). Reader: live "
               f"Gemini <b>gemini-3.5-flash</b> (n_samples=1), paced 25 s/call "
               f"with --resume after quota errors. Runner: <b>scripts/run_test5.py"
               f"</b>; report: <b>scripts/make_test5_pdf.py</b>. Rows: "
               f"<b>{RUN.name}/rows.json</b> ({len(rows)} rows, {n_err} errors).", BODY))
story.append(Spacer(1, 4))

story.append(P('1. What changed since testRun4', H2))
story.append(P('1. <b>Equivalent golds (P3)</b>: QAItem gained acceptable_answers; '
               'a row is equiv-correct if EM, token-F1 &ge; 0.8, or the gold is a '
               'token-subset of the answer (equivalent_correct in eval/metrics). '
               "q4's longer citation form \"important information or citations\" "
               "and q5's OCR typo \"Row 1, Col3\" now score as correct.<br/>"
               '2. <b>OCR-tolerant quote check</b>: quotes are alnum/whitespace-'
               'normalized before the fuzzy match (verify.ocr_normalize), so '
               "garbled page text like \"Col3\" no longer fails verification by "
               'punctuation alone.<br/>'
               '3. <b>UNANSWERABLE retry</b>: when the reader abstains, the pipeline '
               'retries ONCE with the top-5 closest passages '
               '(reader.retry_on_unanswerable, overlap &ge; 0.2) before giving up; '
               'the viewer still shows closest passages on a final abstention.<br/>'
               '4. <b>Quote-refined boxes (P4)</b>: cited-sentence boxes are '
               'tightened to the words matching the quote (refine_box_to_quote), '
               'so final evidence boxes hug the answer instead of whole '
               'sentences; falls back to the sentence box if no word matches.<br/>'
               '5. <b>Abstention metrics</b>: report splits abstention into '
               'false_abstention_rate (answerable rows that abstained) and '
               'unanswerable_abstention_rate — q6 probes the latter directly.<br/>'
               '6. <b>Viewer</b>: answer quote displayed, per-sentence boxes carry '
               'quote tooltips; reader 503/quota backoff + text-layer-first OCR '
               'were already in place from earlier phases.', BODY))
story.append(Spacer(1, 3))
story.append(P('2. Corpus and gold set', H2))
story.append(P('Unchanged index (sample.pdf text layer + sample.png OCR twin, '
               '128-dim ColQwen vectors, sentence migration applied). Gold set:', BODY))
golds = [
    ('q1_title', 'What is the title of the document?', 'Sample Document for PDF Testing', '—'),
    ('q2_emph', 'What kind of emphasis can italic text give?', 'subtle emphasis', '—'),
    ('q3_list', 'What is the second entry of the ordered list?', 'Second item', '—'),
    ('q4_quote', 'What can a block quote be used to highlight?', 'important information',
     '"…or citations" variant accepted'),
    ('q5_table', 'What is in Row 1, Column 3 of the table?', 'Row 1, Col 3',
     '"Col3" OCR typo accepted'),
    ('q6_absent', "What is the CEO's annual salary in this document?", 'UNANSWERABLE',
     'must ABSTAIN (false-answer if shown)'),
]
story.append(table(
    [[P('<b>qid</b>', CELLH), P('<b>question</b>', CELLH),
      P('<b>gold</b>', CELLH), P('<b>equivalence / probe</b>', CELLH)]] +
    [[P(q, CELL), P(question, CELL), P(f'<b>{gold}</b>', CELL), P(note, CELL)]
     for q, question, gold, note in golds],
    widths=[22 * mm, 68 * mm, 40 * mm, 42 * mm]))

story.append(P('3. Per-question results (crop + final-overlay per run)', H2))
story.append(P('For every run: <b>crop</b> = region fed to the reader; '
               '<b>overlay</b> = full page with FINAL quote-refined evidence '
               'boxes in green. sP/sR/sF1 = sentence-citation precision/recall/F1 '
               'vs gold sentence ids; equiv = equivalent-correct (any acceptable '
               'gold). q6 rows expect abstained=True.', BODY))
by_q = {}
for r in rows:
    by_q.setdefault(r['qid'], []).append(r)
for qid in ['q1_title', 'q2_emph', 'q3_list', 'q4_quote', 'q5_table', 'q6_absent']:
    if qid not in by_q:
        continue
    qrows = sorted(by_q[qid], key=lambda r: (r['version'], r['grounding']))
    story.append(P(f"{qid}: {qrows[0]['question']}", H3))
    story.append(P(f"Gold: <b>{qrows[0]['gold_answer']}</b>", BODY))
    for r in qrows:
        if 'error' in r:
            story.append(P(f"<b>{r['version']}/{r['grounding']}</b> — ERROR: "
                           f"{esc(r['error'], 120)}", SMALL))
            continue
        loc = r.get('localize', {}) or {}
        samples = r.get('samples') or [{}]
        quote = samples[0].get('quote', '') or ''
        cited = r.get('cited_sentences') or []
        cited_txt = '; '.join(f"#{c['sent_id']} \u201c{esc(c['text'], 70)}\u201d"
                              for c in cited) if cited else '(none)'
        story.append(P(
            f"<b>{r['version']}/{r['grounding']}</b> (reader {esc(r.get('reader_model'), 30)}) — "
            f"answer: <b>{esc(r.get('answer'))}</b> "
            f"{'(ABSTAINED: ' + esc(r.get('abstain_reason', '')) + ') ' if r.get('abstained') else ''}"
            f"| equiv {r.get('equiv_correct')} (F1 {r.get('equiv_f1') or 0:.2f}) "
            f"| sP/R/F1 {r.get('sent_p')}/{r.get('sent_r')}/{r.get('sent_f1')} "
            f"wP/R/F1 {r.get('word_p')}/{r.get('word_r')}/{r.get('word_f1')} "
            f"IoU {r.get('iou')} "
            f"| score {r.get('score')} vs tau {r.get('tau')} "
            f"| quote_ok {r.get('quote_ok')} ({loc.get('score', '')}) "
            f"| quote: \u201c{esc(quote, 80)}\u201d", SMALL))
        story.append(P(f"cited sentences: {cited_txt}", SMALL))
        if r.get('abstained') and r.get('closest_passages'):
            story.append(P('closest passages: '
                           + '; '.join(f"\u201c{esc(p['text'], 80)}\u201d"
                                       for p in r['closest_passages'][:3]), SMALL))
        story.append(table([
            [P('<b>crop (reader input)</b>', CELLH),
             P('<b>overlay (page + final evidence boxes)</b>', CELLH)],
            [img(r.get('crop'), 85 * mm, 80 * mm),
             img(r.get('overlay'), 85 * mm, 80 * mm)],
        ], widths=[85 * mm, 85 * mm]))
        story.append(Spacer(1, 3))

story.append(P('4. Analysis vs Test Run 5 targets', H2))
story.append(P(f'<b>Targets: &ge;90% equiv-correct on answerable rows, &lt;5% false '
               f'abstention, &ge;90% abstention on unanswerable, every shown answer '
               f'quote+box verified.</b> Observed: equiv_correct '
               f'<b>{equiv_rate:.3f}</b> ({n_equiv}/{len(ans)}), false_abstention '
               f'<b>{false_abs:.3f}</b> ({n_abs_ans}/{len(ans)}), unanswerable '
               f'abstention <b>{una_rate:.3f}</b> ({n_abs_una}/{len(una)}), '
               f'quote_ok on answered rows {n_quote_ok}/{len(answered)}, mean '
               f'sentence-F1 {avg_sf1:.2f}, mean word recall {avg_wr:.2f}, '
               f'{n_err} errors.', BODY))
story.append(P('<b>Reading the deltas vs testRun4.</b> Equivalent golds convert '
               'q4\'s longer citation form and q5\'s OCR typo from "partial" to '
               'correct without loosening verification: the quote must still '
               'match the page (now OCR-normalized) and the boxes still come '
               'from cited sentences, tightened to the quote. The unanswerable '
               'retry gives the reader one second chance with closest passages '
               'before the gate abstains, trading one extra API call per false '
               'abstention for higher answerable coverage. q6 must abstain on '
               'BOTH versions and BOTH groundings (4/4); any shown answer there '
               'counts as a false answer and is called out in the table.', BODY))
story.append(Spacer(1, 2))

story.append(P('5. Result table', H2))
hdr = ['qid', 'ver', 'grnd', 'answer', 'equiv', 'sF1', 'quote',
       'score', 'tau', 'abstain reason']
tdata = [[P(f'<b>{h}</b>', CELLH) for h in hdr]]
for r in rows:
    if 'error' in r:
        tdata.append([P(r['qid'].replace('_', '<br/>'), CELL), P(r['version'], CELL),
                      P(r['grounding'], CELL), P('ERROR', CELL)] +
                     [P('', CELL) for _ in range(6)])
        continue
    tdata.append([
        P(r['qid'].replace('_', '<br/>'), CELL), P(r['version'], CELL),
        P(r['grounding'], CELL), P(esc(r.get('answer'), 24), CELL),
        P('Y' if r.get('equiv_correct') else ('—' if r.get('is_unanswerable') else 'N'), CELL),
        P(str(r.get('sent_f1')), CELL),
        P('Y' if r.get('quote_ok') else ('N' if r.get('quote_ok') is not None else '-'), CELL),
        P(str(r.get('score')), CELL), P(str(r.get('tau')), CELL),
        P(esc(r.get('abstain_reason'), 22), CELL)])
story.append(table(tdata, widths=[22 * mm, 10 * mm, 14 * mm, 32 * mm, 10 * mm,
                                  10 * mm, 12 * mm, 14 * mm, 10 * mm, 28 * mm]))
story.append(Spacer(1, 3))
story.append(P(f'Summary: <b>{n_equiv}/{len(ans)} equiv-correct</b> answerable, '
               f'<b>{n_abs_ans} false abstentions</b>, '
               f'<b>{n_abs_una}/{len(una)} unanswerable abstained</b>, '
               f'{n_quote_ok}/{len(answered)} answered rows quote-verified, '
               f'{n_err} errors. Artifacts: <b>{RUN.name}/</b> (rows.json + '
               f'per-run crop/overlay PNGs); gate: <b>data/gate/scorer.pkl + '
               f'tau.json</b>.', BODY))

from reportlab.platypus import SimpleDocTemplate
doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                        leftMargin=15 * mm, rightMargin=15 * mm,
                        topMargin=15 * mm, bottomMargin=15 * mm,
                        title='Test Run 5', author='altVRAG')
doc.build(story)
print('WROTE', OUT, OUT.stat().st_size, 'bytes')
