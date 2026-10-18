"""testRun4 matrix: same 5 questions x {pdf,png} x {span,heatmap}, NEW pipeline.

Like testRun3 (same index, same gold spans) but end-to-end on the
sentence-citing + verified-gate refactor (context.md):
  - retrieval never abstains; abstain only at the gate (last resort);
  - reader cites SENTENCES with a verbatim quote (retry once on mismatch);
  - deterministic quote check + statement NLI on the cited sentences;
  - final boxes come from cited sentences (fuzzy fallback, else region);
  - gate = CALIBRATED (data/gate/scorer.pkl + tau.json from
    scripts/calibrate_gate.py; verifier rescue applies).

Reader: live Gemini (n_samples=1). Errors recorded, never fatal.
Writes results/demo/testrun4_{stamp}/{rows.json, <qid>_<ver>_<grounding>/
{crop.png (reader input), overlay.png (page + final evidence boxes)}.
"""
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
from vvrag.eval.metrics import exact_match, f1, center_inside  # noqa: E402
from vvrag.eval.metrics import per_line_iou, sentence_precision_recall, union_iou, word_precision_recall_f1  # noqa: E402
from vvrag.ground.crop import overlay_heatmap  # noqa: E402
from vvrag.ground.text_in_box import words_in_boxes  # noqa: E402
from vvrag.ingest.sentences import sentence_ids_for_answer, sentences_from_page  # noqa: E402
from vvrag.schemas import WordBox  # noqa: E402
from vvrag.ui.qa import answer_question, page_words  # noqa: E402

GOLD = [    {"qid": "q1_title", "question": "What is the title of the document?",
     "answer": "Sample Document for PDF Testing",
     "why": "heading words, clean in both versions (exact match)."},
    {"qid": "q2_emph", "question": "What kind of emphasis can italic text give?",
     "answer": "subtle emphasis",
     "why": "exact span in both; nearby png text is garbled ('forties of')."},
    {"qid": "q3_list", "question": "What is the second entry of the ordered list?",
     "answer": "Second item",
     "why": "exact in both; surrounding png list lines are noisy."},
    {"qid": "q4_quote", "question": "What can a block quote be used to highlight?",
     "answer": "important information",
     "why": "pdf exact; png has OCR typo 'impertant' (fuzzy match)."},
    {"qid": "q5_table", "question": "What is in Row 1, Column 3 of the table?",
     "answer": "Row 1, Col 3",
     "why": "pdf exact; png reads 'Row 1, Col3' (fuzzy match)."},
]

VERSIONS = [("pdf", "sample"), ("png", "sample_page")]

# Free-tier quota is ~5 req/min: pace live calls so resume runs survive.
LIVE_CALL_GAP_S = 25.0
_last_live_call = [0.0]


def paced_answer_question(*args, **kwargs):
    """Sleep between live-reader calls to stay under the per-minute quota."""
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


def run_single(item, ver, doc_id, grounding, words, gbox, gwords,
               grow, wdir, img_path, model=""):
    """One matrix cell -> row dict (success or recorded error)."""
    try:
        res = paced_answer_question(
            item['question'], str(INDEX), str(wdir), reader_mode='gemini',
            provider='gemini', model=model, n_samples=1, grounding=grounding,
            gate='calibrated', doc_id=doc_id)
    except Exception as e:  # noqa: BLE001 - record, don't stop
        print(f'    {grounding} ERROR: {type(e).__name__}: {str(e)[:100]}')
        row = dict(grow)
        row.update({'grounding': grounding,
                    'error': f'{type(e).__name__}: {e}'})
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
    # suggestion #1/#6: sentence-citation precision/recall vs gold sentence ids
    pred_sids = ev.get('sentence_ids', [])
    gold_sids = grow.get('gold_sentence_ids', [])
    s_p, s_r, s_f1 = sentence_precision_recall(pred_sids, gold_sids)
    loc = ev.get('localize', {})
    # stable image names for the PDF report
    crop_p = wdir / 'crop.png'
    ov_p = wdir / 'overlay.png'
    crops = sorted(wdir.glob('*_crop.png'))
    if crops:
        shutil.copy(crops[0], crop_p)
    # overlay.png = full page + FINAL evidence boxes (from cited sentences)
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
        'answer': res.get('answer'), 'abstained': res.get('abstained'),
        'abstain_reason': res.get('abstain_reason', ''),
        'score': res.get('score'), 'tau': res.get('tau'),
        'em': exact_match(res.get('answer') or '', item['answer']),
        'f1': f1(res.get('answer') or '', item['answer']),
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
        'word_p': round(wp, 3), 'word_r': round(wr, 3),
        'word_f1': round(wf1, 3),
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
    print(f"    {grounding}: ans={row['answer']!r:.60} "
          f"EM={row['em']} F1={row['f1']:.2f} IoU={row['iou']:.2f} "
          f"wP/R/F1={row['word_p']}/{row['word_r']}/{row['word_f1']} "
          f"sP/R/F1={row['sent_p']}/{row['sent_r']}/{row['sent_f1']} "
          f"quote={row['quote_ok']} score={row['score']} tau={row['tau']} "
          f"abst={row['abstained']} ({row['abstain_reason']})")
    return row


def gold_context(con, item, doc_id):
    words = page_words(con, doc_id, 0)
    gbox = derive_gold_box(item['answer'], words)
    gwords, gscore = derive_answer_words(item['answer'], words)
    sents, _ = sentences_from_page(con, doc_id, 0)
    gsent_ids = sentence_ids_for_answer(item['answer'], sents)
    grow = {'qid': item['qid'], 'version': None, 'doc_id': doc_id,
            'question': item['question'], 'gold_answer': item['answer'],
            'gold_box': list(gbox) if gbox else None,
            'gold_words': [w.text for w in gwords],
            'gold_sentence_ids': gsent_ids,
            'gold_match_score': round(gscore, 4)}
    return words, gbox, gwords, grow


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == '--resume':
        resume(Path(sys.argv[2]))
        return
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = REPO / 'results' / 'demo' / f'testrun4_{stamp}'
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
            print(f'--- {item["qid"]}/{ver}: gold box={grow["gold_box"]} '
                  f'words={len(gwords)} match={grow["gold_match_score"]:.3f} '
                  f'gold_sents={grow["gold_sentence_ids"]}')
            for grounding in ('span', 'heatmap'):
                wdir = out / f'{item["qid"]}_{ver}_{grounding}'
                rows.append(run_single(item, ver, doc_id, grounding, words,
                                       gbox, gwords, grow, wdir, img_path))
    (out / 'rows.json').write_text(json.dumps(rows, indent=2, default=str), encoding='utf-8')
    print(f'\nWROTE {out / "rows.json"} ({len(rows)} rows)')


def resume(out: Path) -> None:
    """Re-run only rows that recorded an API error (quota/503), paced.

    Optional 2nd arg: reader model override (fresh per-model quota bucket),
    e.g. `run_test4.py --resume <dir> gemini-3.5-flash-lite`.
    """
    model = sys.argv[3] if len(sys.argv) > 3 else ""
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
    (out / 'rows.json').write_text(json.dumps(rows, indent=2, default=str), encoding='utf-8')
    n_err = sum(1 for r in rows if 'error' in r)
    print(f'\nRESUME DONE: fixed {fixed}, still errored {n_err}/{len(rows)}')


if __name__ == '__main__':
    main()
