"""Verify the Colab-built GPU index + functional checks on real ColQwen vectors.

Part A: integrity (manifest vs files, grid consistency, SQLite, Qdrant).
Part B: functional (self-retrieval, heatmap highlights, NLI verifier,
        scorer+gate) with patch-derived pseudo-queries, since the text
        query-side ColQwen encoder does not exist yet (documented gap).

Outputs: results/demo/index_verify_{stamp}/{verify.json, overlay.png,
         crop.png, verify.log}. All numbers also printed to stdout.
"""
import datetime as dt
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, r'C:\Users\bipra\OneDrive\Desktop\altVRAG\src')

REPO = Path(r'C:\Users\bipra\OneDrive\Desktop\altVRAG')
INDEX = REPO / 'data' / 'index'
STAMP = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
OUT = REPO / 'results' / 'demo' / f'index_verify_{STAMP}'
OUT.mkdir(parents=True, exist_ok=True)

report: dict = {'stamp': STAMP, 'checks': {}}


def check(name, ok, detail=''):
    report['checks'][name] = {'ok': bool(ok), 'detail': str(detail)}
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


t0 = time.time()
# ---------------- Part A: integrity ----------------
manifest = json.loads((INDEX / 'manifest.json').read_text())
stats = manifest['stats']
print('manifest:', json.dumps(manifest, indent=2)[:600])

npy_p = INDEX / 'embeddings' / 'testimg_0.npy'
jsp = INDEX / 'embeddings' / 'testimg_0.json'
side = json.loads(jsp.read_text())
vecs = np.load(npy_p)
gh, gw, dim = side['grid_h'], side['grid_w'], side['dim']
check('emb_files_exist', npy_p.exists() and jsp.exists())
check('grid_consistency', vecs.shape[0] == gh * gw,
      f'npy {vecs.shape} vs grid {gh}x{gw}={gh * gw}')
check('dim_128', vecs.shape[1] == 128 == dim, f'dim={vecs.shape[1]}')
check('dtype_float16', vecs.dtype == np.float16, str(vecs.dtype))
f32 = vecs.astype(np.float32)
norms = np.linalg.norm(f32, axis=1)
check('l2_normalized', abs(float(norms.mean()) - 1.0) < 0.02,
      f'mean_norm={norms.mean():.4f} std={norms.std():.4f}')
check('no_nan_inf', bool(np.isfinite(f32).all()))
check('special_dropped_11', side['n_special_dropped'] == 11,
      f"n_special_dropped={side['n_special_dropped']} (old (0,1,5) guess would have failed)")

con = sqlite3.connect(INDEX / 'meta.db')
pages = con.execute('SELECT doc_id,page,width,height,dpi,grid_h,grid_w,image_path FROM pages').fetchall()
nwords = con.execute('SELECT COUNT(*) FROM word_boxes').fetchone()[0]
check('sqlite_pages', len(pages) == stats['pages'] == manifest['sqlite_pages'], f'{pages}')
check('sqlite_words', nwords == stats.get('word_boxes', nwords) == manifest['word_boxes'], f'n={nwords}')
W, H = pages[0][2], pages[0][3]
check('meta_grid_match', (pages[0][5], pages[0][6]) == (gh, gw))
img_path = INDEX / 'pages' / 'testimg_p0.png'
check('page_png_exists', img_path.exists(), f'{img_path.stat().st_size / 1024:.0f} KB')

from vvrag.ingest import index as index_mod
try:
    client = index_mod.get_qdrant_client('local', str(INDEX / 'qdrant'))
except Exception as e:
    lock = INDEX / 'qdrant' / '.lock'
    if lock.exists():
        lock.unlink()  # stale Colab lock
        client = index_mod.get_qdrant_client('local', str(INDEX / 'qdrant'))
    else:
        raise
n_full = index_mod.count_points(client, 'pages_full')
n_pool = index_mod.count_points(client, 'pages_pooled')
check('qdrant_full', n_full == manifest['qdrant_full'] == 1, f'n={n_full}')
check('qdrant_pooled', n_pool == manifest['qdrant_pooled'] == gh + gw + 1,
      f'n={n_pool} (grid_h+grid_w+1={gh + gw + 1})')

# OCR text -> real claim + false claim for the verifier test
words = con.execute('SELECT text,x0,y0,x1,y1,block_id,line_id FROM word_boxes ORDER BY page,line_id,x0').fetchall()
lines: dict = {}
for t, x0, y0, x1, y1, b, ln in words:
    lines.setdefault((b, ln), []).append(t)
line_texts = [' '.join(v) for _, v in sorted(lines.items())]
full_text = '\n'.join(line_texts)
print(f'--- OCR: {nwords} words, {len(line_texts)} lines; first 3 lines:')
[print('   ', s[:100]) for s in line_texts[:3]]
# Complete-sentence claims only: NLI was verified on bare-fact sentences;
# a wrapped-line fragment ('...about the Department\\'s.') is ungrammatical
# and scores ~0 even when its words are all in the premise (checked 2026-10-15).
import re as _re
joined = ' '.join(line_texts).replace('�', '').replace('  ', ' ')
sents = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', joined) if s.strip()]
sents = [s for s in sents if len(s.split()) >= 6 and s.endswith(('.', '!', '?'))]
true_claim = next(s for s in sents if 'general information' in s)
false_claim = 'The quarterly revenue grew by 500 percent in Antarctica.'
print('true_claim :', true_claim[:120])
print('false_claim:', false_claim)

# ---------------- Part B: functional ----------------
from vvrag.retrieve.two_stage import TwoStageRetriever, retrieval_margin

rng = np.random.default_rng(0)
patch_idx = rng.choice(f32.shape[0], size=4, replace=False)
q = f32[patch_idx] + 0.15 * rng.standard_normal((4, dim)).astype(np.float32)
q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
retriever = TwoStageRetriever(client, top_k=3, prefetch_k_multiplier=4)
cands = retriever.prefetch(q)
hits = retriever.rerank(q, cands)
top = hits[0]
margin = retrieval_margin(hits) if len(hits) > 1 else 0.0
check('self_retrieval_top1', (top.doc_id, top.page) == ('testimg', 0),
      f'top={top.doc_id}/{top.page} score={top.score:.3f} margin={margin:.3f} (1-page corpus: path exercised)')
report['retrieval'] = {'top': [top.doc_id, top.page], 'score': float(top.score), 'margin': float(margin)}

from vvrag.ground.heatmap import build_heatmap, heat_concentration
from vvrag.ground.boxes import grid_to_boxes
from vvrag.ground.crop import crop_union
from vvrag.ground.text_in_box import extract_crop_text
from vvrag.schemas import WordBox

toks = ['supervision', 'report', 'aims', 'disclaimer']
qv = q[:len(toks)]
hm, norm = build_heatmap(qv, toks, f32, gh, gw, 'testimg', 0, mode='max')
boxes, mask = grid_to_boxes(norm, W, H)
in_bounds = all(0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H for x0, y0, x1, y1 in boxes)
check('heatmap_boxes', len(boxes) > 0 and in_bounds,
      f'n_boxes={len(boxes)} boxes={boxes}')
hc = heat_concentration(norm, mask)
print(f'heat_concentration={hc:.3f}')
report['grounding'] = {'boxes': boxes, 'heat_concentration': float(hc)}

from PIL import Image, ImageDraw
img = Image.open(img_path).convert('RGB')
dr = ImageDraw.Draw(img)
for x0, y0, x1, y1 in boxes:
    dr.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
img.save(OUT / 'overlay.png')
crop_path = crop_union(str(img_path), boxes, str(OUT / 'crop.png'))
wb = [WordBox('testimg', 0, r[0], (r[1], r[2], r[3], r[4]), r[5], r[6]) for r in words]
crop_text, has_text = extract_crop_text(wb, boxes, crop_path)
print(f'crop: {crop_path} has_text={has_text} chars={len(crop_text)}')
print('crop_text head:', crop_text[:200].replace('\n', ' | '))
check('crop_has_text', has_text and len(crop_text) > 0)

from vvrag.verify.nli import NLIEntailer, aggregate_entailment
nli = NLIEntailer(device='auto')
nli.load()
probs = nli.entail_probs(full_text, [true_claim, false_claim])
e_min, e_mean = aggregate_entailment([probs[0]])
print(f'NLI true_claim entail={probs[0]:.3f} | false_claim entail={probs[1]:.3f}')
check('nli_discriminates', probs[0] > 0.7 and probs[1] < probs[0],
      f'true={probs[0]:.3f} false={probs[1]:.3f}')
report['nli'] = {'true_entail': probs[0], 'false_entail': probs[1],
                 'agg_min': e_min, 'agg_mean': e_mean}

from vvrag.verify.features import build_features, to_vector
from vvrag.gate.scorer import hand_weighted
from vvrag.gate.fixed import apply_fixed
feats = build_features(agreement=1.0, entail_min=e_min, entail_mean=e_mean,
                       retrieval_margin=margin, heat_concentration=hc,
                       n_claims=1, crop_has_text=has_text)
score = hand_weighted(feats.agreement, feats.entail_prob, feats.heat_concentration)
answer = apply_fixed(score, 0.5)  # True = answer, False = abstain
print('features:', to_vector(feats))
print(f'hand score={score:.3f} fixed(tau=0.5) answer={answer}')
check('gate_scores', 0.0 <= score <= 1.0)
report['gate'] = {'features': to_vector(feats), 'hand_score': score, 'answer_fixed05': bool(answer)}

# Documented gap: 32-dim FakeQueryEmbedder cannot query the 128-dim index
from vvrag.retrieve.query_embed import FakeQueryEmbedder
fq, _ = FakeQueryEmbedder().embed_query('supervision report')
check('query_gap_dim_mismatch', fq.shape[1] == 32 != dim,
      f'fake_query_dim={fq.shape[1]} vs page_dim={dim} -> text query encoder still missing (known gap)')

report['elapsed_s'] = round(time.time() - t0, 1)
(OUT / 'verify.json').write_text(json.dumps(report, indent=2, default=str))
npass = sum(1 for c in report['checks'].values() if c['ok'])
print(f'\n==== {npass}/{len(report["checks"])} checks passed in {report["elapsed_s"]}s -> {OUT} ====')
