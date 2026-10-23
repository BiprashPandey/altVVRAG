# TODO - what is left after the frozen pilot

Updated: 2026-10-23 (frozen3 built; no frozen3 live run yet)

This is the short, honest plan. Detailed evidence is in `FROZEN_RESULTS.md`,
`PREREGISTRATION_FROZEN3.md`, `frozen_meta.json`, and `PIPELINE.md`.

## Where we are now

- `frozen2` pilot: 6 decks / 120 pages / 28 gold / 19 probes; 47 rows, 0 errors.
  Gemini reader + same-family Gemini examiner; fake embeddings + keyword only.
- Pilot result: 19/28 gold answers shown, 6/28 verified, verified precision 5/6,
  15/19 probes nothing-found, 1 false-verified probe.
- The false-verified probe ("Old Act Section 4 in the New Act Section 12") was
  independently reproduced with Groq. The post-hoc `Section N` constraint makes
  it `not_confident`; it changes 0/28 pilot gold rows.
- `frozen3` is built and frozen **before any live run**:
  - 25 decks / 500 pages / 120 gold / 60 probes.
  - 67 ok, 35 short-answer, 7 OCR-typo, 9 partial, 2 image-only, 0 page
    mismatch after 7 pre-run page corrections.
  - Probe categories: near-miss 15, absent-attribute 13, entity-swap 22,
    false-bridge 10; all check strings verified absent from local OCR.
  - Local `data/index_frozen3_local` is **fake embeddings** and is only for
    audit/probe checks. The real ColQwen index and query cache do not exist
    yet. Local keyword baseline: page_hit@1 33.3%, @5 63.3%, MRR 0.461
    (`results/frozen3_local_keyword_validation.json`).
- Text-only independent examiner mode is implemented (`verify.examiner_evidence:
  text`, Groq model rotation; no images); one test covers it.
- `run_test8.py` supports full-set hybrid (`--hybrid-all`), custom
  `--query-cache`, `--n3 0`, and partial-credit reporting. `partial_answer_correct`
  is report-only; strict scoring is unchanged.
- `make_frozen_report.py` is generic (`--set-dir`, `frozen_meta.json`) and the
  full 180-row mock report runs with 0 errors; a full hybrid-arm mock (180
  rows, missing cache -> keyword fallback) also completes with 0 errors.
- Full pytest passes; the earlier live-run bug (`per_line_iou` missing import)
  was found by the frozen3 mock run and fixed.

## What is left, in priority order

### 1. Build the real index and query cache on Colab

**Why:** frozen3 has only fake page vectors locally. A real visual-retrieval
claim requires ColQwen page vectors and query vectors.

**What:** Run `notebooks/frozen3_colab.ipynb` on a T4 GPU. **No upload:** the
notebook clones `main` (code + frozen JSON set), checks the frozen hashes,
materializes the exact 25 decks, builds `data/index_frozen3`, encodes all 180
questions into `data/query_cache_frozen3`, validates probe absence, and
downloads `frozen3_gpu_bundle.zip`.

**Done when:** The bundle is extracted, `validate_retrieval.py` reports keyword
and hybrid page-hit/MRR, and the notebook assertions pass (25 decks, 500 pages,
60 probes absent, 180/180 query vectors).

### 2. Run frozen3 once with the independent text examiner

**Why:** This is the first fresh-set result after the post-hoc fixes, and the
first full independent-examiner result if Groq quota permits.

**What:** Run keyword 180 rows, then hybrid 180 rows, with frozen
`config.frozen3.text.yaml`, Gemini reader, and Groq text examiner rotation
(`gpt-oss-20b` -> `gpt-oss-120b` -> `qwen3.8-27b`). Use `--n3 0`; do not tune
between arms. Resume errors; do not replace rows.

**Done when:** Both runs have 0 errors, the examiner availability and per-model
errors are recorded, and `frozen_eval_frozen3.pdf` is built from the run dirs.

### 3. Report honestly at 120 gold, not 200-300

**Why:** 120 gold / 60 probes is useful but still fixed-set evidence with wide
intervals. The approved 12-deck/180-page target was not achievable from the
mirror; the set is 25 decks/500 pages.

**What:** Report coverage, EM, containment, F1, verified precision,
confident-wrong, probe answered/nothing-found/false-verified, page-hit/MRR,
partial-credit metrics, and CP95 intervals. State the size limitation and the
train-split/subset construction.

**Done when:** The report contains exact counts, intervals, retrieval mode used,
reader/examiner model names, unavailability, and all deviations.

### 4. Grow to 200-300 gold + 100+ probes only as a new set

**Why:** Do not extend frozen3 after seeing its results; that would make it a
tuning set.

**What:** Build `frozen4` from new decks with the same pre-run audit/probe
protocol, freeze hashes, and run once. Keep frozen3 as the prior fixed result.

**Done when:** A larger deck-disjoint set is frozen and run once, with the same
report contract.

### 5. Fix multi-hop and table citation units

**Why:** Multi-hop remains the dominant honest-abstention class; table cells can
split the answer from its qualifier. Adding pages alone previously regressed an
answer.

**What:** Add table-aware/structured citation units and a qualifier-aware check;
evaluate on a **new** frozen set, not frozen3.

**Done when:** Multi-hop/table results improve on a fresh set without increasing
false-verified probes.

### 6. Keep partial-answer scoring explicit

**Why:** One pilot verified answer was a correct fragment of a longer gold
answer. Strict matching called it wrong.

**What:** Keep strict EM/F1 as primary; report partial-credit separately (already
implemented). Do not silently add acceptable answers after seeing results.

**Done when:** Every report shows strict and partial counts side by side and
states the gold rule.

### 7. Prove the visual-claim path separately

**Why:** The text-only examiner cannot verify chart/image-only claims. frozen3
has only 2 image-only gold items, so it cannot establish visual-tier gains.

**What:** After the text-only full run, run a targeted image-examiner subset on
image-only/chart rows if Groq quota permits. Keep `visual_verified` separate
from `verified`.

**Done when:** The report gives visual-verified counts/precision and examples,
or explicitly says the visual tier was not evaluated.

### 8. Clean up and commit when asked

**Why:** The working tree still has old logs, PDFs, zips, and uncommitted code.

**What:** Remove accidental files, record code/config/data hashes, keep pilot
pre-fix numbers separate from post-fix checks, and commit only when the user
asks. Never commit credentials, data, results, PDFs, or zips.

**Done when:** A new machine can reproduce frozen3 from the documented
commands/hashes.

## Rules

- Do **not** reuse `frozen2` or `frozen3` to show improvement after a fix.
- Do **not** change labels, probes, thresholds, or prompts between arms.
- Do **not** call a same-family examiner "independent".
- Do **not** present 120 + 60 as a final universal estimate.
- Do **not** mix text-verified and visual-verified numbers.
- If quota removes rows, report the holes; do not silently drop them.

## Commands

```bash
# full test suite (do not run during a live run)
python -m pytest -q

# frozen3 mock plumbing (local fake index; no API)
VVRAG_CONFIG=config.frozen3.text.yaml python scripts/run_test8.py \
  --mock --index data/index_frozen3_local \
  --gold data/slidevqa_frozen3/gold.json \
  --probes data/slidevqa_frozen3/probes.json \
  --arms keyword --models gemini-3.5-flash-lite --n3 0

# rebuild the frozen report from a real run
VVRAG_CONFIG=config.frozen3.text.yaml python scripts/make_frozen_report.py \
  <run_dir> frozen_eval_frozen3.pdf \
  --set-dir data/slidevqa_frozen3 --config config.frozen3.text.yaml

# live after Colab bundle extraction
VVRAG_CONFIG=config.frozen3.text.yaml python scripts/run_test8.py \
  --index data/index_frozen3 \
  --gold data/slidevqa_frozen3/gold.json \
  --probes data/slidevqa_frozen3/probes.json \
  --arms keyword --models gemini-3.5-flash-lite --n3 0
VVRAG_CONFIG=config.frozen3.text.yaml python scripts/run_test8.py \
  --index data/index_frozen3 \
  --gold data/slidevqa_frozen3/gold.json \
  --probes data/slidevqa_frozen3/probes.json \
  --arms hybrid --hybrid-all --models gemini-3.5-flash-lite \
  --query-cache data/query_cache_frozen3 --n3 0
```

## One-sentence summary

frozen3 is built and frozen (25 decks, 120 gold, 60 verified-absent probes);
the next real result is the Colab-built visual index plus one independent
keyword/hybrid run, reported at its true size.
