# Frozen 3 preregistration (written before any live run)

**Date frozen:** 2026-10-23
**Set:** `data/slidevqa_frozen3`
**Status:** frozen; no live model has been run on this set.

## Why this set exists

`frozen2` was used for a pilot and for one post-hoc Section-reference fix. It is
not an improvement set. Frozen 3 is a new deck-disjoint set built from
SlideVQA decks that appear in neither the development decks nor the six
`frozen2` decks. It is evaluated once with the frozen code/config below.

## Set construction (done before the run)

- Source: `Ahren09/SlideVQA` mirror, `train` split. The official
  `NTT-hil-insight/SlideVQA` dataset is gated.
- 25 decks / 500 pages / 120 gold QA items. The originally planned 12-deck,
  180-page subset was not achievable because the mirror's test split yielded
  only 49 QA items; the train split was used instead.
- 5 scanned train decks with zero QA were dropped; only decks with gold QA are
  retained.
- `data/frozen_exclude_all.json` excludes 8 development decks + 6 `frozen2`
  decks. `data/frozen_exclude_frozen3_test.json` additionally excludes the 12
  test-split decks that were scanned during construction.
- Gold audit was completed before the run:
  - 67 `ok`, 35 `short_answer`, 7 `ocr_typo`, 9 `partial`, 2 `image_only`,
    0 `page_mismatch` after correction.
  - Seven gold pages were corrected from the audit's answer-coverage best page;
    the original page and a note are stored per item.
  - 2 items are labelled `image-only`; the remaining 118 are labelled
    `single-hop`. No multi-hop claim is made.
- Probe set: 60 verified-absent questions:
  - near-miss 15, absent-attribute 13, entity-swap 22, false-bridge 10.
  - Every probe was checked against normalized OCR from the local index's
    `word_boxes` and `sentences`; any candidate whose check string appeared was
    dropped.
- Local index `data/index_frozen3_local` uses fake embeddings only for OCR,
  gold audit and probe verification. The real index is built from the frozen
  PDFs on Colab.

## Frozen assets

- `data/slidevqa_frozen3/gold.json`
- `data/slidevqa_frozen3/probes.json`
- `data/slidevqa_frozen3/frozen_meta.json` (SHA-256 hashes)
- `data/slidevqa_frozen3/deck_ids.json`
- `data/slidevqa_frozen3/questions_all.json` (180 unique questions)
- `config.frozen3.text.yaml`

Do not edit gold labels, probe labels, pages, deck list, or thresholds after
this point. If a defect is found, record it and report the run as-is; a changed
set is a new set, not this one.

## Frozen system

- Reader: Gemini `gemini-3.5-flash-lite`, one sample per question
  (`--n3 0`), multi-page context as configured.
- Examiner: Groq text-only, models tried in order:
  `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`.
  `examiner_evidence: text`, `claim_level_images: 0`,
  `independence_required: true`, `examiner_max_tokens: 2000` (700 was too
  small for gpt-oss reasoning JSON on a real claim).
- Deterministic citation precheck runs before the examiner and can block
  VERIFIED. A text-only examiner is not allowed to verify visual-only claims.
- Retrieval arms, same frozen index and reader:
  1. keyword
  2. hybrid (ColQwen visual + keyword RRF), full 180 questions with a
     precomputed query cache; no 24-item selection.
- Gate score is logged for display only; it is not in the decision path.

## Outcomes and reporting

- Gold denominator: 120. Probe denominator: 60.
- Primary gold metrics: answer coverage, exact match, containment, token-F1;
  verified precision; shown-but-wrong rate; withheld rate; page_hit@1/@3 and
  MRR; sentence-F1/IoU where evaluable.
- Probe metrics: nothing-found rate, answered rate, false-verified rate.
- Strict `verified` counts are text-verified only. Visual-verified rows are
  reported separately and never counted as strict.
- Partial-credit matching is report-only; strict EM/F1 and partial F1 are both
  shown.
- Report exact counts and Clopper-Pearson 95% intervals where applicable.
- Report failures, examiner unavailability, and retrieval misses; do not hide
  them behind means.

## Analysis rules

- No threshold, label, prompt, or config changes between the keyword and hybrid
  runs.
- No tuning on these outcomes and then calling the set held-out.
- If a live run is interrupted, resume the same rows; do not replace them.
- Report the result even if it misses the targets. The targets are goals, not
  filters.

## Pre-specified expected weaknesses

- 25 decks/500 pages is smaller and more heterogeneous than the eventual
  200-300 item test set.
- Text-only examination cannot verify chart/image-only claims.
- Single-page/multi-hop questions may remain retrieval or citation failures.
- Groq free-tier quota can make the examiner unavailable on some rows; those
  rows remain `not_confident`, not silently verified.

## Pre-run validation

- Full pytest: exit 0 after the frozen3 code changes.
- Offline mock keyword run on the local fake index: **180/180 rows, 0 errors**.
  A full hybrid-arm mock (180 rows; no cache -> keyword fallback) also
  completes with 0 errors. These are plumbing only; the mock answer is not a
  result.
- The mock run exposed and fixed a missing `per_line_iou` import in
  `scripts/run_test8.py` (29/180 errored rows before the fix, 0 after).
- `--check-cache --hybrid-all` reports `0/180` until the Colab bundle is
  extracted, as expected.

## Run record (filled after the run)

- Keyword run dir:
- Hybrid run dir:
- Code commit/status:
- Examiner models actually served:
- Deviations from this preregistration:
