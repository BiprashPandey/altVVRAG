# Improvement plan after frozen3 / report-parity

Date: 2026-10-26
Status: proposal; nothing here has been run live yet.
Evidence base: `REPORT_COMPARISON.md`, `FROZEN_RESULTS.md`,
`results/report_parity_frozen3/summary.json`, and
`results/demo/testrun8_20261026_110734/rows.json`.

## Why this plan exists

The report-parity comparison is useful but is not an apples-to-apples quality
comparison: different decks, questions, readers, examiners, gold boxes and
verification contracts. The submitted report also excludes 18 verifier errors;
our run completed 180/180 rows with 0 errors.

The strongest diagnosis from our own rows is that **retrieval is the dominant
bottleneck**, not the reader:

| Slice | Value |
|---|---:|
| Page hit @1 | 63/120 = 52.5% |
| Page hit @3 | 85/120 = 70.8% |
| Reader abstentions caused by retrieval miss | 17/20 |
| Verified-but-wrong rows caused by retrieval miss | 6/7 |
| Verified precision when the gold page was retrieved | 12/13 = 92.3% |
| Verified precision when it was not retrieved | 9/15 = 60.0% |
| False-verified probes | 0/60 |

So the first improvement target is retrieval. The second is examiner parity
(visual qwen vs our text examiner). Reader substitution is a real confound but
is unlikely to explain most of the gap: 17 of 20 reader abstentions were on
wrong retrieved pages, and forcing the reader to answer them would mostly add
wrong answers.

## Non-negotiable rules

1. Do not tune on `frozen3`; it has been used. Build a fresh `frozen4` for any
   improvement claim.
2. Do not silently switch reader or examiner models. Record `served_by`, the
   requested model, and every fallback/error per row.
3. Keep `verified` and `visual_verified` separate. Do not mix text-verified and
   visual-verified precision.
4. Report both **product-shown** (verified + not_confident) and **strict
   verified**. Do not compare strict `verified` to the submitted report's
   supported-window Verified arm as if the definitions were the same.
5. Report all errors and holes; never drop quota-failed rows silently.
6. Report page-hit-only localization as well as all-row IoU.
7. Freeze code, config, prompts and data hashes before the frozen4 run.

## P0 — Retrieval first (highest leverage)

### Diagnosis

- Page hit @1 is 52.5%; @3 is 70.8%.
- 17/20 reader abstentions and 6/7 verified-wrong rows are retrieval misses.
- On page-hit rows, the existing strict verifier already reaches 12/13 =
  92.3% verified precision, close to the submitted report's 91.1%.
- The current hybrid arm helps, but it still leaves a large page-hit gap. On
  frozen3, hybrid raised page-hit@1 from 42.5% to 51.7% and page-hit@3 from
  55.8% to 70.8%, so there is still room.

### Tasks

1. **Build a retrieval-only dev benchmark.** Use the 8 dev decks and the
   existing dev gold/probes; evaluate page-hit@1/@3 and MRR only. Do not use
   frozen3 or frozen4 for tuning.
2. **Ablate fusion.** Compare keyword-only, visual-only, and hybrid RRF with
   different fusion weights and top-k. Record the visual score margin.
3. **Improve ranking.** Try query expansion / multi-query retrieval, a
   second-stage page reranker (lexical or cross-encoder over OCR text), and a
   visual-score-aware reranker. Keep the best two candidates for frozen4.
4. **Use more context deliberately.** Test `max_pages` and `rerank_top_k`
   combinations. Do not simply append bridge pages: the earlier attempt
   changed one honest `nothing_found` into a wrong answer.
5. **Fix OCR/index quality.** Re-OCR zero-word and low-text pages on a GPU or
   Colab; keep `index_manifest` provenance; verify page/word counts. Evaluate
   whether OCR quality explains page misses on specific decks.
6. **Add table/structured units.** Multi-hop and table questions still fail
   because `12.5 bn` is split from its row label. Extend table-row/cell units
   and use them for retrieval and citation, not just for display.
7. **Multi-hop decomposition.** For qualifier questions, retrieve the
   qualifier page explicitly (question decomposition or a generated subquery)
   instead of relying on a single page score.

### Acceptance for P0

Before frozen4, pre-register a target on the dev retrieval set. Proposed:

- page-hit@1 >= 65% and page-hit@3 >= 80% on the frozen4 sample;
- no drop in verified precision on page-hit rows;
- no increase in false-verified probes;
- every retrieval mode change recorded per row.

Files: `src/vvrag/retrieve/fusion.py`, `src/vvrag/retrieve/rerank.py`,
`src/vvrag/retrieve/query_vectors.py`, `src/vvrag/ui/qa.py`,
`scripts/validate_retrieval.py`, `scripts/precompute_query_vectors.py`.

### Phase 0 implementation status (2026-10-27)

- Added `retriever.keyword_weight`, `retriever.visual_weight`, and
  `retriever.candidate_k` (dataclass defaults `1.0`, `1.0`, `5`; YAML uses
  `candidate_k: 10`).
- `ui/qa.py` now passes fusion weights to `rrf_fuse`, expands the candidate
  pool before reranking/reader selection, and records `keyword_weight`,
  `visual_weight`, `candidate_k`, `rerank_k`, `reader_pages`,
  `candidate_pool_size`, and `prompt_pages` in `retrieval_debug`.
- `reader/prompts.py` records per-page prompt inclusion/truncation through
  `stats_out`; `tests/test_prompt_stats.py` covers full and truncated prompts.
- `retrieve/fusion.py` rejects negative RRF weights; `tests/test_phase4_retrieval.py`
  covers weighted fusion.
- New `scripts/tune_retrieval.py`: deterministic keyword/visual/hybrid sweep
  over fusion weights, RRF `k`, candidate `k`, and reranker off/lexical/both.
  Writes `rows.json` (meta + summary + rows) and `summary.csv`.
- New single self-contained Colab notebook:
  `notebooks/phase0_dev_retrieval_colab.ipynb`. It auto-downloads the
  160-page dev index from the public GitHub Release, verifies SHA256,
  encodes all 81 dev questions itself (incremental cache reuse), runs the
  full sweep, and downloads a compact bundle. There is no separate
  query-vector notebook.
- Public index release:
  `https://github.com/BiprashPandey/altVVRAG/releases/tag/phase0-dev-index-v1`
  (`slidevqa_index.zip`, 134,122,541 bytes, SHA256
  `d30ad0de69d8a1cd8de6a7a22abc0906d4a866d06b37334afad37c34d5cfa226`).
  It is a release asset rather than a repo file because GitHub rejects files
  over 100 MB.
- Offline smoke only (not evidence): on the 24 dev questions with cached
  query vectors, visual-only is hit@1/hit@3/hit@5/MRR =
  0.542/0.750/0.792/0.654; equal-weight hybrid is
  0.292/0.542/0.708/0.474; keyword is 0.125/0.417/0.542/0.321. This matches
  the frozen3 validation direction. Full 40-question dev comparison still
  needs the 81-query cache from Colab; do not change production defaults yet.

## P1 — Visual examiner parity

### Why

The report-parity run used a text-only Groq examiner because qwen vision hit
its tokens-per-day limit. That is not the production path and is not the
submitted report's verifier. It produced 0 visual-verified rows and cannot
verify chart/image-only claims.

### Tasks

1. Run `qwen/qwen3.8-27b` (or the same `qwen/qwen3.6-27b` as the submitted
   report for exact parity) with `examiner_evidence: page_and_region`,
   `claim_level: true`, and `claim_level_images: 1`.
2. Use a paid quota or a self-hosted endpoint; do not silently fall back to a
   text examiner and call it visual.
3. Run a targeted image-only/chart subset first. Report visual-verified counts,
   visual-verified precision, and visual-verified probe behaviour separately
   from strict text verification.
4. Record examiner availability, per-model errors, and evidence mode per row.

### Acceptance for P1

- examiner availability >= 95% on answered gold rows;
- visual-verified precision reported with a denominator and CP95 interval;
- 0 false-verified probes in the targeted subset;
- no row labelled visual unless `visual_verified` is true.

Files: `config.yaml`, `config.report_parity.yaml`, `src/vvrag/ui/qa.py`,
`src/vvrag/verify/examiner.py`, `src/vvrag/reader/prompts.py`.

## P2 — Reader A/B on the same frozen questions

### Why

The submitted report used `gpt-5-nano`; we used `gemini-3.5-flash-lite`
because `gemini-3.5-flash` was quota-blocked. Public benchmarks are mixed and
not slide-QA-specific. We need a paired comparison, not a guess.

### Tasks

1. Add an OpenAI-compatible path for `gpt-5-nano` (direct OpenAI or
   OpenRouter). Keep provider/model in config; do not hardcode.
2. Run the same retrieval, prompts, examiner and `n=1` on a pre-specified
   subset:
   - `gemini-3.5-flash-lite`
   - `gemini-3.5-flash` (when quota allows)
   - `gpt-5-nano`
3. Test two reader contracts:
   - reader may return UNANSWERABLE;
   - reader always answers, with the verifier deciding what to show.
4. Report paired item-level deltas: coverage, exact/containment/F1, shown
   precision, verified precision, probe flags, and reader abstention reasons.

### Acceptance for P2

- a paired table with per-item flips and CP95 intervals;
- no claim of "significantly better" without a paired test;
- reader substitutions recorded in every row.

Files: `src/vvrag/reader/client.py`, `src/vvrag/reader/prompts.py`,
`scripts/run_test8.py`, `config*.yaml`.

## P3 — Verifier policy and calibration

### Why

Our `verified` state requires every extracted claim to be examiner-supported,
a quote on the page, resolved citations, question-detail checks and an
independent examiner. That is stricter than the submitted report's
supported-window gate. It is defensible, but it costs coverage.

### Tasks

1. Keep strict `verified` as the headline safety state.
2. Evaluate a "core answer claim verified, auxiliary claims flagged" policy as
   a separate arm. Do not silently weaken the current state.
3. Calibrate the gate only after a larger frozen set exists. Keep
   feature-order versioning; do not mix 15-col/18-col artifacts.
4. Report precision at matched coverage, not just at one operating point.

### Acceptance for P3

- at matched coverage, verified precision is not lower than the current run;
- false-verified probes remain 0 on the new probe set;
- accepted-error rate and CP95 upper bound are reported;
- no threshold is fitted on frozen3.

Files: `src/vvrag/verify/features.py`, `src/vvrag/gate/`,
`scripts/calibrate_gate.py`, `src/vvrag/ui/qa.py`.

## P4 — Grounding and localization

### Why

The submitted report reports IoU 0.5229; our all-row IoU is 0.1749 and our
page-hit-only IoU is 0.3446. Part of that is retrieval, part is different gold
boxes, and part is candidate granularity: we cite sentences, they cite
matched lines. The comparison is not grounding-only.

### Tasks

1. Add line-level candidate boxes and quote-refined boxes; prefer the tightest
   line/span that contains the quote.
2. Expose claim-region boxes in the final display path, not only in the
   examiner path.
3. Report page-hit-only IoU, hit@0.25/0.50/0.75, and the gold-box definition.
4. Align evaluation gold boxes with the submitted report's definition when
   making the comparison, or label the difference explicitly.

### Acceptance for P4

- pre-register a page-hit-only IoU target (proposed >= 0.45);
- no drop in verified precision or probe safety;
- both all-row and page-hit-only tables in the report.

Files: `src/vvrag/ground/candidates.py`, `src/vvrag/ground/boxes.py`,
`src/vvrag/ui/qa.py`, `scripts/make_frozen_report.py`.

## P5 — Frozen4 evaluation

### Build

- 200-300 gold questions and 100+ verified-absent probes.
- Deck-disjoint from the 8 dev decks, `frozen2`, and `frozen3`.
- Audit gold pages/answers and probe absence before freezing.
- Record hashes for gold, probes, code, config and index.
- Write a pre-registration before the run.

### Run

- Run once after the code/config freeze.
- Primary arms: keyword and hybrid.
- Examiner: visual qwen (P1), with text-only as an ablation only.
- Reader A/B on a pre-specified subset (P2), not the full matrix.
- Keep `--resume` and per-row JSON persistence.

### Report

- product-shown and strict verified side by side;
- page-hit@1/@3, MRR, page-hit-only IoU;
- probes: answered, nothing-found, false-verified;
- examiner availability and every model substitution;
- failure audit split into retrieval misses, wrong-page answers, same-page
  label ambiguity, and verifier errors;
- CP95 intervals for every rate with a small denominator.

### Acceptance for P5

- one complete run with 0 unexplained errors;
- no tuning after the run;
- claims of improvement only on frozen4, never on frozen3.

## P6 — Operations and reliability

- Move reader/examiner calls to paid tiers or self-hosted endpoints where
  possible; free-tier quotas caused most substitutions.
- Remove dead examiner models (404 fallbacks) before the run.
- Aggregate per-model examiner errors; the current `state_reason` can show
  only the last fallback.
- Keep disk caching and `--resume`.
- Record wall-clock, API calls, and model identity per row.
- Do not run pytest during a live run (NLI/CPU contention).

## Execution order

1. **Retrieval dev work** (P0): ablation on dev decks, choose two candidates.
2. **Freeze** code/config/prompts and build `frozen4` (P5).
3. **Live run**: hybrid + visual examiner; reader A/B on a subset.
4. **Report**: product-shown, strict verified, probes, page-hit-only IoU,
   failure audit, provenance.
5. Only after that, decide whether a gate-calibrated arm is worth adding.

## What would count as "equivalently good or better"

Use like-for-like comparisons:

- their Baseline vs our **product-shown** answers;
- their Verified arm vs our `verified` on the **same frozen sample** with the
  same reader and visual examiner;
- their grounding IoU vs our page-hit-only IoU under the same gold-box
  definition.

On the current frozen3 rows, our verified precision on page-hit rows is
already 12/13 = 92.3%, close to their 91.1%. The remaining gap is mainly
retrieval coverage and the stricter verification contract, not evidence that
the reader or verifier is fundamentally worse.

## Appendix — ColQwen is open weights; the limit is compute, not usage

`vidore/colqwen2-v1.0` is an open-weight model. The Hugging Face card lists
Apache-2.0 for the Qwen2-VL backbone and MIT for the adapters. We download and
run it ourselves; there is no per-question API quota or license fee.

The limits we actually hit are:

- **GPU compute and time.** The model is a Qwen2-VL-2B backbone with a
  multi-vector head; loading and encoding pages needs CUDA in practice. The
  current laptop has CPU-only `torch`, so online query encoding is guarded off.
- **Colab/GPU availability.** Page vectors and query vectors were precomputed
  on Colab and cached, because the laptop cannot run the model inside a
  request.
- **Reader/examiner API quotas.** Gemini and Groq free tiers are the source of
  the 429s and model substitutions, not ColQwen.
- **Disk/bandwidth.** The model and index are multi-GB; that is a one-time
  download/storage cost.

To use ColQwen without a query cache:

1. Run on a CUDA GPU machine (local, Colab, or cloud) with
   `retriever.online_encoder: true` and `online_encoder_device: cuda`.
2. Or precompute query vectors for the new questions with
   `scripts/precompute_query_vectors.py`; hybrid then uses the cache.
3. Or force CPU with `ALLOW_CPU_QUERY_ENCODER=1` and
   `online_encoder_device: cpu`; this works but is very slow and is not
   recommended for a live request.
4. If using a hosted inference provider, that provider's limits apply; the
   open weights themselves do not impose one.
