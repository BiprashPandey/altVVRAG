# decisions.md — log every non-obvious decision + deviation (Section 0.5)

## 2026-10-14 — Phase 6 (verification, C2+C3)
- NLI model downloaded + verified LIVE on CPU: cross-encoder/nli-deberta-v3-base,
  label2id = {contradiction:0, entailment:1, neutral:2} read from model config
  (never assumed). CrossEncoder.predict returns LOGITS; softmax applied in code.
  Entailment 0.99+ on true pairs, ~0 on false; sentence-windowing finds buried facts.
- Claim template lesson: hypotheses of form "X is reported in the document" score
  NEUTRAL (meta-talk); bare-fact claims ("X.") entail at 0.996. SimulatedReader
  uses bare facts (also matches the Phase 5 atomic-claim prompt better).
- Crop text expanded to full visual lines (y-clustered) so NLI gets complete
  sentences; join order is visual (lines by y, words by x), robust to multi-block PDFs.
- VLM fallback implemented, OFF by default (flag only).
- Missing entailment encoded as entail_prob=0.5 + crop_has_text=0 (spec 7.4).

## 2026-10-14 — Phase 7 (gate, C4)
- DEVIATION x3 from 8.2 (proven degenerate as written, see conformal.py docstring):
  (a) coarse grid (0..1 step 0.05) instead of unique calib scores (strictest unique
  candidate has n_acc~1 -> CP bound 1-delta > alpha always -> always abstain);
  (b) skip grid points with n_acc < min_n (default 30; eval auto-scales to
  max(5, n_calib//3)); (c) loosest-passing selection without early stopping
  (UCB non-monotone; strict-end tiny-n failures blocked the whole walk).
  grid=None recovers the literal spec form.
- Empirical guarantee check PASSES (200 splits, violation ~= <= delta + 0.05).
- Scorer: sklearn LogisticRegression + StandardScaler on FEATURE_ORDER columns;
  hand baseline 0.4*agr + 0.4*ent + 0.2*heat (missing entail -> 0.5).

## 2026-10-14 — Phase 8 (eval harness)
- Synthetic debug set plants each fact sentence inside a KNOWN 2x2 patch-cell
  block (pixel-mapped via grid_cell_range_to_pixel_box), storing flat patch idx
  on the item; queries built from those patches. C1 grounding is thus genuinely
  tested offline: center_in_box = 1.0. IoU~0.03 expected (1-line strip vs 2x2-cell
  box) — strip-vs-block geometry, not a bug; real IoU needs SlideVQA.
- SimulatedReader: seeded, acc=0.7, distractors = sibling facts from same doc
  (plausible wrong values -> NLI must discriminate; page-true distractors
  sometimes entail — realistic NLI noise, cf. spec 14 domain-shift note).
- Determinism: Python hash() is salted per-process; all seeded RNG uses
  stable_seed() (sha256). Same command -> same numbers.
- Retrieval on synthetic: Hit@3 0.667 (mult 4) -> 0.75 (mult 8): pooled-prefetch
  dilution is real and measurable (spec ablation 10.4#5). Rerank loses the rest
  (noise-0.15 queries vs 32 random pages).
- SlideVQA loader tries Hub mirrors, raises with instructions otherwise;
  download_data.sh documents the same. Real-data runs pending GPU+key.

## 2026-10-14 — Phase 9 (debug full runs, --all, 16 decks / 96 items)
- train 24 / calib 48 / test 24 (by deck). Learned scorer on train(+calib),
  tau on calib, reported on test with bootstrap CIs.
- verified-conformal: cover 0.667, sel_acc 0.875, tau=0.65, realized err 0.125
  (n=16 answered; tiny-test noise, alpha=0.1). verified-fixed: 0.542/0.846.
  Baseline: 1.0/0.708. AURC ~0.08-0.10. Full table: results/runs (gitignored).
- Ablations: heat-mean finite (0.625/0.933); k=5 (0.708/0.824); alpha=0.05 -> +inf
  (uncertifiable on 48 calib — correct behavior, real finding); hand scorer
  (0.417/1.0); prefetch8 Hit@3 0.75.
- Small-calib lesson: 10-deck runs inf'd every conformal arm; 16 decks work.
  Article runs need hundreds of calib questions for tight, finite gates.

## 2026-10-14 — Phase 5 (reader)
- Normalization: lower, strip punct, drop articles, un-thousands numbers, collapse WS. answers_match = exact-norm OR token-F1 >= 0.8.
- Sampling: sample 0 @ temp 0 (primary), rest @ reader.temperature; invalid JSON retried ONCE then marked failed (raw kept).
- Providers: mock (tests) + gemini/groq/openrouter clients coded but UNVERIFIED — no SDKs installed here, no GEMINI_API_KEY. Gemini uses response_mime_type=application/json; Groq/OpenRouter use base64 image_url + json_object format.
- Live 95%-JSON check exists as skipped test (needs GEMINI_API_KEY + GEMINI_READER_MODEL). To run: set key, `pytest tests/test_reader.py -k live`. Reader model still CONFIGURE_ME until free-list probe.
- Dedup: pipeline.parse_claims now wraps claims.parse_reader_output.

## 2026-10-14 — Phase 4 (grounding, C1)
- Aggregation default max, mean behind flag (ablation in Phase 9). No token-weighting yet (flag-ready).
- Box ranking = SUMMED heat (spec: "top max_boxes by summed heat") — favors larger blobs; test pins both directions.
- Components: scipy 8-connectivity. heat_concentration = fraction of heat inside boxes.
- Crop = UNION of boxes (spec allows union-or-each; union chosen so NLI gets full context).
- Snap-to-lines implemented but OFF by default (ablation arm later).
- Real-data overlay inspection (8/10 pages) deferred to SlideVQA (Phase 8); synthetic check: box (392,192,608,408) vs true (400,200,600,400) = exact + 8px padding. Overlays in results/demo/ (gitignored).
- Token drop-list covers common specials; ColQwen query-processor specials TBD on GPU box.

## 2026-10-14 — Phase 3 (retrieval)
- MaxSim = sum_t max_p dot, float32; brute-force test uses approx (float32 vs float64 accumulation).
- Stage 1: per-token search over pooled points (limit = mult*top_k), page score = sum of token-best pooled dots; returns mult*top_k candidates. Stage 2: scroll full multivector by (doc_id, page) filter, local MaxSim rerank.
- Qdrant MAX_SIM ordering matches local maxsim_rank exactly (top-1 + top-3 order verified on 6-page fake index).
- Real query embedder (ColQwen query processor) UNVERIFIED — same GPU caveat as Phase 2; FakeQueryEmbedder is text-seeded/deterministic for tests only.
- Debug-set caveat: queries are perturbed gold-patch copies (guaranteed-near by construction), so Hit@3 ~= 1.0 here measures pipeline correctness, not model quality. Real Hit@3 comes in Phase 8 with SlideVQA.

## 2026-10-14 — Phase 2 (ingestion)
- PyMuPDF 1.28.2 (`import pymupdf`; `fitz` alias deprecated). Words via `page.get_text("words")`, scale dpi/72. Text-layer threshold: >=10 words.
- qdrant-client 1.19.1: multivector verified live (`VectorParams(multivector_config=MultiVectorConfig(MAX_SIM))`, upsert+query in :memory:). Full=1 point/page; pooled=1 point/pooled-vector with deterministic uuid5 ids (idempotent upsert).
- Pooled count per page = grid_h + grid_w + 1 (global). Fake grid 8x6 -> 15.
- Tesseract binary + pytesseract NOT on laptop; `ocr_page` raises clean OCRError. Colab notebook installs tesseract-ocr. OCR live test deferred to GPU box.
- ColQwen2 path UNVERIFIED (no torch on laptop). `ColQwenEmbedder._deduce_page_vectors` tries batch grid metadata then raises GridMismatchError rather than guessing. GPU operator must run `probe_environment()` + dry-run download first (notebook cell 2).
- requirements: added pymupdf/pillow/qdrant-client (installed). torch/transformers/colpali-engine stay GPU-only.
- CPU acceptance: 3 PDFs x 2 pages with fake embedder -> 6 full + 90 pooled points, word boxes in SQLite, 2nd run skips all 6 (resumable).

## 2026-10-14 — Phase 1
- Python 3.14.7 on dev laptop (guideline asks 3.11+; OK). Compat of torch/colpali-engine with 3.14 to be verified at Phase 2; Colab GPU will use its own Python.
- Schemas use stdlib `dataclasses` (guideline allows pydantic or dataclasses). Chose dataclasses to keep Phase 1 dependency-free.
- `.env` loader: uses python-dotenv if installed, else minimal fallback parser in `config.py`.
- Retriever: Option A (verify `vidore/colqwen2-v1.0` on Hub at Phase 2, fallback to working ColPali ckpt). Not verified yet.
- Reader default: `gemini` (model `CONFIGURE_ME` until Phase 5 free-list check). Phase 1 implements `mock` provider only; gemini/groq/openrouter raise NotImplementedError.
- Cache: one JSON file per SHA256 key, atomic write via tmp+rename. Key = full request (provider, model, prompt, image hashes, temperature, sample index).
- Rate limiter: token bucket, default 10 rpm per config; retry with exponential backoff + jitter; tests use base_delay=0.
- Grid->pixel helper lives in `schemas.py` as single source of truth (Section 4 requirement).
