# decisions.md — log every non-obvious decision + deviation (Section 0.5)

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
