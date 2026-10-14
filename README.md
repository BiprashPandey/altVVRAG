# Verifiable Visual RAG (altVRAG)

A cheaper/faster variant of a three-stage verifiable document-QA system
(**retrieve → ground → verify-or-abstain**) with a conformal abstention gate.
Full spec: `projectGuidelines.md` (local-only, kept in your `docs/` folder).

Research contributions (vs. the reference design):
- **C1** — patch-similarity heatmap grounding from the retriever's own MaxSim scores
- **C2** — self-consistency (reader sampled k times) instead of a verifier VLM
- **C3** — NLI entailment on the cropped evidence region instead of full-page VLM checks
- **C4** — conformal abstention gate with a statistical risk guarantee (vs. fixed threshold)

## Quickstart (CPU-only debug run)

```bash
git clone <this-repo> && cd altVRAG
pip install -r requirements.txt
pip install -e . --no-deps
cp .env.example .env          # fill keys only for live-reader runs
pytest                        # must be green (~60 tests)
python -m vvrag.eval.run_eval --config config.yaml --all   # one-command debug results
```

This builds a synthetic 16-deck debug set, ingests it (fake embeddings),
runs 4 arms + 4 ablations (real retrieval/grounding/NLI/scorer/gate, simulated
reader), and writes `results/runs/{stamp}/` with `predictions.jsonl`,
`summary.json/.md`, `main_table.tex`, and `figures/` (risk-coverage, IoU,
calibration). Latest debug table is in `docs/decisions.md` (local-only).
(Phase 9 entry).

## Free API keys (for live-reader runs)

- **Gemini** (default reader): https://aistudio.google.com → API key → `GEMINI_API_KEY=...`
  in `.env`. Then: `pip install google-genai`, run
  `probe_gemini()` (`src/vvrag/reader/client.py`) to list current free models,
  set `reader.model` in `config.yaml` (never hardcode), and run
  `pytest tests/test_reader.py -k live` (expects ≥95% valid JSON).
- **Groq / OpenRouter** (alternate VLM family): `GROQ_API_KEY` / `OPENROUTER_API_KEY`,
  `reader.provider: groq|openrouter` + a free vision model in `config.yaml`.

## GPU ingestion (Colab/Kaggle, run once)

`notebooks/ingest_colab.ipynb` is a thin wrapper: installs deps, probes the
environment, verifies the retriever checkpoint (**Option A**: tries
`vidore/colqwen2-v1.0` first, falls back to a working ColPali checkpoint),
runs `python -m vvrag.ingest.run_ingest --input … --out … --embedder colqwen`,
and zips `data/index` for download. Unzip into `data/` on the laptop; the
online pipeline runs on CPU from the export. CLI:

```bash
python -m vvrag.ingest.run_ingest --input data/raw --out data/index
python -m vvrag.pipeline --question "..." --corpus data/index --out results/demo/result.json
```

## Real-data eval (SlideVQA)

```bash
bash scripts/download_data.sh data/raw_eval   # tries Hub mirrors, else use synthetic
```

Needs the GPU index + a reader key. Split is by deck (no leakage):
train/calib/test = 30/30/40. Scorer trains on train, tau on calib, all
reported numbers on test with bootstrap 95% CIs.

## Layout

- `src/vvrag/` — `config schemas cache ratelimit pipeline` + `ingest/ retrieve/
  ground/ reader/ verify/ gate/ eval/` (one module per guideline section)
- `tests/` — pytest suite (unit + smoke, no keys/GPU except marked live tests)
- `config.yaml` / `.env.example` — every model, threshold, and path (nothing hardcoded)
- `docs/` — local-only notes (gitignored): decisions/deviations log, run guide
- `results/` (gitignored) — `runs/{stamp}/` tables+figures, `demo/` overlays

## Status / known limits (honest)

- Debug runs use synthetic stand-ins (fake embeddings, planted queries,
  simulated reader); see `docs/decisions.md` (local-only) for exactly what is/isn't measured.
- Small calib sets → conformal may return `tau=+inf` (correctly abstains);
  needs hundreds of calib questions for tight guarantees.
- ColQwen weights, live Tesseract, SlideVQA download, live reader JSON-rate:
  verified only in their target environments (decisions.md records each).
