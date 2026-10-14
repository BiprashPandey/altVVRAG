# Verifiable Visual RAG (altVRAG)

Cheaper/faster variant of a three-stage verifiable document-QA system
(retrieve → ground → verify-or-abstain) with patch-similarity heatmap grounding,
self-consistency, crop-text NLI, and a conformal abstention gate.
Full spec: `projectGuidelines.md`.

## Phase 1 status (skeleton)
- `pip install -r requirements.txt` (Phase 1 needs only pyyaml, python-dotenv, numpy, pytest)
- `cp .env.example .env` and fill keys when needed (never commit `.env`)
- `pytest` — must be green
- `python -m vvrag.pipeline --question "test?" --offline-mock --out results/demo/result.json`

## Layout
- `src/vvrag/` — package (`config`, `schemas`, `cache`, `ratelimit`, `pipeline`, `reader/`, + per-phase subpackages)
- `tests/` — pytest suite (grows each phase per Section 11)
- `config.yaml` / `.env.example` — all thresholds, models, paths (nothing hardcoded)
- `docs/decisions.md` — decision log (required by guidelines)

## Roadmap
Phases 1–10 per `projectGuidelines.md` Section 12. Full README/reproducibility lands in Phase 10.
