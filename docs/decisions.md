# decisions.md — log every non-obvious decision + deviation (Section 0.5)

## 2026-10-14 — Phase 1
- Python 3.14.7 on dev laptop (guideline asks 3.11+; OK). Compat of torch/colpali-engine with 3.14 to be verified at Phase 2; Colab GPU will use its own Python.
- Schemas use stdlib `dataclasses` (guideline allows pydantic or dataclasses). Chose dataclasses to keep Phase 1 dependency-free.
- `.env` loader: uses python-dotenv if installed, else minimal fallback parser in `config.py`.
- Retriever: Option A (verify `vidore/colqwen2-v1.0` on Hub at Phase 2, fallback to working ColPali ckpt). Not verified yet.
- Reader default: `gemini` (model `CONFIGURE_ME` until Phase 5 free-list check). Phase 1 implements `mock` provider only; gemini/groq/openrouter raise NotImplementedError.
- Cache: one JSON file per SHA256 key, atomic write via tmp+rename. Key = full request (provider, model, prompt, image hashes, temperature, sample index).
- Rate limiter: token bucket, default 10 rpm per config; retry with exponential backoff + jitter; tests use base_delay=0.
- Grid->pixel helper lives in `schemas.py` as single source of truth (Section 4 requirement).
