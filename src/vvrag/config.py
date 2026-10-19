"""Typed config loading: config.yaml + .env. Nothing hardcoded downstream."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RenderConfig:
    dpi: int = 150


@dataclass
class RetrieverConfig:
    model_id: str = "vidore/colqwen2-v1.0"
    device: str = "cuda"
    dtype: str = "bfloat16"
    prefetch_k_multiplier: int = 4
    top_k: int = 3
    pooling: str = "row_col_mean"
    query_encoder: str = "colqwen"
    # Phase 1 (2026-10-18): keyword | visual | hybrid page ranking.
    # "keyword" preserves the pre-Run-7 behavior; the UI/run harnesses can
    # override per call. `rrf_k` is the reciprocal-rank-fusion constant.
    retrieval_mode: str = "keyword"
    rrf_k: int = 60
    query_vectors_dir: str = "data/query_cache"


@dataclass
class QdrantConfig:
    mode: str = "local"
    path: str = "data/qdrant"
    collection_full: str = "pages_full"
    collection_pooled: str = "pages_pooled"


@dataclass
class ReaderConfig:
    provider: str = "gemini"
    model: str = "CONFIGURE_ME"
    n_samples: int = 3
    temperature: float = 0.7
    max_output_tokens: int = 512
    max_claims: int = 5
    fallback_provider: str = ""
    fallback_model: str = ""
    # suggestion #7: reduce false abstentions on noisy OCR
    retry_on_unanswerable: bool = True
    unanswerable_retry_overlap: float = 0.2
    max_answer_retries: int = 1
    # Phase 2: multi-page reading (top-k pages in the prompt; primary has images)
    max_pages: int = 3
    next_page_retry: bool = True


@dataclass
class VerifyConfig:
    quote_threshold: float = 0.8
    # suggestion #7: tolerant quote check that ignores OCR whitespace/case noise
    ocr_normalize: bool = True
    # Phase 3: deterministic answer-support veto (answer must appear in the
    # cited text or be entailed by it, else the evidence is not "verified").
    require_answer_support: bool = False
    support_entail_threshold: float = 0.5
    # Phase 9 (Run 8): question-aware checks
    answer_type_check: bool = True       # "how many" needs a number, ...
    comparison_check: bool = True        # comparisons need both values cited
    # Visual middle tier: "off" | "agreement" | "cross_family"
    visual_tier: str = "cross_family"
    visual_min_agreement: float = 0.66
    cross_family_provider: str = "groq"
    cross_family_model: str = "qwen/qwen3.8-27b"


@dataclass
class RunConfig:
    """Live test-run harness settings (moved out of scripts 2026-10-18)."""
    live_call_gap_s: float = 25.0
    model_candidates: list = field(default_factory=lambda: [
        "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.6-flash",
        "gemini-3.1-flash-lite", "gemini-3.7-flash",
    ])
    reader_model_pinned: str = ""


@dataclass
class NliConfig:
    model_id: str = "cross-encoder/nli-deberta-v3-base"
    device: str = "cpu"


@dataclass
class GroundConfig:
    heat_threshold_rel: float = 0.5
    min_box_area_frac: float = 0.002
    max_boxes: int = 3
    box_padding_px: int = 8
    heatmap_as_prior_only: bool = True
    smooth_sigma: float = 0.0


@dataclass
class GateConfig:
    mode: str = "conformal"
    alpha: float = 0.10
    delta: float = 0.10
    fixed_tau: float = 0.5
    scorer_path: str = "data/gate/scorer.pkl"
    tau_path: str = "data/gate/tau.json"


@dataclass
class RatelimitConfig:
    requests_per_minute: int = 10
    max_retries: int = 6


@dataclass
class EvalConfig:
    dataset: str = "slidevqa"
    n_debug: int = 45
    seed: int = 42
    calib_frac: float = 0.5


@dataclass
class AppConfig:
    render: RenderConfig = field(default_factory=RenderConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    reader: ReaderConfig = field(default_factory=ReaderConfig)
    nli: NliConfig = field(default_factory=NliConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    ground: GroundConfig = field(default_factory=GroundConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    ratelimit: RatelimitConfig = field(default_factory=RatelimitConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    run: RunConfig = field(default_factory=RunConfig)
    cache_dir: str = "data/cache"


def _load_dotenv(dotenv_path: str | Path = ".env") -> None:
    """Minimal .env loader (no hard dependency for Phase 1 beyond python-dotenv if present)."""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path)
        return
    except ImportError:
        pass
    p = Path(dotenv_path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def load_config(config_path: str | Path = "config.yaml", dotenv_path: str | Path = ".env") -> AppConfig:
    _load_dotenv(dotenv_path)
    data: dict = {}
    p = Path(config_path)
    if p.exists():
        with p.open() as f:
            data = yaml.safe_load(f) or {}
    return AppConfig(
        render=RenderConfig(**data.get("render", {})),
        retriever=RetrieverConfig(**data.get("retriever", {})),
        qdrant=QdrantConfig(**data.get("qdrant", {})),
        reader=ReaderConfig(**data.get("reader", {})),
        nli=NliConfig(**data.get("nli", {})),
        verify=VerifyConfig(**data.get("verify", {})),
        ground=GroundConfig(**data.get("ground", {})),
        gate=GateConfig(**data.get("gate", {})),
        ratelimit=RatelimitConfig(**data.get("ratelimit", {})),
        eval=EvalConfig(**data.get("eval", {})),
        run=RunConfig(**data.get("run", {})),
        cache_dir=data.get("cache_dir", "data/cache"),
    )


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)
