"""End-to-end pipeline stub (Phase 1). Full pipeline lands in Phase 8 (Section 9)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vvrag.cache import DiskCache
from vvrag.config import load_config
from vvrag.logging_utils import get_logger
from vvrag.ratelimit import RateLimiter
from vvrag.reader.client import create_reader
from vvrag.schemas import Claim, Evidence, Features, ReaderSample, Result

log = get_logger(__name__)


def parse_claims(raw: str, max_claims: int = 5) -> tuple[str, list[Claim]]:
    """Parse strict-JSON reader output {answer, claims[]} robustly (strip fences)."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return "UNANSWERABLE", []
    answer = str(obj.get("answer", "UNANSWERABLE"))
    claims: list[Claim] = []
    for c in obj.get("claims", [])[:max_claims]:
        if isinstance(c, dict) and "text" in c:
            claims.append(Claim(text=str(c["text"]), answer_span=c.get("answer_span")))
    return answer, claims


def answer(
    question: str,
    corpus_filter: str | None = None,
    offline_mock: bool = True,
    config_path: str = "config.yaml",
) -> Result:
    """Phase 1 smoke implementation: mock retrieval + mock reader, no network/GPU."""
    t0 = time.time()
    cfg = load_config(config_path)
    cache = DiskCache(cfg.cache_dir)
    limiter = RateLimiter(cfg.ratelimit.requests_per_minute)
    reader = create_reader("mock", "mock", cache, limiter, cfg.ratelimit.max_retries)

    t1 = time.time()
    raw_samples = reader.generate([], f"Q: {question}", temperature=0.0, n=1)
    t2 = time.time()
    ans, claims = parse_claims(raw_samples[0], cfg.reader.max_claims)
    samples = [ReaderSample(sample_id=0, answer=ans, claims=claims, raw=raw_samples[0])]
    evidence = Evidence(doc_id="mock_doc", page=0, boxes=[], crop_path="", crop_text="", heat_concentration=0.0)
    features = Features(
        agreement=1.0, entail_prob=None, retrieval_margin=0.0,
        heat_concentration=0.0, n_claims=len(claims), crop_has_text=False,
    )
    return Result(
        question=question,
        answer=ans,
        abstained=False,
        score=1.0,
        tau=0.0,
        evidence=evidence,
        samples=samples,
        features=features,
        timings={"total": time.time() - t0, "reader": t2 - t1},
        api_calls=0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="vvrag single-question demo (Phase 1 stub)")
    ap.add_argument("--question", required=True)
    ap.add_argument("--corpus", default="data/index")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--offline-mock", action="store_true", default=True)
    ap.add_argument("--out", default="results/demo/result.json")
    args = ap.parse_args()
    res = answer(args.question, offline_mock=args.offline_mock, config_path=args.config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res.to_dict(), indent=2))
    print(out)


if __name__ == "__main__":
    main()
