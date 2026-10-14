"""Reader tests (Section 7.3, Phase 5 acceptance without live key)."""
import os

import pytest

from vvrag.reader.claims import (
    answers_match,
    normalize_answer,
    parse_reader_output,
    token_f1,
)
from vvrag.reader.client import SequenceMockReader, run_reader_samples
from vvrag.reader.prompts import build_prompt

GOOD = '{"answer": "42 units", "claims": [{"text": "The chart shows 42 units.", "answer_span": "42 units"}]}'


def test_prompt_contract():
    p = build_prompt("What is total?", crop_text="total 42", max_claims=3)
    assert "UNANSWERABLE" in p and "at most 3 claims" in p and "total 42" in p
    assert '"answer"' in p and '"claims"' in p


def test_parse_valid_fenced_and_capped():
    ans, claims, ok = parse_reader_output("```json\n" + GOOD + "\n```", max_claims=5)
    assert ok and ans == "42 units" and len(claims) == 1
    many = '{"answer": "x", "claims": [{"text": "t%d"}]}' % 0
    import json as _j

    big = _j.dumps({"answer": "x", "claims": [{"text": f"t{i}"} for i in range(9)]})
    _, c2, ok2 = parse_reader_output(big, max_claims=5)
    assert ok2 and len(c2) == 5


def test_parse_invalid_then_retry_once():
    r = SequenceMockReader(["not json{{", GOOD])
    samples, calls = run_reader_samples(r, [], "prompt", n_samples=1, temperature=0.7)
    assert calls == 2 and samples[0].answer == "42 units"
    r2 = SequenceMockReader(["bad1", "bad2"])
    s2, c2 = run_reader_samples(r2, [], "prompt", n_samples=1, temperature=0.7)
    assert c2 == 2 and s2[0].answer == "UNANSWERABLE" and s2[0].raw == "bad2"


def test_primary_sample_uses_temp_zero():
    r = SequenceMockReader([GOOD, GOOD, GOOD])
    run_reader_samples(r, [], "prompt", n_samples=3, temperature=0.7)
    assert r.temps == [0.0, 0.7, 0.7]


def test_normalization_and_f1():
    assert normalize_answer("The Quick, Brown Fox!") == "quick brown fox"
    assert normalize_answer("1,000 kg") == "1000 kg"
    assert token_f1("42 units", "42 units") == 1.0
    assert answers_match("The 42 units", "42 units")
    assert answers_match("revenue was 42 million",
                         "revenue 42 million")  # F1 ~0.86 >= 0.8
    assert not answers_match("apples", "oranges")


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"),
                    reason="live Gemini check needs GEMINI_API_KEY")
def test_live_gemini_json_rate(tmp_path):
    """Phase 5 acceptance: >=95% valid JSON on live calls (needs key)."""
    from vvrag.cache import DiskCache
    from vvrag.ratelimit import RateLimiter
    from vvrag.reader.client import create_reader

    reader = create_reader("gemini", os.environ.get("GEMINI_READER_MODEL", "gemini-2.0-flash"),
                           DiskCache(tmp_path / "c"), RateLimiter(600))
    ok = 0
    n = 5
    for _ in range(n):
        out = reader.generate([], build_prompt("Say hello."), 0.0, 1)[0]
        _, _, good = parse_reader_output(out)
        ok += good
    assert ok / n >= 0.95
