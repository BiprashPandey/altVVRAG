"""Reader tests (Section 7.3, Phase 5 acceptance without live key)."""
import os

import pytest

from vvrag.reader.claims import (
    answers_match,
    normalize_answer,
    parse_reader_output,
    parse_reader_output_full,
    token_f1,
    validate_evidence_ids,
    validate_sentence_ids,
)
from vvrag.reader.client import SequenceMockReader, run_reader_samples
from vvrag.reader.prompts import build_prompt, build_sentence_prompt
from vvrag.schemas import SentenceSpan

GOOD = '{"answer": "42 units", "claims": [{"text": "The chart shows 42 units.", "answer_span": "42 units"}]}'


def _sents():
    return [SentenceSpan("d", 0, 0, "Nepal is a country.", (0, 0, 100, 20)),
            SentenceSpan("d", 0, 1, "Rastra means nation.", (0, 30, 100, 50))]


def test_prompt_contract():
    p = build_prompt("What is total?", crop_text="total 42", max_claims=3)
    assert "UNANSWERABLE" in p and "at most 3 claims" in p and "total 42" in p
    assert '"answer"' in p and '"claims"' in p
    g = build_sentence_prompt("q?", _sents(), "crop", max_claims=5)
    assert "[S0] Nepal is a country." in g and "sentence_ids" in g
    assert "verbatim extractive" in g  # rec 7: short verbatim answers


def test_sentence_ids_parse_and_validate():
    raw = ('{"answer": "Nepal Rastra", "quote": "Nepal Rastra", "sentence_ids": [17, 18],'
           ' "claims": [{"text": "x", "answer_span": "Nepal Rastra",'
           ' "evidence_sentence_ids": [17, 18]}]}')
    ans, quote, sids, claims, ok = parse_reader_output_full(raw)
    assert ok and sids == [17, 18] and claims[0].evidence_sentence_ids == [17, 18]
    # legacy 3-tuple wrapper still works
    ans2, claims2, ok2 = parse_reader_output(raw)
    assert ok2 and answers_match(ans2, "Nepal Rastra")
    assert validate_sentence_ids([17, 18, 19], 215) == (True, True)
    assert validate_sentence_ids([17, 19], 215) == (True, True)  # non-consecutive ok
    assert validate_sentence_ids([999], 215) == (False, False)  # out of range
    assert validate_sentence_ids(None, 215) == (False, False)
    assert validate_sentence_ids([], 215) == (False, False)
    # legacy alias maps word-id callers to the sentence validator
    assert validate_evidence_ids([17, 19], 215) == (True, True)


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


def test_parse_visual_region_ext():
    from vvrag.reader.claims import parse_reader_output_ext

    raw = ('{"answer": "65", "quote": "", "sentence_ids": [],'
           ' "evidence_type": "visual",'
           ' "region": {"page": "p1", "box_frac": [0.1, 0.2, 0.4, 0.5]},'
           ' "claims": []}')
    ans, quote, sids, refs, region, claims, ok = parse_reader_output_ext(raw)
    assert ok and ans == "65"
    assert region == {"page": 1, "box_frac": [0.1, 0.2, 0.4, 0.5],
                      "evidence_type": "visual"}
    # invalid boxes / pages are dropped, never crash
    bad = ('{"answer": "x", "region": {"page": "p0",'
           ' "box_frac": [0.9, 0.9, 0.1, 0.1]}}')
    _a, _q, _s, _r, reg2, _c, ok2 = parse_reader_output_ext(bad)
    assert ok2 and reg2 is None
    no_reg = '{"answer": "x", "sentence_ids": [3]}'
    _a, _q, _s, _r, reg3, _c, _o = parse_reader_output_ext(no_reg)
    assert reg3 is None


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
