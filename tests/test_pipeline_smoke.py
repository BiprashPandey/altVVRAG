"""Full-pipeline smoke test with mocked reader (Section 11). No API keys / GPU."""
from vvrag.pipeline import answer, parse_claims


def test_parse_claims_code_fence():
    raw = '```json\n{"answer": "42", "claims": [{"text": "c", "answer_span": "42"}]}\n```'
    ans, claims = parse_claims(raw)
    assert ans == "42" and len(claims) == 1


def test_pipeline_smoke_offline_mock(tmp_path):
    import vvrag.pipeline as P

    res = P.answer("What is the answer?", offline_mock=True, config_path="config.yaml")
    assert res.answer == "mock answer"
    assert len(res.samples) == 1
    assert res.evidence.doc_id == "mock_doc"
    # second call exercises cache path without error
    res2 = answer("What is the answer?", offline_mock=True, config_path="config.yaml")
    assert res2.answer == "mock answer"
