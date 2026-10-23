import json

from vvrag.verify.claim_verifier import (claim_confidence, claim_is_supported,
                                        normalize_claim_label, parse_claims)
from vvrag.verify.examiner import parse_examiner_output


def test_normalize_claim_labels_and_citation_veto():
    assert normalize_claim_label("yes") == "supported"
    assert normalize_claim_label("partly") == "partially_supported"
    assert normalize_claim_label("unclear") == "insufficient"
    assert normalize_claim_label("no") == "unsupported"
    assert normalize_claim_label("supported", citation_specific=False) == (
        "partially_supported")
    assert normalize_claim_label("unsupported", citation_specific=False) == (
        "unsupported")


def test_claim_is_supported_requires_specific_region():
    v = {"label": "supported", "citation_specific": True,
         "region_supports": True}
    assert claim_is_supported(v)
    assert claim_is_supported(v, require_citation_specific=False)
    assert not claim_is_supported({**v, "citation_specific": None})
    assert not claim_is_supported({**v, "region_supports": False})
    assert not claim_is_supported({**v, "label": "partially_supported"})
    assert claim_confidence("supported") == 1.0
    assert claim_confidence("unsupported") == 0.0


def test_parse_claims_and_examiner_output():
    obj = {
        "quote_found": True, "quote_seen": "12.5 bn",
        "answer_supported": "yes", "answers_the_question": True,
        "citation_specific": True, "region_supports": True,
        "claims": [
            {"index": 0, "label": "supported",
             "citation_specific": True, "region_supports": True},
            {"index": 1, "answer_supported": "partly",
             "citation_specific": False, "region_supports": True},
        ],
    }
    v = parse_examiner_output(json.dumps(obj))
    assert len(v["claims"]) == 2
    assert v["claims"][0]["label"] == "supported"
    assert v["claims"][1]["label"] == "partially_supported"
    parsed = parse_claims([{"index": "2", "label": "not_supported"}])
    assert parsed[0]["index"] == 2
    assert parsed[0]["label"] == "unsupported"
    assert parse_claims(None) == []
