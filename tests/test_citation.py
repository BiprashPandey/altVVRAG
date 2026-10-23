"""Citation-specific deterministic guards (report-gap fix)."""
from vvrag.verify.citation import (citation_overlap, content_terms,
                                   page_number_sink, shared_terms)


def test_content_terms_keep_numbers_and_drop_question_words():
    terms = content_terms("What was the 2011 revenue of 12.5 bn?")
    assert "2011" in terms and "revenue" in terms
    assert "what" not in terms and "the" not in terms
    assert any(t.startswith("12.5") for t in terms)


def test_citation_overlap_true_for_real_region():
    ok, score, shared = citation_overlap(
        "Revenue was 12.5 bn", ["Trading Operating Profit 12.5 bn 7.5 bn"])
    assert ok and score > 0 and "12.5" in shared


def test_short_numeric_answer_does_not_match_substring():
    ok, _score, shared = citation_overlap("12.5", ["112.5"])
    assert not ok and shared == []


def test_citation_overlap_short_labels():
    assert citation_overlap("SQL", ["SQL"])[0]
    assert citation_overlap("Command", ["Command"])[0]


def test_shared_terms_are_order_independent():
    assert shared_terms("ALCATEL-LUCENT OVERVIEW 2012",
                        ["ALCATEL-LUCENT OVERVIEW 2012"]) == [
        "2012", "alcatel", "lucent", "overview"]


def test_page_number_sink_rejects_number_only_region():
    claim = "Revenue was 12.5 bn in 2011"
    assert page_number_sink(claim, ["12"])
    assert not page_number_sink(claim, ["12.5"])
    assert page_number_sink("12", ["12"]) is False


def test_empty_claim_fails_closed():
    ok, score, shared = citation_overlap("", ["Revenue 12.5 bn"])
    assert not ok and score == 0.0 and shared == []
