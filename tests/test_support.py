"""Phase 3: deterministic answer-support veto (number/unit normalization).
Phase 9: question-aware support (type match + comparison values)."""
from vvrag.verify.support import (answer_kind, answer_supported,
                                  answer_type_match, comparison_values_present,
                                  is_comparison, normalize_numbers, qa_support)


def test_number_normalization():
    assert normalize_numbers("42 bn") == "42 billion"
    assert normalize_numbers("12.5bn") == "12.5 billion"
    assert normalize_numbers("1,234") == "1234"
    assert normalize_numbers("12,5") == "12.5"
    assert normalize_numbers("three") == "3"
    assert normalize_numbers("42%") == "42 percent"
    assert normalize_numbers("$4.2 million") == "4.2 million"


def test_containment_supported():
    ok, score, method = answer_supported(
        "42 billion", ["Revenue reached 42 bn in 2025."])
    assert ok and score == 1.0 and method == "containment"


def test_number_word_supported():
    ok, _s, method = answer_supported(
        "3", ["The company opened three factories."])
    assert ok and method == "containment"


def test_unsupported_but_quoted():
    # Run 6 failure mode: answer is not in the cited text at all
    ok, score, method = answer_supported(
        "2015", ["Guaranteed by the seller."], entail_min=0.0001)
    assert not ok and score < 0.5 and method == "nli_low"


def test_nli_fallback_entails():
    ok, score, method = answer_supported(
        "the seller", ["The vendor guarantees the product."], entail_min=0.8)
    assert ok and score == 0.8 and method == "nli"


def test_empty_and_unanswerable():
    assert answer_supported("", ["x"], None) == (False, 0.0, "empty")
    assert answer_supported("UNANSWERABLE", ["x"], None) == (False, 0.0, "empty")


# --------------------------- Phase 9: question-aware -----------------------

def test_answer_kind_and_type_match():
    assert answer_kind("How many factories does Nestle operate?") == "count"
    assert answer_kind("In which year did revenue peak?") == "year"
    assert answer_kind("Who is the CEO?") == "name"
    assert answer_kind("Where is the plant located?") == "location"
    assert answer_kind("What is the revenue?") == "money"
    assert answer_kind("What colour is the logo?") == "other"
    # type checks
    assert answer_type_match("How many factories?", "46") is True
    assert answer_type_match("How many factories?", "Belgium") is False
    assert answer_type_match("In which year?", "2015") is True
    assert answer_type_match("In which year?", "Belgium") is False
    assert answer_type_match("Who leads it?", "Paul Bulcke") is True
    assert answer_type_match("Who leads it?", "12") is False
    assert answer_type_match("What colour is the logo?", "red") is None


def test_is_comparison_and_values_present():
    q = "Did revenue grow more than 7.5% or less than 3.9%?"
    assert is_comparison(q)
    assert not is_comparison("What is the revenue?")
    texts = ["Organic growth 7.5% versus total sales growth 3.9%."]
    assert comparison_values_present(q, "3.9", texts)
    assert not comparison_values_present(q, "3.9", ["total sales growth only"])
    # fewer than 2 numbers in question+answer -> nothing to compare
    assert comparison_values_present("Which is better?", "good", ["x"])


def test_qa_support_type_mismatch_vetoes():
    # Run 7 idx10: "how many business segments" answered with a segment LIST
    ok, score, method = qa_support(
        "How many business segments does Nestle report?", "Business Skills, Analytic Skills",
        ["Business Skills Analytic Skills Digital Marketing"], entail_min=0.95)
    assert not ok and method == "type_mismatch"
    # NLI-only supported path (verbatim number absent from the cited text)
    ok, score, method = qa_support(
        "How many factories?", "46", ["Nestle operates many factories worldwide."],
        entail_min=0.91)
    assert ok and score == 0.91 and method == "entail+type"
    # verbatim containment beats a low NLI score (legacy behavior kept)
    ok, score, method = qa_support(
        "What did revenue do?", "in June",
        ["The team increased the price in June."], entail_min=0.06)
    assert ok and method == "containment+type"


def test_qa_support_comparison_requires_both_values():
    q = "Did revenue grow more than 7.5% or less than 3.9%?"
    ok, _s, method = qa_support(q, "3.9", ["Revenue grew 3.9 percent."],
                                entail_min=0.9)
    assert not ok and method == "missing_comparison_values"
    ok, _s, method = qa_support(
        q, "3.9",
        ["Revenue grew 7.5 percent versus 3.9 percent last year."],
        entail_min=0.9)
    assert ok and method == "containment+type"


def test_qa_support_containment_fallback_and_nli_low():
    ok, score, method = qa_support(
        "How many factories?", "46", ["Nestle operates 46 factories."])
    assert ok and method == "containment+type"
    ok, score, method = qa_support(
        "How many factories?", "99", ["Nestle operates 46 factories."],
        entail_min=0.1)
    assert not ok and method == "nli_low"
