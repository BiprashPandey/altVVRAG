"""Unit tests for the cross-family visual-check verdict (Run 8 Phase C).

`_visual_check_verdict` decides whether an independent vision model's reply
confirms the reader's claimed answer. Semantics: an explicit "supports" field
is authoritative (a contradictory UNANSWERABLE read is never accepted);
without it, normalized string/number containment decides.
"""
from vvrag.ui.qa import _visual_check_verdict


def test_supports_true_with_reading_matches():
    match, sup, read = _visual_check_verdict(
        "12.5 bn", '{"answer": "12.5 bn", "supports": true}')
    assert match is True and sup is True and read == "12.5 bn"


def test_supports_true_but_unanswerable_read_is_rejected():
    match, sup, read = _visual_check_verdict(
        "12.5 bn", '{"answer": "UNANSWERABLE", "supports": true}')
    assert match is False and sup is True


def test_supports_false_is_authoritative_even_if_read_matches():
    match, sup, _ = _visual_check_verdict(
        "12.5 bn", '{"answer": "12.5 bn", "supports": false}')
    assert match is False and sup is False


def test_string_supports_value_truthy():
    match, _, _ = _visual_check_verdict(
        "12.5 bn", '{"answer": "12.5 bn", "supports": "true"}')
    assert match is True


def test_no_verdict_field_falls_back_to_containment():
    match, sup, _ = _visual_check_verdict("12.5 bn", '{"answer": "12.5 bn"}')
    assert match is True and sup is None


def test_no_verdict_number_normalized_containment():
    match, _, _ = _visual_check_verdict("12.5 bn", '{"answer": "12.5"}')
    assert match is True


def test_non_json_raw_with_answer_text_matches():
    match, sup, read = _visual_check_verdict(
        "12.5 bn", "the table shows 12.5 bn in 2011")
    assert match is True and sup is None and "12.5 bn" in read


def test_unrelated_non_json_raw_rejected():
    match, _, _ = _visual_check_verdict("12.5 bn", "UNANSWERABLE")
    assert match is False


def test_wrong_answer_rejected():
    match, _, _ = _visual_check_verdict(
        "12.5 bn", '{"answer": "15.0%", "supports": false}')
    assert match is False
