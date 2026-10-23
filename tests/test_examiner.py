"""Run 9 (guideline s4.3): examiner parsing, confirmation rules, mock stand-in."""
import json

from vvrag.verify.examiner import (MockExaminer, examiner_confirmed,
                                   parse_examiner_output, quote_seen_match,
                                   verdict_supported_value)


def _v(**kw):
    base = {"quote_found": True, "quote_seen": "12.5 bn", "answer_supported": "yes",
            "answers_the_question": True, "problem": ""}
    base.update(kw)
    return parse_examiner_output(json.dumps(base))


def test_parse_examiner_output_strict_and_fenced():
    v = parse_examiner_output('{"quote_found": true, "quote_seen": "42", '
                              '"answer_supported": "yes", "answers_the_question": true}')
    assert v["ok"] and v["quote_found"] and v["answer_supported"] == "yes"
    fenced = parse_examiner_output('```json\n{"quote_found": false, '
                                   '"quote_seen": "", "answer_supported": "no", '
                                   '"answers_the_question": false, "problem": "not visible"}\n```')
    assert fenced["ok"] and fenced["quote_found"] is False
    assert fenced["problem"] == "not visible"
    bad = parse_examiner_output("I think the answer is fine.")
    assert not bad["ok"] and bad["parse_error"]


def test_quote_seen_match_prefers_ocr_containment():
    texts = ["Total revenue was 12.5 bn in 2011."]
    assert quote_seen_match("12.5 bn", "12.5 billion", texts) >= 0.6
    assert quote_seen_match("12.5 bn", "", texts) == 1.0
    assert quote_seen_match("completely unrelated words", "12.5 bn", texts) < 0.6
    assert quote_seen_match("", "12.5 bn", texts) == 0.0


def test_quote_seen_match_short_quote_and_page_ocr():
    # The examiner may quote the answer plus surrounding row text.
    assert quote_seen_match("Trading Operating Profit 12.5 bn", "12.5 bn") == 1.0
    # A table answer with no cited sentence can still be checked on page OCR.
    assert quote_seen_match("12.5 bn", "", page_texts=[
        "Trading Operating Profit 12.5 bn 7.5 bn"]) == 1.0
    assert quote_seen_match("15%", "", page_texts=["15 percent growth"]) == 1.0
    ok, match, reason = examiner_confirmed(
        _v(quote_seen="Trading Operating Profit 12.5 bn"), "12.5 bn", [],
        page_texts=["Trading Operating Profit 12.5 bn"])
    assert ok and match == 1.0 and reason == ""


def test_examiner_confirmed_rules():
    ok, match, reason = examiner_confirmed(_v(), "12.5 bn",
                                           ["Revenue 12.5 bn in 2011."])
    assert ok and match == 1.0 and reason == ""
    ok, _m, reason = examiner_confirmed(_v(quote_found=False, quote_seen=""),
                                        "12.5 bn", ["Revenue 12.5 bn."])
    assert not ok and "could not find" in reason
    ok, _m, reason = examiner_confirmed(_v(answer_supported="partly"),
                                        "12.5 bn", ["Revenue 12.5 bn."])
    assert not ok and "partly" in reason
    ok, _m, reason = examiner_confirmed(
        _v(answers_the_question=False, problem="wrong year"),
        "12.5 bn", ["Revenue 12.5 bn."])
    assert not ok and "does not address" in reason
    ok, _m, reason = examiner_confirmed(_v(quote_seen="something else"),
                                        "12.5 bn", ["Revenue 12.5 bn."])
    assert not ok and "does not match" in reason
    ok, _m, reason = examiner_confirmed(
        {"ok": False, "parse_error": "bad"}, "q", [])
    assert not ok and "usable verdict" in reason


def test_examiner_requires_citation_specific_when_configured():
    texts = ["Total revenue 12.5 bn in 2011."]
    ok, _m, reason = examiner_confirmed(
        _v(), "12.5 bn", texts, require_citation_specific=True)
    assert not ok and "did not judge" in reason
    ok, _m, reason = examiner_confirmed(
        _v(citation_specific=False, region_supports=True), "12.5 bn", texts,
        require_citation_specific=True)
    assert not ok and "not the specific evidence" in reason
    ok, _m, reason = examiner_confirmed(
        _v(citation_specific=True, region_supports=False), "12.5 bn", texts,
        require_citation_specific=True)
    assert not ok and "does not support" in reason
    ok, _m, reason = examiner_confirmed(
        _v(citation_specific=True, region_supports=True), "12.5 bn", texts,
        require_citation_specific=True)
    assert ok and reason == ""
    assert verdict_supported_value(
        _v(citation_specific=False, region_supports=True)) == 0.0


def test_verdict_supported_value():
    assert verdict_supported_value(_v()) == 1.0
    assert verdict_supported_value(_v(answer_supported="partly")) == 0.5
    assert verdict_supported_value(_v(answer_supported="no")) == 0.0
    assert verdict_supported_value({"ok": False}) == 0.5


def test_mock_examiner_confirms_only_from_text():
    ex = MockExaminer()
    raw = ex.examine("q", "12.5 bn", "total revenue 12.5 bn",
                     ["Total revenue 12.5 bn in 2011."])
    v = parse_examiner_output(raw)
    assert v["quote_found"] and v["quote_seen"], v
    ok, match, reason = examiner_confirmed(v, "total revenue 12.5 bn",
                                           ["Total revenue 12.5 bn in 2011."])
    assert ok and match >= 0.6 and reason == ""
    raw2 = ex.examine("q", "3.9", "", ["Total revenue 12.5 bn in 2011."])
    v2 = parse_examiner_output(raw2)
    assert not v2["quote_found"] and v2["answer_supported"] == "partly"
