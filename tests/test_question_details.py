"""Run 9 (guideline s4.4): question-detail grounding (years/dates/units/entities)."""
from vvrag.verify.question_details import (check_question_details, nearest_year,
                                           question_details)


def test_question_details_extraction():
    d = question_details("What was Nestle's organic growth in 2021, in percent?")
    assert d["years"] == ["2021"]
    assert "Nestle" in d["entities"] and "percent" not in [e.lower() for e in d["entities"]]
    d2 = question_details("How many factories did Acme open in March 2015?")
    assert d2["years"] == ["2015"]
    assert any(x.startswith("march") for x in d2["dates"])
    assert "Acme" in d2["entities"]
    assert question_details("Compare the 12.5 bn and 7.5 bn figures.")["units"] != []


def test_check_question_details_ok_and_missing():
    ok = check_question_details(
        "What was Nestle's growth in 2011?",
        ["Nestle grew 4.1 percent in 2011."], corpus_years={"2011"})
    assert ok["ok"] and not ok["warnings"]
    bad = check_question_details(
        "What was Nestle's growth in 2021?",
        ["Nestle grew 4.1 percent in 2011."],
        corpus_years={"2009", "2011", "2013"})
    assert not bad["ok"]
    assert "2021" in bad["missing"]
    assert "2013" in bad["warnings"][0]  # nearest year message
    assert "does not appear anywhere" in bad["warnings"][0]


def test_check_question_details_examiner_quote_rescues():
    # The entity is visible to the examiner but absent from the cited OCR text.
    rescued = check_question_details(
        "What did the PowerBar label show?",
        ["Only the chart labels are OCR-visible here."],
        quote_seen="PowerBar 65g protein",
        corpus_years=set())
    assert rescued["ok"], rescued


def test_check_question_details_off_mode():
    off = check_question_details("What was growth in 2021?", ["2011 only"],
                                 mode="off")
    assert off["checked"] is False and off["ok"]


def test_nearest_year():
    assert nearest_year("2021", {"2010", "2013", "2018"}) == "2018"
    assert nearest_year("1999", set()) is None
