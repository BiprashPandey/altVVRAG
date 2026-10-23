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
    assert bad["impossible_years"] == ["2021"]
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


def test_check_question_details_soft_entity_warns_only():
    # Generic/tagline words are skipped; a real entity absent only from the
    # cited sentence but present on the page does not block verification.
    generic = check_question_details(
        "What did Good Food Life say about growth?", ["Growth was 3.6%."],
        mode="warn")
    assert generic["ok"] and not generic["hard_missing"]
    page_rescue = check_question_details(
        "What was Nestle's growth in 2011?", ["Growth was 3.6%."],
        page_texts=["Nestle annual report 2011"], mode="warn")
    assert page_rescue["ok"] and not page_rescue["hard_missing"]


def test_check_question_details_hard_year_blocks_even_on_page_miss():
    bad = check_question_details(
        "What was Nestle's growth in 2010?",
        ["Growth was 3.6% in 2011."], page_texts=["Nestle annual report 2011"],
        mode="warn")
    assert not bad["ok"] and "2010" in bad["hard_missing"]
    assert bad["soft_ok"] is True
    assert bad["impossible_years"] == []  # corpus vocab unknown here


def test_check_question_details_off_mode():
    off = check_question_details("What was growth in 2021?", ["2011 only"],
                                 mode="off")
    assert off["checked"] is False and off["ok"]


def test_nearest_year():
    assert nearest_year("2021", {"2010", "2013", "2018"}) == "2018"
    assert nearest_year("1999", set()) is None


def test_check_question_details_duration_scope():
    # "20-year" is absent from the corpus; the cited sentence says "10-year".
    bad = check_question_details(
        "What is the 20-year annual average?",
        ["10-year annual averages: 3.6%"], corpus_durations={"10-year"})
    assert not bad["ok"]
    assert bad["impossible_durations"] == ["20-year"]
    assert "20-year" in bad["hard_missing"]
    # A matching duration is fine.
    ok = check_question_details(
        "What is the 10-year annual average?",
        ["10-year annual averages: 3.6%"], corpus_durations={"10-year"})
    assert ok["ok"] and not ok["impossible_durations"]


def test_section_reference_is_checked_in_cited_unit():
    # The false-verified frozen probe: Old Act Section 4 -> New Act Section 12.
    # The cited row contains 4 and 11, not 12. This is a question constraint,
    # not a generic page-number lookup.
    bad = check_question_details(
        "What is Old Act Section 4 in the New Act Section 12?",
        ["4 | Award | 11 | 23/30/."], mode="warn")
    assert not bad["ok"]
    assert "section 12" in bad["hard_missing"]
    assert "section 4" not in bad["hard_missing"]
    assert any("section 12" in w for w in bad["warnings"])

    # A question that only asks for Section 4 is satisfied by the same row.
    ok = check_question_details(
        "What is Old Act Section 4 in the New Act Section?",
        ["4 | Award | 11 | 23/30/."], mode="warn")
    assert ok["ok"] and not ok["hard_missing"]

    # Explicit section wording in the cited evidence also passes.
    explicit = check_question_details(
        "What does New Act Section 12 say?",
        ["New Act Section 12 covers awards."], mode="warn")
    assert explicit["ok"]
    assert question_details("Old Act Section 4 in New Act Section 12")[
        "sections"] == ["4", "12"]
