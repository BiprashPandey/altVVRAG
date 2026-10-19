"""Deterministic answer-support check (Phase 3, 2026-10-18).

A quote that matches the cited sentence is NOT sufficient: the quote may be
verbatim while the ANSWER is not in it (Run 6 rows 37/38: answer "2015",
quote "Guaranteed by", entailment 0.0001). The veto: the answer must appear
in the cited text (after number/unit normalization) or be entailed by it
(NLI score above threshold). Unsupported cited answers are not shown as
verified evidence.

Normalization is deliberately modest: thousand separators, decimal commas,
number words (three -> 3), unit aliases (bn -> billion, mn -> million,
k -> thousand, % -> percent).
"""
from __future__ import annotations

import re

_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
}
_UNIT_SUFFIX = re.compile(r"(\d+(?:[.,]\d+)?)\s*(bn|mn|tn|k)\b")
_UNIT_WORDS = {"bn": "billion", "mn": "million", "tn": "trillion", "k": "thousand"}
_STRIP = re.compile(r"[^\w.%\s]+")
_WS = re.compile(r"\s+")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_DECIMAL_COMMA = re.compile(r"(\d),(\d)")


def normalize_numbers(text: str) -> str:
    """Lowercase + number/unit normalization; keeps decimals and %."""
    s = str(text or "").lower()
    s = _THOUSANDS.sub("", s)         # 1,234 -> 1234
    s = _DECIMAL_COMMA.sub(r"\1.\2", s)   # 12,5 -> 12.5
    s = s.replace(",", "")            # leftover separators
    s = _UNIT_SUFFIX.sub(lambda m: f"{m.group(1)} {_UNIT_WORDS[m.group(2)]}", s)
    s = s.replace("%", " percent ")
    s = _STRIP.sub(" ", s)
    toks = []
    for t in s.split():
        t = _NUM_WORDS.get(t, t)
        t = _UNIT_WORDS.get(t, t)
        toks.append(t)
    return _WS.sub(" ", " ".join(toks)).strip()


def answer_supported(answer: str, texts: list[str],
                     entail_min: float | None = None,
                     entail_threshold: float = 0.5) -> tuple[bool, float, str]:
    """Is `answer` contained in (or entailed by) the cited `texts`?

    Returns (supported, score, method) with method in
    {containment, nli, nli_low, no_nli, empty}.
    """
    a = normalize_numbers(answer)
    if not a or a == "unanswerable":
        return False, 0.0, "empty"
    joined = normalize_numbers(" ".join(texts or []))
    if joined and a in joined:
        return True, 1.0, "containment"
    if entail_min is not None:
        if entail_min >= entail_threshold:
            return True, float(entail_min), "nli"
        return False, float(entail_min), "nli_low"
    return False, 0.0, "no_nli"


# ---------------------------------------------------------------------------
# Phase 9 (Run 8): question-aware support.
# The displayed contract is entailment-first: a cited answer counts as
# supported when the NLI entailment of the question+answer statement clears
# the threshold AND the answer's TYPE matches the question AND (for
# comparison questions) both compared values are present in the cited text.
# Containment is only a fallback when no NLI score exists at all.
# ---------------------------------------------------------------------------

_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_COMPARISON = re.compile(
    r"\b(higher|lower|more|less|greater|fewer|better|larger|smaller|"
    r"difference|compared|versus|vs\.?)\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_ALPHA = re.compile(r"[a-z]{2,}")

_TYPE_RULES = (
    ("percent", re.compile(r"\b(percent|percentage|%)", re.I)),
    ("year", re.compile(r"\b(which|what)\s+year\b|\bin\s+which\s+year\b|"
                        r"\bwhat\s+year\b|\byear\b.*\b(19|20)\d{2}\b", re.I)),
    ("date", re.compile(r"\bwhen\b|\bwhat\s+date\b|\bwhich\s+month\b", re.I)),
    ("count", re.compile(r"\bhow\s+many\b|\bnumber\s+of\b|\bcount\b|"
                         r"\bkinds\b|\btypes\b", re.I)),
    ("money", re.compile(r"\bhow\s+much\b|"
                         r"\b(what|which)\s+(is|was|were|are)\b[^?]{0,40}?"
                         r"\b(revenue|profit|cost|price|salary|pay|income|debt|"
                         r"amount|value|worth)\b", re.I)),
    ("name", re.compile(r"\bwho\b|\bname\s+of\b|\btitle\b|"
                        r"\bwhich\s+(company|brand|bank|person|operator|model)\b",
                        re.I)),
    ("location", re.compile(r"\bwhere\b|\bwhich\s+(city|country|region)\b", re.I)),
)

_NUMERIC_KINDS = {"percent", "year", "date", "count", "money"}


def _number_values(text: str) -> list[str]:
    """Normalized numeric surface forms in `text` ("12,5" -> "12.5")."""
    out = []
    for m in _NUM.finditer(str(text or "")):
        out.append(m.group(0).replace(",", ".") if "," in m.group(0)
                   else m.group(0))
    return out


def answer_kind(question: str) -> str:
    """Classify the expected answer type from the question wording."""
    q = str(question or "")
    for kind, rx in _TYPE_RULES:
        if rx.search(q):
            return kind
    return "other"


def answer_type_match(question: str, answer: str) -> bool | None:
    """Does the answer's surface form fit the question's expected type?

    Returns None when the question type is unknown (no constraint).
    Numeric kinds require a digit (after word/number normalization);
    name/location require at least one alphabetic token.
    """
    kind = answer_kind(question)
    if kind in _NUMERIC_KINDS:
        if kind == "year":
            return bool(_YEAR.search(str(answer or ""))
                        or _YEAR.search(normalize_numbers(answer)))
        return bool(_NUM.search(normalize_numbers(answer)))
    if kind in ("name", "location"):
        return bool(_ALPHA.search(normalize_numbers(answer)))
    return None


def is_comparison(question: str) -> bool:
    """Comparison-style question (two values must be visible side by side)."""
    return bool(_COMPARISON.search(str(question or "")))


def comparison_values_present(question: str, answer: str,
                              texts: list[str]) -> bool:
    """For comparison questions: are the compared values in the cited text?

    Needs the numbers mentioned by the question or answer to appear
    (normalized) in the cited text. Fewer than 2 candidate numbers exist ->
    nothing to compare -> True.
    """
    needed = set(_number_values(question)) | set(_number_values(answer))
    needed = {n for n in needed if len(n) <= 8}
    if len(needed) < 2:
        return True
    joined = normalize_numbers(" ".join(texts or []))
    present = set(_number_values(joined))
    return len(needed & present) >= 2


def qa_support(question: str, answer: str, texts: list[str],
               entail_min: float | None = None,
               entail_threshold: float = 0.5,
               type_check: bool = True,
               comparison_check: bool = True) -> tuple[bool, float, str]:
    """Question-aware support verdict.

    Order: hard question-aware vetoes first (answer type, comparison values),
    then verbatim containment (strongest, kept from the Phase 3 check), then
    NLI entailment as the paraphrase fallback.
    Returns (supported, score, method) with method in
    {containment+type, entail+type, nli_low, type_mismatch,
     missing_comparison_values, no_nli, empty}.
    """
    a = normalize_numbers(answer)
    if not a or a == "unanswerable":
        return False, 0.0, "empty"
    if question and type_check:
        match = answer_type_match(question, answer)
        if match is False:
            return False, float(entail_min or 0.0), "type_mismatch"
    if question and comparison_check and is_comparison(question):
        if not comparison_values_present(question, answer, texts):
            return False, float(entail_min or 0.0), "missing_comparison_values"
    joined = normalize_numbers(" ".join(texts or []))
    if joined and a in joined:
        return True, 1.0, "containment+type"
    if entail_min is not None:
        if entail_min >= entail_threshold:
            return True, float(entail_min), "entail+type"
        return False, float(entail_min), "nli_low"
    return False, 0.0, "no_nli"
