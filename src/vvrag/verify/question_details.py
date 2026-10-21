"""Question-detail grounding (Run 9, guideline s4.4).

The reader must not silently drop a detail the question asked for: a year, a
date, a unit attached to a number, or a named entity. If the detail is not in
the evidence, the answer can still be shown (always answer), but it can never
be VERIFIED and the UI says which detail is missing.

Checks are deterministic:
- normalize with `support.normalize_numbers` (thousand separators, decimal
  commas, bn/mn/tn/k, percent);
- years must appear in the evidence; if absent from the whole corpus too, say
  so and give the nearest year present in the corpus;
- dates (month + day/year) same treatment;
- number+unit details must appear together;
- named entities (capitalized tokens not in question stopwords) checked
  against the evidence and, when available, the corpus vocabulary.
"""
from __future__ import annotations

import re
import unicodedata

from vvrag.verify.support import normalize_numbers

_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_MONTHS = (
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec",
)
_MONTH_RX = re.compile(r"\b(" + "|".join(_MONTHS) + r")\b", re.I)
_NUMBER_UNIT = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(%|percent|percentage|bn|mn|tn|billion|million|"
    r"trillion|thousand|k)\b", re.I)
_WORD = re.compile(r"[A-Z\u00c0-\u00de][\w\u00c0-\u00ff'&.\-]*")
_STOP_ENTITIES = {
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "in", "on", "at", "for", "from", "by", "with", "the", "a", "an", "of",
    "and", "or", "to", "is", "was", "were", "are", "be", "been", "did",
    "does", "do", "has", "have", "had", "this", "that", "these", "those",
    "name", "company", "group", "report", "year", "many", "much", "number",
    "according", "based", "please", "tell", "give", "list", "describe",
    "during", "after", "before", "between", "over", "under", "its", "their",
    "it", "he", "she", "they", "we", "you", "i", "as", "if", "than", "then",
    "also", "increase", "decrease", "grow", "growth", "revenue", "profit",
    "sales", "margin", "value", "total", "average", "percent", "percentage",
    "quarter", "month", "date", "city", "country", "region", "brand",
    "person", "operator", "model", "figure", "amount", "cost", "price",
    "salary", "pay", "income", "debt", "worth", "title", "kind", "types",
    "third", "first", "second", "fourth", "fifth", "largest", "smallest",
    "highest", "lowest", "best", "worst", "more", "less", "higher", "lower",
}


def question_details(question: str) -> dict:
    """Extract the checkable details from a question (deterministic)."""
    q = str(question or "")
    years = [m.group(0) for m in _YEAR.finditer(q)]
    dates = []
    for m in _MONTH_RX.finditer(q):
        tok = m.group(1).lower()
        tail = q[m.end():m.end() + 12]
        day = re.search(r"\b(\d{1,2})\b", tail)
        yr = re.search(r"\b((?:19|20)\d{2})\b", tail)
        label = tok
        if day:
            label += " " + day.group(1)
        if yr:
            label += " " + yr.group(1)
        dates.append(label)
    units = []
    for m in _NUMBER_UNIT.finditer(q):
        units.append(normalize_numbers(m.group(0)))
    entities = []
    for m in _WORD.finditer(q):
        tok = re.sub(r"['\u2019]s$", "", m.group(0))
        if not tok or tok.lower() in _STOP_ENTITIES:
            continue
        if len(tok) < 2:
            continue
        if tok.isupper() and len(tok) <= 2:  # Q3, A1 ... not an entity
            continue
        if tok not in entities:
            entities.append(tok)
    return {
        "years": sorted(set(years)),
        "dates": dates,
        "units": sorted(set(units)),
        "entities": entities,
    }


def _fold(text: str) -> str:
    """Lowercase + strip diacritics (OCR often drops accents)."""
    s = unicodedata.normalize("NFKD", str(text or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"['\u2019]s\b", "", s.lower())


def _present(detail: str, haystack_norm: str, haystack_fold: str = "") -> bool:
    d = normalize_numbers(detail)
    if not d:
        return False
    if d in haystack_norm:
        return True
    return _fold(d) in (haystack_fold or _fold(haystack_norm))


def nearest_year(year: str, corpus_years) -> str | None:
    """Closest year present in the corpus (for the 'nearest match' message)."""
    if not corpus_years:
        return None
    try:
        target = int(year)
    except (TypeError, ValueError):
        return None
    best = None
    for y in corpus_years:
        try:
            yi = int(y)
        except (TypeError, ValueError):
            continue
        if best is None or abs(yi - target) < abs(int(best) - target):
            best = str(yi)
    return best


def check_question_details(
    question: str,
    texts: list[str],
    quote_seen: str = "",
    corpus_years=None,
    corpus_entities=None,
    mode: str = "warn",
) -> dict:
    """Check question details against the evidence; warn, never silently drop.

    Returns {ok, missing, warnings, details, checked}. `ok=False` means at
    least one asked-for detail is not visible in the evidence. `quote_seen`
    is the examiner's quote (can rescue details visible on the image but not
    in OCR).
    """
    cfg_mode = str(mode or "warn").lower()
    if cfg_mode in ("off", "false", "0", "none"):
        return {"ok": True, "missing": [], "warnings": [], "details": {},
                "checked": False}
    details = question_details(question)
    evidence = normalize_numbers(" ".join(texts or []) + " " +
                                 (quote_seen or ""))
    evidence_fold = _fold(" ".join(texts or []) + " " + (quote_seen or ""))
    corpus_years = set(str(y) for y in (corpus_years or []))
    corpus_entities_norm = set(_fold(e) for e in (corpus_entities or []))
    missing: list[str] = []
    warnings: list[str] = []

    for year in details["years"]:
        if _present(year, evidence, evidence_fold):
            continue
        missing.append(year)
        if corpus_years and year not in corpus_years:
            near = nearest_year(year, corpus_years)
            if near:
                warnings.append(
                    f"the question asks about {year}, which does not appear "
                    f"anywhere in this document set (nearest year present: "
                    f"{near}).")
            else:
                warnings.append(
                    f"the question asks about {year}, which does not appear "
                    f"anywhere in this document set.")
        else:
            warnings.append(
                f"the year {year} from the question is not in the cited "
                f"evidence.")
    for date in details["dates"]:
        if _present(date, evidence, evidence_fold):
            continue
        missing.append(date)
        warnings.append(f"the date '{date}' from the question is not in the "
                        f"cited evidence.")
    for unit in details["units"]:
        if _present(unit, evidence, evidence_fold):
            continue
        missing.append(unit)
        warnings.append(
            f"the unit/value '{unit}' from the question is not in the cited "
            f"evidence.")
    for ent in details["entities"]:
        if _present(ent, evidence, evidence_fold):
            continue
        if corpus_entities_norm and _fold(ent) not in corpus_entities_norm:
            warnings.append(
                f"the named entity '{ent}' does not appear anywhere in this "
                f"document set.")
        else:
            warnings.append(
                f"the named entity '{ent}' from the question is not in the "
                f"cited evidence.")
        missing.append(ent)

    return {
        "ok": not missing,
        "missing": missing,
        "warnings": warnings,
        "details": details,
        "checked": True,
        "mode": cfg_mode,
    }


def corpus_detail_vocab(con) -> tuple[set, set]:
    """Years + capitalized tokens present in all stored sentences.

    Used to distinguish "not on this page" from "not anywhere in the corpus"
    and to offer the nearest year. Best-effort: a missing table returns empty
    sets (legacy index / tests).
    """
    years: set[str] = set()
    entities: set[str] = set()
    try:
        rows = con.execute("SELECT text FROM sentences").fetchall()
    except Exception:  # noqa: BLE001
        try:
            rows = con.execute("SELECT text FROM word_boxes").fetchall()
        except Exception:  # noqa: BLE001
            return years, entities
    for (text,) in rows:
        t = str(text or "")
        for m in _YEAR.finditer(t):
            years.add(m.group(0))
        for m in _WORD.finditer(t):
            tok = re.sub(r"['\u2019]s$", "", m.group(0))
            if len(tok) >= 2 and tok.lower() not in _STOP_ENTITIES:
                entities.add(tok)
    return years, entities
