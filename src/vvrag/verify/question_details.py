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
_DURATION = re.compile(r"\b(\d{1,3})\s*[- ]\s*years?\b", re.I)
_SECTION_REF = re.compile(
    r"(?:\bsections?\b|\bsec\.?|\u00a7)\s*(\d+[A-Za-z]?)", re.I)
_WORD = re.compile(r"[A-Z\u00c0-\u00de][\w\u00c0-\u00ff'&.\-]*")
_GENERIC_ENTITIES = {
    # Taglines, slide furniture, and generic business/UI words. These are
    # capitalized in questions but are not useful question constraints.
    "good", "food", "life", "key", "notes", "tam", "enterprise", "go",
    "market", "activity", "activities", "email", "emails", "social",
    "network", "networks", "minutes", "hours", "patents", "transformation",
    "center", "centre", "overview", "report", "slide", "page", "chart",
    "table", "figure", "data", "result", "results", "source", "sources",
}
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
    durations = sorted({f"{m.group(1)}-year" for m in _DURATION.finditer(q)})
    sections = sorted({m.group(1) for m in _SECTION_REF.finditer(q)},
                      key=lambda x: (int(re.match(r"\d+", x).group(0)),
                                     x.lower()))
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
        "durations": durations,
        "sections": sections,
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


def _evidence_blob(parts: list[str]) -> tuple[str, str]:
    raw = " ".join(p for p in parts if p)
    return normalize_numbers(raw), _fold(raw)


def check_question_details(
    question: str,
    texts: list[str],
    quote_seen: str = "",
    corpus_years=None,
    corpus_entities=None,
    corpus_durations=None,
    mode: str = "warn",
    page_texts: list[str] | None = None,
    deck_texts: list[str] | None = None,
) -> dict:
    """Check question details against the evidence.

    Hard details (years, dates, numbers/units) must be visible on the evidence
    page; a missing hard detail blocks VERIFIED. Named entities are soft: they
    are checked against the page, deck, and corpus text, but a miss is only a
    warning under the default ``warn`` mode. This avoids blocking a correct
    answer because a company name sits in a header or deck title rather than
    inside the one cited sentence.

    Returns {ok, hard_ok, soft_ok, missing, hard_missing, soft_missing,
    warnings, details, checked}. ``quote_seen`` is the examiner's quote and
    can rescue details visible on the image but not in OCR.
    """
    cfg_mode = str(mode or "warn").lower()
    if cfg_mode in ("off", "false", "0", "none"):
        return {"ok": True, "hard_ok": True, "soft_ok": True,
                "missing": [], "hard_missing": [], "soft_missing": [],
                "impossible_years": [], "impossible_durations": [],
                "warnings": [], "details": {}, "checked": False,
                "mode": cfg_mode}

    details = question_details(question)
    cited = [t for t in (texts or []) if t]
    page = [t for t in (page_texts or []) if t]
    deck = [t for t in (deck_texts or []) if t]
    quote = [quote_seen] if quote_seen else []

    # Hard details: the evidence page is the right scope. The cited sentence
    # alone is too narrow (years often live in a header next to the answer).
    hard_parts = cited + page + quote
    hard_norm, hard_fold = _evidence_blob(hard_parts)
    cited_norm = normalize_numbers(" ".join(cited))

    # Entities: the whole evidence page/deck is acceptable. A company name in
    # the deck title is still a valid constraint, even if the cited sentence
    # does not repeat it.
    soft_parts = cited + page + deck + quote
    soft_norm, soft_fold = _evidence_blob(soft_parts)

    corpus_years = set(str(y) for y in (corpus_years or []))
    corpus_entities_norm = set(_fold(e) for e in (corpus_entities or []))
    corpus_durations = set(str(d) for d in (corpus_durations or []))
    hard_missing: list[str] = []
    soft_missing: list[str] = []
    impossible_years: list[str] = []
    impossible_durations: list[str] = []
    warnings: list[str] = []

    for year in details["years"]:
        if _present(year, hard_norm, hard_fold):
            continue
        hard_missing.append(year)
        if corpus_years and year not in corpus_years:
            impossible_years.append(year)
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
                f"the year {year} from the question is not in the evidence "
                f"page.")

    for date in details["dates"]:
        if _present(date, hard_norm, hard_fold):
            continue
        hard_missing.append(date)
        warnings.append(f"the date '{date}' from the question is not in the "
                        f"evidence page.")

    for unit in details["units"]:
        if _present(unit, hard_norm, hard_fold):
            continue
        hard_missing.append(unit)
        warnings.append(
            f"the unit/value '{unit}' from the question is not in the "
            f"evidence page.")

    for section in details.get("sections", []):
        # A section reference is a question constraint, not a generic number:
        # "Section 4 in the New Act Section 12" needs both 4 and 12 in the
        # cited unit. Checking the page instead would pass because the page
        # contains many unrelated numbers.
        if not cited:
            continue
        token = re.compile(rf"(?<![\w.]){re.escape(section)}(?![\w.])", re.I)
        if token.search(cited_norm):
            continue
        hard_missing.append(f"section {section}")
        warnings.append(
            f"the question references section {section}, but the cited "
            "evidence does not contain that section number.")

    for duration in details["durations"]:
        if _present(duration, hard_norm, hard_fold):
            continue
        hard_missing.append(duration)
        if corpus_durations and duration not in corpus_durations:
            impossible_durations.append(duration)
            warnings.append(
                f"the question asks for a {duration} average/period, which "
                f"does not appear anywhere in this document set.")
        else:
            warnings.append(
                f"the duration '{duration}' from the question is not in the "
                f"evidence page.")

    for ent in details["entities"]:
        if _fold(ent) in _GENERIC_ENTITIES:
            continue
        if _present(ent, soft_norm, soft_fold):
            continue
        soft_missing.append(ent)
        if corpus_entities_norm and _fold(ent) not in corpus_entities_norm:
            warnings.append(
                f"the named entity '{ent}' does not appear anywhere in this "
                f"document set.")
        else:
            warnings.append(
                f"the named entity '{ent}' from the question is not in the "
                f"evidence page or deck title.")

    hard_ok = not hard_missing
    soft_ok = not soft_missing
    # "warn" is the default: hard details gate VERIFIED, entities only warn.
    # "block"/"strict" also gates on entity misses; kept for reproducibility.
    strict_soft = cfg_mode in ("block", "strict", "on", "true", "1")
    return {
        "ok": bool(hard_ok and (soft_ok if strict_soft else True)),
        "hard_ok": bool(hard_ok),
        "soft_ok": bool(soft_ok),
        "missing": hard_missing + soft_missing,
        "hard_missing": hard_missing,
        "soft_missing": soft_missing,
        "impossible_years": impossible_years,
        "impossible_durations": impossible_durations,
        "warnings": warnings,
        "details": details,
        "checked": True,
        "mode": cfg_mode,
    }


def corpus_detail_vocab(con) -> tuple[set, set, set]:
    """Years + entity tokens + durations present in all stored sentences.

    Used to distinguish "not on this page" from "not anywhere in the corpus"
    and to offer the nearest year. Best-effort: a missing table returns empty
    sets (legacy index / tests).
    """
    years: set[str] = set()
    entities: set[str] = set()
    durations: set[str] = set()
    try:
        rows = con.execute("SELECT text FROM sentences").fetchall()
    except Exception:  # noqa: BLE001
        try:
            rows = con.execute("SELECT text FROM word_boxes").fetchall()
        except Exception:  # noqa: BLE001
            return years, entities, durations
    for (text,) in rows:
        t = str(text or "")
        for m in _YEAR.finditer(t):
            years.add(m.group(0))
        for m in _DURATION.finditer(t):
            durations.add(f"{m.group(1)}-year")
        for m in _WORD.finditer(t):
            tok = re.sub(r"['\u2019]s$", "", m.group(0))
            if len(tok) >= 2 and tok.lower() not in _STOP_ENTITIES:
                entities.add(tok)
    return years, entities, durations
