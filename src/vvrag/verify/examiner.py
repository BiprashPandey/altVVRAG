"""Independent examiner (Run 9, guideline s4.3).

The reader's claim is SHOWN to a second, different-family model together with
the evidence image(s). The examiner finds the quote itself and judges whether
the claim answers the question. VERIFIED requires its confirmation
(guideline s7); a failed/unavailable examiner means "not confident enough",
never a veto and never a silent fallback to another family.

Determinism:
- `parse_examiner_output` is strict JSON parsing with a tolerant fallback;
- `examiner_confirmed` is a pure function over the verdict + claimed quote +
  OCR texts (the `quote_seen` cross-check, guideline s4.3);
- `MockExaminer` is a documented test stand-in that examines the OCR text
  deterministically (used by mock runs/tests only; it has no image).
"""
from __future__ import annotations

import json
import re

from vvrag.reader.claims import normalize_answer
from vvrag.verify.claim_verifier import parse_claims
from vvrag.verify.quote_check import quote_match_score

DEFAULT_QUOTE_SEEN_THRESHOLD = 0.6
_ANSWER_SUPPORTED = ("yes", "partly", "no")


def parse_examiner_output(raw: str) -> dict:
    """Parse the examiner's strict JSON. Returns a normalized verdict dict.

    {"ok": bool, "quote_found": bool, "quote_seen": str,
     "answer_supported": "yes|partly|no"|"", "answers_the_question": bool|None,
     "problem": str, "parse_error": str}
    """
    verdict = {"ok": False, "quote_found": False, "quote_seen": "",
               "answer_supported": "", "answers_the_question": None,
               "citation_specific": None, "region_supports": None,
               "claims": [], "problem": "", "parse_error": ""}
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = "\n".join(ln for ln in text.splitlines()
                         if not ln.strip().startswith("```")).strip()
    obj = None
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict):
        verdict["parse_error"] = "examiner output was not JSON"
        return verdict
    verdict["ok"] = True
    verdict["quote_found"] = bool(obj.get("quote_found", False))
    verdict["quote_seen"] = str(obj.get("quote_seen", "") or "")
    sup = str(obj.get("answer_supported", "") or "").strip().lower()
    if sup not in _ANSWER_SUPPORTED:
        sup = ""
    verdict["answer_supported"] = sup
    atq = obj.get("answers_the_question")
    verdict["answers_the_question"] = (
        bool(atq) if isinstance(atq, bool) else None)
    for key in ("citation_specific", "region_supports"):
        val = obj.get(key)
        verdict[key] = bool(val) if isinstance(val, bool) else None
    verdict["claims"] = parse_claims(obj.get("claims"))
    verdict["problem"] = str(obj.get("problem", "") or "")[:400]
    return verdict


def _canonical_text(text: str) -> str:
    """Number/unit-aware normalization used for short quote matching."""
    try:
        from vvrag.verify.support import normalize_numbers

        return normalize_numbers(text)
    except Exception:  # pragma: no cover - defensive for partial installs
        return normalize_answer(text)


def _canonical_tokens(text: str) -> list[str]:
    return [t for t in _canonical_text(text).split() if t]


def _token_contained(needle: list[str], haystack: list[str]) -> bool:
    if not needle or not haystack:
        return False
    hay = set(haystack)
    return all(t in hay for t in needle)


def _quote_seen_score(
    qs: str,
    claimed_quote: str = "",
    ocr_texts: list[str] | None = None,
    ocr_normalize: bool = True,
    page_texts: list[str] | None = None,
    region_texts: list[str] | None = None,
) -> float:
    texts = [t for t in (ocr_texts or []) if t]
    pages = [t for t in (page_texts or []) if t]
    regions = [t for t in (region_texts or []) if t]
    candidates = texts + pages + regions

    qs_tokens = _canonical_tokens(qs)
    claimed_tokens = _canonical_tokens(claimed_quote) if claimed_quote else []

    # Fast path: token-set containment. This is safer than raw substring
    # matching for numbers ("12.5" must not match inside "112.5").
    if qs_tokens and any(_token_contained(qs_tokens, _canonical_tokens(t))
                         for t in candidates):
        return 1.0

    # The examiner may quote the answer plus context: if it contains the
    # reader's claimed quote verbatim, the cross-check is satisfied.
    if _token_contained(claimed_tokens, qs_tokens):
        return 1.0

    # Short quotes are brittle under token-F1: use set containment instead.
    if len(qs_tokens) <= 6:
        if _token_contained(claimed_tokens, qs_tokens):
            return 1.0
        for text in candidates:
            if _token_contained(qs_tokens, _canonical_tokens(text)):
                return 1.0

    best = 0.0
    if claimed_quote:
        best = quote_match_score(qs, [claimed_quote],
                                 ocr_normalize=ocr_normalize)
    if candidates:
        best = max(best, quote_match_score(qs, candidates,
                                           ocr_normalize=ocr_normalize))
    return float(best)


def quote_seen_match(
    quote_seen: str,
    claimed_quote: str = "",
    ocr_texts: list[str] | None = None,
    ocr_normalize: bool = True,
    page_texts: list[str] | None = None,
    region_texts: list[str] | None = None,
) -> float:
    """How well does the examiner's seen-quote line up with the claim/OCR?

    Best of: normalized containment in the reader's quote, cited sentence
    text, full page OCR, or the reader's declared region text. Short quotes
    (<=6 tokens) use token-set containment because token-F1 is noisy on them.
    """
    qs = str(quote_seen or "").strip()
    if not qs:
        return 0.0
    return _quote_seen_score(qs, claimed_quote, ocr_texts, ocr_normalize,
                             page_texts, region_texts)


def examiner_confirmed(
    verdict: dict,
    claimed_quote: str = "",
    ocr_texts: list[str] | None = None,
    threshold: float = DEFAULT_QUOTE_SEEN_THRESHOLD,
    ocr_normalize: bool = True,
    page_texts: list[str] | None = None,
    region_texts: list[str] | None = None,
    require_citation_specific: bool = False,
) -> tuple[bool, float, str]:
    """(confirmed, quote_seen_match, reason).

    Confirmed iff: valid verdict, quote found, answer supported == yes,
    answers_the_question true, and the seen quote cross-checks against the
    reader's quote or the cited OCR text (guideline s4.3). Otherwise the
    returned reason explains why not.
    """
    if not verdict or not verdict.get("ok"):
        return False, 0.0, "examiner did not return a usable verdict"
    if not verdict.get("quote_found"):
        return False, 0.0, ("the examiner could not find the claimed answer in "
                            "the evidence image")
    if verdict.get("answer_supported") != "yes":
        sup = verdict.get("answer_supported") or "unclear"
        reason = (f"the examiner judged the answer '{sup}'"
                  + (f" ({verdict['problem']})" if verdict.get("problem") else ""))
        return False, 0.0, reason
    if verdict.get("answers_the_question") is False:
        reason = ("the examiner says the answer does not address the question"
                  + (f" ({verdict['problem']})" if verdict.get("problem") else ""))
        return False, 0.0, reason
    if require_citation_specific:
        if verdict.get("citation_specific") is False:
            reason = ("the examiner says the highlighted region is not the "
                      "specific evidence for the claim"
                      + (f" ({verdict['problem']})" if verdict.get("problem") else ""))
            return False, 0.0, reason
        if verdict.get("region_supports") is False:
            reason = ("the examiner says the highlighted region does not support "
                      "the claim"
                      + (f" ({verdict['problem']})" if verdict.get("problem") else ""))
            return False, 0.0, reason
        if (verdict.get("citation_specific") is None
                or verdict.get("region_supports") is None):
            return False, 0.0, ("the examiner did not judge whether the cited "
                                "region specifically supports the claim")
    match = quote_seen_match(verdict.get("quote_seen", ""), claimed_quote,
                             ocr_texts, ocr_normalize, page_texts, region_texts)
    if match < threshold:
        reason = ("the examiner's quote does not match the cited text "
                  f"(match {match:.2f} < {threshold:.2f})")
        return False, match, reason
    return True, match, ""


def verdict_supported_value(verdict: dict) -> float:
    """Examiner support as a gate feature: yes=1, partly=0.5, no/unknown=0."""
    if not verdict or not verdict.get("ok"):
        return 0.5
    if (verdict.get("citation_specific") is False
            or verdict.get("region_supports") is False):
        return 0.0
    sup = verdict.get("answer_supported")
    if sup == "yes":
        return 1.0
    if sup == "partly":
        return 0.5
    if sup == "no":
        return 0.0
    return 0.5


class MockExaminer:
    """Deterministic examiner stand-in for tests/mock runs (no image access).

    It examines the OCR text of the cited sentences: the claimed quote (or
    answer) must be found there verbatim after normalization. This mirrors a
    functioning examiner over what OCR can see and is documented as a mock.
    """

    served_by = "mock/examiner"

    def examine(
        self,
        question: str,
        answer: str,
        claimed_quote: str = "",
        ocr_texts: list[str] | None = None,
        **_kwargs,
    ) -> str:
        texts = [t for t in (ocr_texts or []) if t]
        joined = normalize_answer(" ".join(texts))
        quote = str(claimed_quote or "").strip()
        n_quote = normalize_answer(quote)
        n_ans = normalize_answer(answer)
        if quote and n_quote and n_quote in joined:
            return json.dumps({
                "quote_found": True, "quote_seen": quote,
                "answer_supported": "yes", "answers_the_question": True,
                "citation_specific": True, "region_supports": True,
                "problem": "",
            })
        if n_ans and n_ans in joined:
            idx = joined.find(n_ans)
            seen = " ".join(joined[max(0, idx - 20):idx + len(n_ans) + 20].split())
            return json.dumps({
                "quote_found": True, "quote_seen": seen,
                "answer_supported": "yes", "answers_the_question": True,
                "citation_specific": True, "region_supports": True,
                "problem": "",
            })
        if answer and normalize_answer(answer) == "unanswerable":
            return json.dumps({
                "quote_found": False, "quote_seen": "",
                "answer_supported": "no", "answers_the_question": False,
                "citation_specific": False, "region_supports": False,
                "problem": "nothing relevant on the page",
            })
        return json.dumps({
            "quote_found": False, "quote_seen": "",
            "answer_supported": "partly", "answers_the_question": True,
            "citation_specific": False, "region_supports": False,
            "problem": "the answer is not in the OCR text of the cited page",
        })
