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
               "problem": "", "parse_error": ""}
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
    verdict["problem"] = str(obj.get("problem", "") or "")[:400]
    return verdict


def quote_seen_match(
    quote_seen: str,
    claimed_quote: str = "",
    ocr_texts: list[str] | None = None,
    ocr_normalize: bool = True,
) -> float:
    """How well does the examiner's seen-quote line up with the claim/OCR?

    Best of: fuzzy match against the reader's quote; verbatim containment in
    the cited OCR texts (examiner saw words the page really carries).
    """
    qs = str(quote_seen or "").strip()
    if not qs:
        return 0.0
    best = 0.0
    if claimed_quote:
        best = quote_match_score(qs, [claimed_quote],
                                 ocr_normalize=ocr_normalize)
    if ocr_texts:
        contained = quote_match_score(qs, ocr_texts, ocr_normalize=ocr_normalize)
        best = max(best, contained)
        n_qs = normalize_answer(qs)
        if n_qs and any(n_qs in normalize_answer(t) for t in ocr_texts if t):
            best = 1.0
    return float(best)


def examiner_confirmed(
    verdict: dict,
    claimed_quote: str = "",
    ocr_texts: list[str] | None = None,
    threshold: float = DEFAULT_QUOTE_SEEN_THRESHOLD,
    ocr_normalize: bool = True,
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
    match = quote_seen_match(verdict.get("quote_seen", ""), claimed_quote,
                             ocr_texts, ocr_normalize)
    if match < threshold:
        reason = ("the examiner's quote does not match the cited text "
                  f"(match {match:.2f} < {threshold:.2f})")
        return False, match, reason
    return True, match, ""


def verdict_supported_value(verdict: dict) -> float:
    """Examiner support as a gate feature: yes=1, partly=0.5, no/unknown=0."""
    if not verdict or not verdict.get("ok"):
        return 0.5
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
                "problem": "",
            })
        if n_ans and n_ans in joined:
            idx = joined.find(n_ans)
            seen = " ".join(joined[max(0, idx - 20):idx + len(n_ans) + 20].split())
            return json.dumps({
                "quote_found": True, "quote_seen": seen,
                "answer_supported": "yes", "answers_the_question": True,
                "problem": "",
            })
        if answer and normalize_answer(answer) == "unanswerable":
            return json.dumps({
                "quote_found": False, "quote_seen": "",
                "answer_supported": "no", "answers_the_question": False,
                "problem": "nothing relevant on the page",
            })
        return json.dumps({
            "quote_found": False, "quote_seen": "",
            "answer_supported": "partly", "answers_the_question": True,
            "problem": "the answer is not in the OCR text of the cited page",
        })
