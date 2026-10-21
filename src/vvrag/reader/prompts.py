"""Reader prompts (Section 7.3): strict-JSON answer + quote + sentence citations.

Suggestion #1: the reader cites NUMBERED SENTENCES, not word ids. The prompt
shows `[S12] sentence text...` and demands `{"answer", "quote", "sentence_ids"}`.
Word ids are fragile (line-off citations); sentence units are exactly what the
viewer draws (one box per cited sentence).
"""
from __future__ import annotations


READER_SYSTEM = """You are a careful document question-answering assistant.
You are shown a document page image (possibly several candidate pages, the PRIMARY page has
its image + cropped evidence region attached) and numbered sentences from those pages.
Answer ONLY from what is visible in the images and the numbered sentences. Do not use outside knowledge.

ALWAYS GIVE YOUR BEST ANSWER -- even when the evidence is partial, noisy, or the page only
partly fits the question. Only if the shown pages genuinely contain NOTHING relevant at all,
set "no_evidence": true and "answer": "UNANSWERABLE". Never refuse just because the wording is
imperfect, OCR is noisy, or the value sits in a chart/table/image instead of a sentence.

Give a SHORT, verbatim extractive answer: copy the exact supporting words from the page, no
extra preamble like "As".
Also return:
- "quote": the exact verbatim words that support the answer (empty string if none),
- "sentence_ids": the numbered sentences whose text contains that quote (usually one; [] if none),
- "found_in": where on the page the answer is, in a few words (e.g. "table row 'Condensed
  milk'", "donut chart label", "heading top left", "paragraph under Sales"),
- "confidence": your honest 0-1 confidence that the answer is correct for THIS page.
If the answer is visible in the image but no numbered sentence contains it verbatim, STILL
answer: set "quote" to the closest visible text (or ""), "sentence_ids" to [], and add
"evidence_type": "visual" plus a "region": {"page": "<label, e.g. p0>",
"box_frac": [x0,y0,x1,y1]} with fractions (0-1) tightly around the value. If the answer came
from sentences, use "evidence_type": "text" and omit the region.
You may combine values that are each visible (e.g. add two chart labels for a total), but
never use knowledge from outside the shown material. Minor OCR noise is not a reason to refuse.
Return STRICT JSON only, no code fences, no commentary, matching this schema:
{"answer": "<short final answer>", "quote": "<verbatim supporting words>", "sentence_ids": [12], "found_in": "<where on the page>", "confidence": 0.8, "no_evidence": false, "evidence_type": "text", "region": null, "claims": [{"text": "<one atomic fact>", "answer_span": "<exact supporting words or null>"}]}
Atomic means: one fact per claim, no pronouns without referents.
The page may be a slide with OCR errors, spacing quirks (e.g. "Col 3" vs "Col3"),
or ALL-CAPS headings. Answer the INTENDED meaning and match the closest visible
text; do not refuse just because of minor OCR noise. Only set "no_evidence": true when
the page genuinely lacks the information."""


def build_prompt(question: str, crop_text: str | None = None, max_claims: int = 5) -> str:
    parts = [
        READER_SYSTEM,
        f"Question: {question}",
        f"Give at most {max_claims} claims.",
    ]
    if crop_text:
        parts.append(
            "Text read from the evidence crop (may contain OCR noise, verify against the images):\n"
            + crop_text[:2000]
        )
    parts.append("JSON:")
    return "\n\n".join(parts)


def build_sentence_prompt(
    question: str,
    sentences: list,  # list[SentenceSpan] (duck-typed: .sent_id + .text)
    crop_text: str | None = None,
    max_claims: int = 5,
) -> str:
    """Prompt variant that asks the reader to cite numbered SENTENCE ids.

    `sentences` are SentenceSpan units in page reading order. The reader must
    return `quote` + `sentence_ids`: the ids of the sentences whose text
    contains the supporting quote. Ids let us draw evidence boxes exactly on
    the cited sentences (no fuzzy matching) and expose made-up evidence.
    """
    base = build_prompt(question, crop_text, max_claims)
    if not sentences:
        return base
    # cap prompt size: keep head+tail when the page is long
    shown = sentences if len(sentences) <= 200 else (sentences[:150] + sentences[-50:])
    vocab = "\n".join(f"[S{s.sent_id}] {s.text}" for s in shown)
    extra = (
        "Numbered sentences from the page (cite these as evidence):\n" + vocab[:8000]
        + '\nCite evidence as "quote" (verbatim supporting words) and '
          '"sentence_ids" (the [S#] ids whose text contains the quote, usually one). '
          'Example: {"answer": "...", "quote": "...", "sentence_ids": [12]}. '
          "Ids must be sentences that appear above and must actually contain your quote."
    )
    return base.replace("\n\nJSON:", f"\n\n{extra}\n\nJSON:")


def build_multipage_sentence_prompt(
    question: str,
    page_blocks: list,  # list[dict]: {"page": int, "sentences": [SentenceSpan], "is_primary": bool}
    crop_text: str | None = None,
    max_claims: int = 5,
    max_chars: int = 12000,
) -> str:
    """Phase 2: one prompt carrying several pages (primary has the images).

    Sentences are namespaced `[p{page}.s{sent_id}]` so citations are
    unambiguous across pages; the reader may cite any page block. Secondary
    pages are text-only (OCR sentences), the primary page also has the page
    image + evidence crop attached.
    """
    base = build_prompt(question, crop_text, max_claims)
    if not page_blocks:
        return base
    blocks: list[str] = []
    for b in page_blocks:
        lab = int(b.get("label", b["page"]))
        page = int(b["page"])
        doc = str(b.get("doc_id", "") or "")
        tag = ("PRIMARY page -- image + evidence crop attached"
               if b.get("is_primary") else "secondary page -- text only")
        sents = list(b.get("sentences") or [])
        shown = sents if len(sents) <= 200 else (sents[:150] + sents[-50:])
        vocab = "\n".join(f"[p{lab}.s{s.sent_id}] {s.text}" for s in shown)
        hdr = f"PAGE p{lab} (pdf page {page}" + (f", document {doc}" if doc else "") + f"; {tag}):"
        blocks.append(f"{hdr}\n{vocab if vocab else '(no readable sentences)'}")
    body = "\n\n".join(blocks)
    if len(body) > max_chars:
        body = body[:max_chars - 3] + "..."
    extra = (
        "Numbered sentences from the candidate pages (cite these as evidence). "
        "Each citation id is namespaced as `p<label>.s<id>`, e.g. [p1.s12] must "
        'be cited as "p1.s12". Any page above may hold the answer; the PRIMARY '
        "page also has its image attached.\n" + body
        + '\nCite evidence as "quote" (verbatim supporting words) and '
          '"sentence_ids" (e.g. ["p1.s12"]; plain [12] also means the primary page). '
          'Example: {"answer": "...", "quote": "...", "sentence_ids": ["p1.s12"]}. '
          "Ids must appear above and must actually contain your quote. If the "
          "answer needs facts from several pages, cite the sentence(s) on each."
    )
    return base.replace("\n\nJSON:", f"\n\n{extra}\n\nJSON:")


# Alias kept for callers/tests that used the old word-id prompt name; it now
# builds the sentence prompt when given SentenceSpan objects.
def build_grounded_prompt(question, crop_text=None, numbered_words=None, max_claims=5):
    """Deprecated shim -> sentence prompt. `numbered_words` is ignored unless
    items expose `.sent_id`/`.text` (then treated as sentences)."""
    sents = [w for w in (numbered_words or []) if hasattr(w, "sent_id")]
    return build_sentence_prompt(question, sents, crop_text, max_claims)


VISUAL_CHECK_SYSTEM = """You are an independent verifier looking at provided image(s) of one document
(a zoomed region and/or the surrounding page area). A reader was asked a question
and replied with a claimed answer. Decide honestly whether the image shows that
answer. The claimed answer may be a number or value visible in a chart, table or
figure even if the surrounding labels/units are outside the crop.
Return STRICT JSON only: {"answer": "<the answer YOU read from the image(s)>", "supports": true|false}
Set "supports" to true only if the image states or directly shows the claimed
answer (values visible in charts/tables/figures count). If not shown, return
{"answer": "UNANSWERABLE", "supports": false}. No outside knowledge."""


def build_visual_check_prompt(question: str, claimed_answer: str) -> str:
    """Cross-family confirmation prompt for the visual middle tier (Run 8)."""
    return ("\n\n".join([
        VISUAL_CHECK_SYSTEM,
        f"Question: {question}",
        f"Claimed answer: {claimed_answer}",
        "JSON:",
    ]))


EXAMINER_SYSTEM = """You are a strict, independent examiner. You are shown the evidence image(s)
for one document page (a zoomed region and/or the page itself). Another model was asked the
question below and produced a claim. Your job is to find problems with the claim, not to be polite.

Rules:
- Work only from the image(s) shown. NEVER use outside knowledge.
- Do not assume the other model is right. First look for the answer YOURSELF in the image.
- "quote_seen": copy the exact words visible in the image that carry the answer (verbatim).
  If the answer value is a label/number in a chart or table, copy that label/number.
  If you cannot find the claimed answer at all in the image, set "quote_found": false and
  "quote_seen": "".
- "quote_found": true only when the answer is really visible in the image.
- "answer_supported": "yes" (visible and correct), "partly" (partially visible or ambiguous),
  "no" (not visible or wrong for this page).
- "answers_the_question": true only if the claim actually answers the question as asked
  (right kind of thing, right entity, right year/unit/scope when the question specifies one).
- "problem": one short sentence naming the biggest problem, or "" when there is none.
Return STRICT JSON only:
{"quote_found": true, "quote_seen": "<verbatim words from the image>", "answer_supported": "yes", "answers_the_question": true, "problem": ""}"""


def build_examiner_prompt(
    question: str,
    claimed_answer: str,
    claimed_quote: str = "",
    found_in: str = "",
) -> str:
    """Run 9 (guideline s4.3): the examiner is SHOWN the claim (not just asked
    to answer), then asked to find the quote itself and judge fit."""
    parts = [
        EXAMINER_SYSTEM,
        f"Question: {question}",
        f"Claimed answer: {claimed_answer}",
    ]
    if claimed_quote:
        parts.append(f"The other model's quote: {claimed_quote!r}")
    if found_in:
        parts.append(f"The other model says it is located at: {found_in}")
    parts.append("JSON:")
    return "\n\n".join(parts)


def build_passage_retry_prompt(
    original_prompt: str,
    passages: list,  # list of (sent_id, text, score) or (text, score) tuples
    question: str,
) -> str:
    """Corrective prompt after an UNANSWERABLE verdict (suggestion #7).

    Shows the closest passages on the page verbatim and asks the reader to
    re-examine the image / sentences, tolerating OCR noise, before abstaining.
    """
    lines: list[str] = []
    for p in passages[:5]:
        if isinstance(p, dict):
            sid, text = p.get("sent_id"), p.get("text", "")
        elif len(p) == 3:
            sid, text = p[0], p[1]
        else:
            sid, text = None, p[0]
        tag = f"[S{sid}] " if sid is not None else ""
        if text:
            lines.append(f"{tag}{text}")
    block = "\n".join(lines)[:3000]
    correction = (
        f"You answered UNANSWERABLE, but these passages from the page look related to "
        f"the question {question!r}:\n{block}\n\n"
        "Re-examine the page image and the numbered sentences. If the INTENDED answer "
        "is visible (ignoring minor OCR noise such as extra/missing spaces or casing), "
        'return it with a verbatim "quote" and the "sentence_ids" containing it. '
        'Only keep "answer": "UNANSWERABLE" if the page truly lacks the information.'
    )
    return original_prompt + "\n\n" + correction
