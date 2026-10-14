"""Reader prompts (Section 7.3): strict-JSON answer + atomic claims."""
from __future__ import annotations


READER_SYSTEM = """You are a precise document question-answering assistant.
You are shown a document page image and a cropped evidence region that is likely to contain the answer.
Answer ONLY from what is visible in the images. Do not use outside knowledge.
If the page does not contain the answer, set "answer" to "UNANSWERABLE" and "claims" to [].
Return STRICT JSON only, no code fences, no commentary, matching this schema:
{"answer": "<short final answer>", "claims": [{"text": "<one atomic fact>", "answer_span": "<exact supporting words or null>"}]}
Atomic means: one fact per claim, no pronouns without referents."""


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
