"""VLM transcription of zero-OCR pages (Phase 5, optional, code-only offline).

Slide pages that are pure images (charts, logos, big visuals) yield ~no OCR
words, so answers living there are unreachable. This module asks a vision
model to transcribe ALL visible text on such a page. The transcript carries no
word geometry: each transcript sentence becomes ONE WordBox whose box is the
page rectangle (a documented approximation - boxes are page-level, not tight).

Requires GEMINI_API_KEY (+ GEMINI_API_KEY2..) and google-genai; the model id
comes from config `reader.model` (never hardcoded).
"""
from __future__ import annotations

import re
from pathlib import Path

from vvrag.logging_utils import get_logger
from vvrag.schemas import WordBox

log = get_logger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

TRANSCRIBE_PROMPT = (
    "Transcribe ALL text visible in this slide image, in natural reading "
    "order. Include titles, bullet points, table cells, chart labels and axis "
    "text. Output plain text only - no commentary, no markdown."
)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split((text or "").strip()) if s.strip()]


def transcribe_page(
    image_path: str | Path,
    model: str,
    max_output_tokens: int = 2048,
) -> str:
    """One VLM call -> transcription text. Raises RuntimeError w/o key/SDK."""
    from google import genai  # noqa: F401 - explicit import error to caller
    from PIL import Image

    from vvrag.config import env
    from vvrag.reader.client import _gemini_keys

    keys = _gemini_keys(env)
    if not keys:
        raise RuntimeError("GEMINI_API_KEY missing (needed for --vlm-transcribe-empty)")
    client = genai.Client(api_key=keys[0])
    with Image.open(image_path) as im:
        img = im.convert("RGB")
        resp = client.models.generate_content(
            model=model,
            contents=[img, TRANSCRIBE_PROMPT],
            config={"temperature": 0.0, "max_output_tokens": max_output_tokens},
        )
    text = (resp.text or "").strip()
    log.info("VLM transcript %s: %d chars", image_path, len(text))
    return text


def transcript_word_boxes(
    text: str, doc_id: str, page: int, width: int, height: int,
) -> list[WordBox]:
    """Transcript -> WordBoxes with a page-level box per sentence (no geometry)."""
    box = (0, 0, int(width), int(height))
    return [WordBox(doc_id=doc_id, page=page, text=s, box=box,
                    block_id=0, line_id=i)
            for i, s in enumerate(split_sentences(text))]
