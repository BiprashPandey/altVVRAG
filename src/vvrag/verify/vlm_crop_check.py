"""Optional VLM crop check (Section 7.4). OFF by default.

Used only when the crop has no text: asks a DIFFERENT-family VLM whether the
crop image supports the claim (yes/no). Kept separate from the reader to hold
family diversity; API-light by design.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence


def parse_yes_no(raw: str) -> bool | None:
    """Parse a yes/no verdict out of free text.

    Must tolerate JSON-mode readers (GeminiReader forces
    response_mime_type=application/json, so real replies look like
    '{"answer": "yes"}' or '"no"' — bare word-split matching misses both;
    found 2026-10-15 on live crops). First yes/no token in reading order
    wins; anything else is unparseable.
    """
    import re

    toks = re.findall(r"[a-z]+", raw.lower())
    positions = [(i, t) for i, t in enumerate(toks) if t in ("yes", "no")]
    if not positions:
        return None
    return positions[0][1] == "yes"


def vlm_crop_supports(
    reader,
    crop_image: str | Path,
    claim: str,
    temperature: float = 0.0,
) -> bool | None:
    """Returns True/False, or None if the answer is unparseable."""
    prompt = (
        "Look at this cropped document region. Statement: " + claim + "\n"
        'Does this image support the statement? Answer with exactly "yes" or "no".'
    )
    out = reader.generate([crop_image], prompt, temperature, 1)[0]
    return parse_yes_no(out)


def check_claims(
    reader,
    crop_image: str | Path | None,
    claims: Sequence[str],
    enabled: bool = False,
) -> list[bool | None]:
    if not enabled or crop_image is None:
        return [None] * len(claims)
    return [vlm_crop_supports(reader, crop_image, c) for c in claims]
