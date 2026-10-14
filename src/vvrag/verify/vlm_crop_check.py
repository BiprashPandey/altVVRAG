"""Optional VLM crop check (Section 7.4). OFF by default.

Used only when the crop has no text: asks a DIFFERENT-family VLM whether the
crop image supports the claim (yes/no). Kept separate from the reader to hold
family diversity; API-light by design.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence


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
    out = reader.generate([crop_image], prompt, temperature, 1)[0].strip().lower()
    if "yes" in out.split() or out.startswith("yes"):
        return True
    if "no" in out.split() or out.startswith("no"):
        return False
    return None


def check_claims(
    reader,
    crop_image: str | Path | None,
    claims: Sequence[str],
    enabled: bool = False,
) -> list[bool | None]:
    if not enabled or crop_image is None:
        return [None] * len(claims)
    return [vlm_crop_supports(reader, crop_image, c) for c in claims]
