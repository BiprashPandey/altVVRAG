"""Cropping + overlay visualization (Section 7.2 steps 6-7)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vvrag.logging_utils import get_logger

log = get_logger(__name__)


def union_box(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    if not boxes:
        raise ValueError("no boxes to union")
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def crop_union(
    image_path: str | Path, boxes: list[tuple[int, int, int, int]], out_path: str | Path
) -> str:
    """Crop the union of evidence boxes from the page image. Returns crop path."""
    from PIL import Image

    image_path, out_path = Path(image_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as im:
        w, h = im.size
        x0, y0, x1, y1 = union_box(boxes)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"degenerate crop box {(x0, y0, x1, y1)} for {w}x{h}")
        im.crop((x0, y0, x1, y1)).save(out_path)
    return str(out_path)


def overlay_heatmap(
    image_path: str | Path,
    grid_norm: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    out_path: str | Path,
    alpha: float = 0.45,
) -> str:
    """Render heat overlay (red, alpha ~ heat) + green evidence boxes. For inspection."""
    from PIL import Image

    image_path, out_path = Path(image_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path).convert("RGB") as im:
        w, h = im.size
        heat_img = Image.fromarray((np.asarray(grid_norm) * 255).astype(np.uint8), mode="L")
        heat_img = heat_img.resize((w, h), Image.BILINEAR)
        red = Image.new("RGB", (w, h), (255, 0, 0))
        im = Image.blend(im, Image.composite(red, im, heat_img.point(lambda v: int(v * alpha))), 1.0)
        from PIL import ImageDraw

        d = ImageDraw.Draw(im)
        for x0, y0, x1, y1 in boxes:
            d.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)
        im.save(out_path)
    return str(out_path)
