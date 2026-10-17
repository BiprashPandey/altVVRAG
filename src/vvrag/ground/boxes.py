"""Grid -> pixel boxes (Section 7.2 step 5): threshold + connected components.

Single source of truth for cell->pixel conversion stays in schemas.py
(grid_cell_range_to_pixel_box); this module only finds the cell ranges.
"""
from __future__ import annotations

import numpy as np

from vvrag.schemas import WordBox, grid_cell_range_to_pixel_box


def grid_to_boxes(
    grid_norm: np.ndarray,
    page_width: int,
    page_height: int,
    threshold_rel: float = 0.5,
    min_box_area_frac: float = 0.002,
    max_boxes: int = 3,
    padding_px: int = 8,
) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    """Returns (pixel boxes sorted by summed heat desc, boolean cell mask)."""
    try:
        from scipy.ndimage import label
    except ImportError as e:
        raise RuntimeError("scipy required for connected components") from e

    grid_norm = np.asarray(grid_norm, dtype=np.float32)
    gh, gw = grid_norm.shape
    mask = grid_norm >= float(threshold_rel)
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled, ncomp = label(mask, structure=structure)

    min_cells = max(1, int(min_box_area_frac * gh * gw))
    scored: list[tuple[float, tuple[int, int, int, int]]] = []
    for comp in range(1, ncomp + 1):
        cells = np.argwhere(labeled == comp)
        if len(cells) < min_cells:
            continue
        r0, c0 = cells.min(axis=0)
        r1, c1 = cells.max(axis=0) + 1  # exclusive
        heat = float(grid_norm[labeled == comp].sum())
        scored.append((heat, (int(r0), int(c0), int(r1), int(c1))))
    scored.sort(key=lambda s: s[0], reverse=True)

    boxes = [
        grid_cell_range_to_pixel_box(r0, c0, r1, c1, page_width, page_height, gh, gw,
                                     padding_px)
        for _, (r0, c0, r1, c1) in scored[:max_boxes]
    ]
    return boxes, mask


def snap_box_to_words(
    box: tuple[int, int, int, int], words: list[WordBox]
) -> tuple[list[WordBox], tuple[int, int, int, int]]:
    """Word-level output: words whose centers fall inside the box + their
    tight union box. Same center-in-box rule as text_in_box.words_in_boxes,
    so the emitted region is exactly word-aligned (no heat-blob padding).
    Returns ([], box) when no word center falls inside.
    """
    from vvrag.ground.text_in_box import words_in_boxes

    inside = words_in_boxes(words, [box])
    if not inside:
        return [], box
    union = (
        min(w.box[0] for w in inside),
        min(w.box[1] for w in inside),
        max(w.box[2] for w in inside),
        max(w.box[3] for w in inside),
    )
    return inside, union


def answer_to_line_boxes(words: list[WordBox]) -> list[tuple[int, int, int, int]]:
    """One tight box per visual line (rec 4: never one big multi-line union).

    Merges neighbouring answer words on the same line; a multi-line answer
    yields one box per line so unrelated side text is not covered.
    """
    if not words:
        return []
    from vvrag.ground.text_in_box import group_lines

    boxes: list[tuple[int, int, int, int]] = []
    for line in group_lines(words):
        if line:
            boxes.append((
                min(w.box[0] for w in line),
                min(w.box[1] for w in line),
                max(w.box[2] for w in line),
                max(w.box[3] for w in line),
            ))
    return boxes


def snap_boxes_to_word_lines(
    boxes: list[tuple[int, int, int, int]], words: list[WordBox]
) -> tuple[list[WordBox], list[tuple[int, int, int, int]]]:
    """Snap each candidate box to word centers, then split to per-line boxes.

    Returns (unique_inside_words, per_line_boxes). Empty when nothing hits.
    Heat-blob padding is stripped: output is exactly word-aligned.
    """
    from vvrag.ground.text_in_box import words_in_boxes

    inside = words_in_boxes(words, boxes)
    if not inside:
        return [], []
    # dedupe (a word center can fall in overlapping boxes)
    seen, uniq = set(), []
    for w in inside:
        key = (w.text, w.box)
        if key not in seen:
            seen.add(key)
            uniq.append(w)
    return uniq, answer_to_line_boxes(uniq)


def snap_box_to_lines(
    box: tuple[int, int, int, int], words: list[WordBox]
) -> tuple[int, int, int, int]:
    """OPTIONAL (off by default): snap a heatmap box to nearest word-line bounds.

    Expands the box to include full lines of any word whose center falls
    inside it. Evaluated as a separate ablation arm (Section 10.4).
    """
    x0, y0, x1, y1 = box
    inside = [
        w for w in words
        if x0 <= (w.box[0] + w.box[2]) / 2 <= x1 and y0 <= (w.box[1] + w.box[3]) / 2 <= y1
    ]
    if not inside:
        return box
    lines: dict[tuple[int, int], list[WordBox]] = {}
    for w in inside:
        lines.setdefault((w.block_id, w.line_id), []).append(w)
    lx0 = min(w.box[0] for ws in lines.values() for w in ws)
    ly0 = min(w.box[1] for ws in lines.values() for w in ws)
    lx1 = max(w.box[2] for ws in lines.values() for w in ws)
    ly1 = max(w.box[3] for ws in lines.values() for w in ws)
    return (min(x0, lx0), min(y0, ly0), max(x1, lx1), max(y1, ly1))
