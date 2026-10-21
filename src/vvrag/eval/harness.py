"""Small, testable helpers shared by the live run harnesses.

The live matrix scripts are command-line tools, but the argument checks and
resume-selection rules are pure functions; keeping them here makes them
unit-testable without importing the heavy reader/NLI stack.
"""
from __future__ import annotations

from pathlib import Path


def validate_index_arg(path: str | Path) -> Path:
    """Return the index directory, rejecting the SQLite file path.

    The QA pipeline joins ``meta.db`` onto the supplied directory; passing
    ``data/index/meta.db`` used to produce ``data/index/meta.db/meta.db`` and a
    ``FileExistsError`` deep inside ingestion. Fail fast at the CLI instead.
    """
    p = Path(path)
    if p.name.lower() == "meta.db":
        raise SystemExit(
            "--index expects the index DIRECTORY (e.g. data/index), "
            f"not the SQLite file: {path}")
    return p


def select_resume_rows(rows: list[dict],
                       only_models: list[str] | None = None) -> list[int]:
    """Indices of rows to re-run: errored rows, optionally model-filtered.

    A successful row has no ``error`` key. ``PENDING`` rows and quota/transient
    failures all carry one, so they are resumable. ``only_models`` restricts
    the selection to the pinned models named on the command line.
    """
    out: list[int] = []
    for k, row in enumerate(rows):
        if "error" not in row:
            continue
        if only_models and row.get("model") not in only_models:
            continue
        out.append(k)
    return out
