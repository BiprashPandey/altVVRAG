"""Hollow smoke test for notebooks/frozen3_colab.ipynb.

Executes the notebook's lightweight cells against the local frozen inputs and
the local fake-embedding index. Downloads, GPU encoding, the real index build,
and the bundle download are skipped and reported as SKIP.

Usage:
    python scripts/smoke_frozen3_colab.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "frozen3_colab.ipynb"
LOCAL_INDEX = REPO / "data" / "index_frozen3_local"
SETD = REPO / "data" / "slidevqa_frozen3"

sys.path.insert(0, str(REPO / "src"))

RESULTS: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    RESULTS.append((status, name, detail))
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")


def strip_magics(source: str) -> str:
    return "\n".join(
        "" if line.lstrip().startswith(("%", "!")) else line
        for line in source.splitlines()
    )


def run_cell(namespace: dict, source: str) -> None:
    code = strip_magics(source)
    exec(compile(code, "<notebook-cell>", "exec"), namespace)


def find_cell(cells: list[str], *needles: str) -> str:
    for source in cells:
        if all(n in source for n in needles):
            return source
    raise KeyError(f"no cell matching {needles}")


def run_help(cmd: list[str], flags: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(
        cmd, cwd=REPO, capture_output=True, text=True, timeout=120
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    missing = [f for f in flags if f not in text]
    if proc.returncode != 0:
        return False, f"exit {proc.returncode}: {text.strip()[-300:]}"
    if missing:
        return False, f"missing flags: {', '.join(missing)}"
    return True, "all flags present"


def main() -> int:
    print("FROZEN3 COLAB HOLLOW SMOKE")
    print(f"repo: {REPO}")
    print(f"notebook: {NOTEBOOK}")

    try:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    except Exception as exc:
        record("FAIL", "notebook JSON", f"{type(exc).__name__}: {exc}")
        return 1

    cells = ["".join(c["source"]) for c in notebook["cells"]
             if c["cell_type"] == "code"]
    record("PASS", "notebook JSON", f"{len(notebook['cells'])} cells, "
                                    f"{len(cells)} code cells")

    syntax_bad = []
    for i, source in enumerate(cells):
        try:
            compile(strip_magics(source), f"cell-{i}", "exec")
        except SyntaxError as exc:
            syntax_bad.append((i, exc))
    if syntax_bad:
        record("FAIL", "code-cell syntax", str(syntax_bad[:3]))
    else:
        record("PASS", "code-cell syntax", "all code cells compile")

    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        record(
            "PASS" if cuda else "SKIP",
            "GPU check",
            f"torch {torch.__version__}, cuda={cuda}",
        )
    except Exception as exc:
        record("SKIP", "GPU check", f"torch unavailable: {exc}")

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--dry-run",
             "--no-deps", "-e", "."],
            cwd=REPO, capture_output=True, text=True, timeout=180,
        )
        text = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0 and "vvrag" in text.lower()
        record("PASS" if ok else "FAIL", "editable install (dry run)",
               "would install vvrag" if ok else text.strip()[-300:])
    except Exception as exc:
        record("FAIL", "editable install (dry run)",
               f"{type(exc).__name__}: {exc}")

    try:
        import vvrag

        record("PASS", "vvrag import", vvrag.__file__ or "")
    except Exception as exc:
        record("FAIL", "vvrag import", f"{type(exc).__name__}: {exc}")

    namespace: dict = {"__name__": "__main__"}
    try:
        cell = find_cell(cells, "frozen_meta.json", "gold_sha256")
        cell = cell.replace("Path('/content/altVRAG')", f"Path({str(REPO)!r})")
        run_cell(namespace, cell)
        record("PASS", "frozen JSON hashes/counts",
               f"{len(namespace['decks'])} decks, "
               f"{len(namespace['gold'])} gold, "
               f"{len(namespace['probes'])} probes, "
               f"{len(namespace['questions'])} questions")
    except Exception as exc:
        record("FAIL", "frozen JSON hashes/counts",
               f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    try:
        ok, detail = run_help(
            [sys.executable, "scripts/prepare_slidevqa.py", "--help"],
            ["--decks-file", "--gold-out", "--max-rows", "--n-decks",
             "--n-qa", "--out", "--split"],
        )
        record("PASS" if ok else "FAIL", "deck materialization CLI", detail)
    except Exception as exc:
        record("FAIL", "deck materialization CLI",
               f"{type(exc).__name__}: {exc}")

    try:
        cell = find_cell(cells, "page_count", "materialization OK")
        run_cell(namespace, cell)
        record("PASS", "local PDF count/pages",
               f"{len(namespace['pdfs'])} PDFs, {namespace['pages']} pages")
    except Exception as exc:
        record("FAIL", "local PDF count/pages",
               f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    try:
        ok, detail = run_help(
            [sys.executable, "-m", "vvrag.ingest.run_ingest", "--help"],
            ["--input", "--out", "--embedder", "--model-id", "--device",
             "--dtype", "--dpi", "--upscale", "--qdrant-path",
             "--batch-size"],
        )
        record("PASS" if ok else "FAIL", "ingest CLI", detail)
    except Exception as exc:
        record("FAIL", "ingest CLI", f"{type(exc).__name__}: {exc}")

    try:
        namespace["index"] = LOCAL_INDEX
        cell = find_cell(cells, "select count(*) from pages",
                         "index OK")
        run_cell(namespace, cell)
        record("PASS", "index shape",
               f"{namespace['pages_n']} pages, "
               f"{namespace['words_n']} words, "
               f"{namespace['sent_n']} sentences")
    except Exception as exc:
        record("FAIL", "index shape", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    try:
        ok, detail = run_help(
            [sys.executable, "scripts/precompute_query_vectors.py", "--help"],
            ["--gold", "--out", "--model", "--device", "--check"],
        )
        record("PASS" if ok else "FAIL", "query-cache CLI", detail)
    except Exception as exc:
        record("FAIL", "query-cache CLI", f"{type(exc).__name__}: {exc}")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            cache.mkdir()
            proc = subprocess.run(
                [sys.executable, "scripts/precompute_query_vectors.py",
                 "--gold", "data/slidevqa_frozen3/questions_all.json",
                 "--out", str(cache), "--check"],
                cwd=REPO, capture_output=True, text=True, timeout=120,
            )
            text = (proc.stdout or "") + (proc.stderr or "")
            expected = "0/180" in text
            record("SKIP" if expected else "FAIL", "query-cache coverage",
                   "empty hollow cache reports 0/180; real run must assert "
                   "180/180" if expected else text.strip()[-300:])
    except Exception as exc:
        record("FAIL", "query-cache coverage",
               f"{type(exc).__name__}: {exc}")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "retrieval"
            cache = Path(tmp) / "cache"
            cache.mkdir()
            proc = subprocess.run(
                [sys.executable, "scripts/validate_retrieval.py",
                 "--index", str(LOCAL_INDEX),
                 "--gold", str(SETD / "gold.json"),
                 "--query-cache", str(cache),
                 "--out", str(out)],
                cwd=REPO, capture_output=True, text=True, timeout=300,
            )
            text = (proc.stdout or "") + (proc.stderr or "")
            ok = proc.returncode == 0 and "keyword:" in text
            tail = [line for line in text.splitlines() if line.strip()][-3:]
            record("PASS" if ok else "FAIL", "retrieval validation",
                   " | ".join(tail) if ok else text.strip()[-400:])
    except Exception as exc:
        record("FAIL", "retrieval validation",
               f"{type(exc).__name__}: {exc}")

    try:
        cell = find_cell(cells, "check_absent", "probe absence OK")
        run_cell(namespace, cell)
        record("PASS", "probe absence", "60 probes, 0 present checks")
    except Exception as exc:
        record("FAIL", "probe absence", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    try:
        import torch

        torch.cuda.get_device_name = lambda *a, **k: "CPU (hollow)"
        namespace["torch"] = torch
        cell = find_cell(cells, "colab_frozen3_provenance.json")
        with tempfile.TemporaryDirectory() as tmp:
            old = Path.cwd()
            os.chdir(tmp)
            try:
                run_cell(namespace, cell)
                prov_path = Path(tmp) / "results" / "colab_frozen3_provenance.json"
                prov = json.loads(prov_path.read_text(encoding="utf-8"))
                ok = (prov.get("n_gold") == 120
                      and prov.get("n_probes") == 60
                      and prov.get("n_questions") == 180)
                record("PASS" if ok else "FAIL", "provenance assembly",
                       f"gpu={prov.get('gpu')}, "
                       f"query_vectors={prov.get('n_query_vectors')}")
            finally:
                os.chdir(old)
    except Exception as exc:
        record("FAIL", "provenance assembly", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "zipfile", "--help"],
            capture_output=True, text=True, timeout=60,
        )
        record("PASS" if proc.returncode == 0 else "FAIL",
               "bundle ZIP support", "python -m zipfile available")
    except Exception as exc:
        record("FAIL", "bundle ZIP support", f"{type(exc).__name__}: {exc}")
    record("SKIP", "bundle download", "not exercised in hollow mode")

    failed = [r for r in RESULTS if r[0] == "FAIL"]
    passed = [r for r in RESULTS if r[0] == "PASS"]
    skipped = [r for r in RESULTS if r[0] == "SKIP"]
    print("\nHOLLOW SUMMARY: "
          f"{len(passed)} passed, {len(failed)} failed, "
          f"{len(skipped)} skipped")
    if failed:
        for status, name, detail in failed:
            print(f"  FAILED: {name} - {detail}")
        return 1
    print("HOLLOW PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
