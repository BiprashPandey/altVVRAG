"""Precompute ColQwen query multivectors for the disk cache (Phase 1).

Run on a GPU box (Colab) or any machine that can hold the model:

    python scripts/precompute_query_vectors.py                    # gold.json default
    python scripts/precompute_query_vectors.py --questions-file qs.txt
    python scripts/precompute_query_vectors.py --extra "Q1?" "Q2?"
    python scripts/precompute_query_vectors.py --check            # coverage only (laptop-safe)

Output: data/query_cache/<sha1(question)>.json (QueryVectorStore format),
read by ui.qa.answer_question for visual/hybrid retrieval and by the heatmap
prior. Cache misses degrade to keyword-only at runtime (recorded per answer).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vvrag.retrieve.query_vectors import QueryVectorStore  # noqa: E402


def collect_questions(gold_path: str, questions_file: str, extra: list[str]) -> list[str]:
    qs: list[str] = []
    if gold_path:
        p = Path(gold_path)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data = data.get("items", [])
            for it in data:
                q = str(it.get("question", "")).strip()
                if q:
                    qs.append(q)
    if questions_file:
        for line in Path(questions_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                qs.append(line)
    for q in extra:
        if q.strip():
            qs.append(q.strip())
    seen, out = set(), []
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", default="data/gold.json",
                    help="gold items JSON with a 'question' per item")
    ap.add_argument("--questions-file", default="", help="one question per line")
    ap.add_argument("--extra", nargs="*", default=[], help="ad-hoc question strings")
    ap.add_argument("--out", default="data/query_cache")
    ap.add_argument("--model", default="vidore/colqwen2-v1.0")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--force", action="store_true", help="re-encode cached questions")
    ap.add_argument("--check", action="store_true", help="report coverage only")
    args = ap.parse_args()

    store = QueryVectorStore(args.out)
    questions = collect_questions(args.gold, args.questions_file, args.extra)
    if not questions:
        print("no questions found (use --gold/--questions-file/--extra)")
        return
    cached, total = store.coverage(questions)
    todo = [q for q in questions if args.force or not store.has(q)]
    print(f"query cache {Path(args.out)}: {cached}/{total} present; "
          f"{len(todo)} to encode")
    if args.check or not todo:
        if not args.check:
            print("nothing to encode")
        return

    from vvrag.retrieve.query_embed import create_query_embedder

    enc = create_query_embedder("colqwen", model_id=args.model, device=args.device)
    print(f"loading {args.model} (first run downloads weights) ...", flush=True)
    t0 = time.time()
    enc.load()
    print(f"model ready on {enc.device} in {time.time() - t0:.0f}s", flush=True)

    ok = 0
    for i, q in enumerate(todo, 1):
        try:
            vecs, tokens = enc.embed_query(q)
        except Exception as e:  # noqa: BLE001 - keep going, report per question
            print(f"[{i}/{len(todo)}] FAIL {q[:70]!r}: {e}", flush=True)
            continue
        if int(vecs.shape[1]) != args.dim:
            print(f"[{i}/{len(todo)}] WARNING dim {vecs.shape[1]} != {args.dim} "
                  f"(index expects {args.dim}); storing anyway", flush=True)
        store.put(q, vecs, tokens, model=args.model, device=str(enc.device))
        ok += 1
        print(f"[{i}/{len(todo)}] ok {vecs.shape} {q[:70]!r}", flush=True)
    cached, total = store.coverage(questions)
    print(f"DONE: encoded {ok}/{len(todo)}; cache now {cached}/{total}")


if __name__ == "__main__":
    main()
