#!/usr/bin/env bash
# Download eval data (Section 10.1). Primary: SlideVQA; fallback: DocVQA subset.
set -euo pipefail
OUT="${1:-data/raw_eval}"
mkdir -p "$OUT"
echo "== Trying SlideVQA mirrors on the Hugging Face Hub =="
python - "$OUT" <<'EOF'
import sys
out = sys.argv[1]
try:
    from datasets import load_dataset
except ImportError:
    print("pip install datasets pyarrow first"); sys.exit(2)
for cand in ("lmms-lab/SlideVQA", "SlideVQA/SlideVQA"):
    try:
        ds = load_dataset(cand, split="test")
        ds.to_json(f"{out}/slidevqa_test.jsonl")
        print(f"OK: {cand} ({len(ds)} rows) -> {out}/slidevqa_test.jsonl")
        break
    except Exception as e:
        print(f"miss {cand}: {type(e).__name__}")
else:
    print("SlideVQA unavailable; documented fallback: use --dataset synthetic, "
          "or place DocVQA pages under data/raw and re-run ingest.")
    sys.exit(1)
EOF
