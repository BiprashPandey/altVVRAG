"""Validate provider keys without printing secrets.

Checks the environment variables this project reads, reports only a safe
fingerprint (variable name, length, prefix), and lists models visible to each
working key. Exits 0 when at least one usable key exists for every requested
provider; exits 2 otherwise.

Usage:
  python scripts/check_keys.py                 # groq + gemini
  python scripts/check_keys.py --provider groq
  python scripts/check_keys.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vvrag.config import _load_dotenv, load_config  # noqa: E402


def _fingerprint(value: str) -> dict:
    v = value or ""
    if not v:
        return {"present": False, "len": 0, "prefix": ""}
    return {"present": True, "len": len(v),
            "prefix": (v[:4] + "…") if len(v) > 4 else "…"}


def check_groq(key: str) -> dict:
    import requests

    out = {"provider": "groq", "models": [], "error": ""}
    if not key:
        out["error"] = "missing"
        return out
    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {key}"}, timeout=30)
        if r.status_code != 200:
            out["error"] = f"HTTP {r.status_code}: {r.text[:160]}"
            return out
        out["models"] = sorted(m.get("id", "") for m in
                               (r.json().get("data") or []) if m.get("id"))
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    return out


def check_gemini(key: str) -> dict:
    out = {"provider": "gemini", "models": [], "error": ""}
    if not key:
        out["error"] = "missing"
        return out
    try:
        from google import genai

        client = genai.Client(api_key=key)
        out["models"] = sorted(
            (m.name or "").replace("models/", "")
            for m in client.models.list())
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["groq", "gemini", "all"],
                    default="all")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    _load_dotenv(".env")
    cfg = load_config()

    report = {"groq": [], "gemini": [], "config": {
        "groq_examiner_models": list(cfg.verify.examiner_models or []),
        "gemini_reader_model": cfg.reader.model,
    }}
    ok = {"groq": False, "gemini": False}
    if args.provider in ("groq", "all"):
        for name in ("GROQ_API_KEY", "GROQ_API_KEY2", "GROQ_API_KEY3"):
            val = os.environ.get(name, "")
            fp = _fingerprint(val)
            fp["name"] = name
            rec = check_groq(val) if val.startswith("gsk_") else {
                "provider": "groq", "models": [],
                "error": ("not a gsk_ key; skipped (not a Groq key)"
                          if val else "missing")}
            fp.update(rec)
            report["groq"].append(fp)
            if not rec["error"]:
                ok["groq"] = True
    if args.provider in ("gemini", "all"):
        for name in ("GEMINI_API_KEY", "GEMINI_API_KEY2", "GEMINI_API_KEY3",
                     "GOOGLE_API_KEY"):
            val = os.environ.get(name, "")
            fp = _fingerprint(val)
            fp["name"] = name
            rec = check_gemini(val) if val else {
                "provider": "gemini", "models": [], "error": "missing"}
            fp.update(rec)
            report["gemini"].append(fp)
            if not rec["error"]:
                ok["gemini"] = True

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for provider in ("groq", "gemini"):
            if not report[provider]:
                continue
            print(f"\n{provider.upper()}")
            for r in report[provider]:
                state = "OK" if not r["error"] else f"FAIL ({r['error']})"
                print(f"  {r['name']}: len={r['len']} prefix={r['prefix']} {state}")
                if r["models"]:
                    models = ", ".join(r["models"][:16])
                    more = "" if len(r["models"]) <= 16 else (
                        f" … (+{len(r['models']) - 16})")
                    print(f"    models: {models}{more}")
            wanted = report["config"].get(
                "groq_examiner_models" if provider == "groq"
                else "gemini_reader_model")
            if provider == "groq" and wanted:
                available = {m for r in report[provider] for m in r["models"]}
                missing = [m for m in wanted if m not in available]
                if missing:
                    print(f"  configured examiner models not visible: {missing}")
        print("\nusable:", ", ".join(p for p, v in ok.items() if v) or "none")
    return 0 if all(v for p, v in ok.items()
                    if args.provider in ("all", p)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
