# TODO — Run 9 live validation ("answer, verify, show the source")

Updated: 2026-10-21 10:30

Status: **DONE for model A + hybrid; model B partial (quota)**.

## Final state

- Run of record: `results/demo/testrun8_20261019_141118/rows.json`
  - 186 rows = keyword 162 (model A 81/81, model B 28/81) + hybrid 24/24.
  - 133 completed / 53 resumable errors (all `keyword|gemini-3.8-flash` quota).
  - Keyword-only backup: `rows_keyword_run9.json`.
- Report: `testRun9.pdf` (2.2 MB, built from the merged 186-row dir).
- Gate: previous 18-col artifact retained (`data/gate/tau.json` tau=0.95,
  n_calib=64, accepted 21 / 0 errors). A runs-5-9 refit was attempted
  (233 labels); at target 5% no threshold accepted any row (tau=1.0,
  0 accepted) because several wrong rows scored at the top. Artifacts:
  `results/gate_refit_run9/`. The gate is not the Run 9 abstention mechanism.
- Suite: **173 collected, 172 passed, 1 skipped, 0 failed**
  (`python -m pytest -q`, EXIT=0).

## Results (completed rows)

| cell | rows | shown | strict | false abst | ph@1 | MRR | answered | verified n | verified prec | examinable | probes flagged | false verified |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| keyword / gemini-3.5-flash-lite | 81/81 | 0.650 | 0.050 | 0.200 | 0.700 | 0.858 | 32 | 3 | 0.667 | 21/40 | 1.000 (41/41) | 0.000 |
| keyword / gemini-3.8-flash (partial) | 28/81 | 0.393 | 0.071 | 0.571 | 0.643 | 0.873 | 12 | 3 | 0.667 | 11/28 | n/a | n/a |
| hybrid / gemini-3.5-flash-lite | 24/24 | 0.583 | 0.083 | 0.250 | 0.667 | 0.774 | 18 | 3 | 0.667 | 17/24 | n/a | n/a |
| aggregate | 133/186 | — | — | — | — | — | — | 9 | 0.667 (CP95 [0.299, 0.925]) | 49/92 | 1.000 (41/41) | 0.000 |

- States: 9 verified / 65 not_confident / 59 nothing_found (completed rows).
- Probe categories (keyword A): absent-attribute 11/11, entity-swap 13/13,
  false-bridge 9/9, near-miss 8/8 flagged; **false_verified_on_probes = 0**.
- Hybrid on the same 24 items: page_hit@1 **16/24 vs 13/24** keyword,
  shown **14 vs 11**, strict 2 vs 2, MRR (found-only) 0.774 vs 0.801.
- Category (keyword A, shown/strict): single-hop 0.933/0.067 (n=15),
  multi-hop 0.400/0.200 (n=5), image-only 0.500/0.000 (n=20).

## Findings (keep these)

1. **Verified-but-wrong = 3 rows, all the same question**: the Alcatel slogan
   item (`..._95-17`) reached VERIFIED in keyword A, keyword B and hybrid A
   with answer "ALCATEL-LUCENT OVERVIEW 2012" vs gold "AT THE SPEED OF IDEAS".
   The examiner confirmed the on-page slide title as answering the question.
   This is examiner leniency (report section 4c), not a quote failure.
2. **Examiner availability 49/92 gold rows**; 43 were not_confident because
   the examiner could not run (Groq TPD, plus 404 fallback models), not
   because verification failed. Examiner-only denominators exclude them;
   the report prints the unavailable count per cell.
3. **Quote-seen false negatives**: e.g. "12.5 bn" vs the examiner's fuller
   "Trading Operating Profit 12.5 bn" gives match 0.57 < 0.60 and
   not_confident on a correct answer. Conservative (answer shown, not
   verified), not a false verified answer.
4. **Table/cell citation units** remain heuristic; a quoted answer spanning a
   cell boundary can still fail quote/support checks.
5. **Model B quota** was still exhausted on 2026-10-21; each blocked row burns
   ~5 min of retries. The 53 remaining rows are resumable, not failures.

## Done this session

- Harness hardening:
  - `src/vvrag/eval/harness.py` (`validate_index_arg`, `select_resume_rows`).
  - `scripts/run_test8.py`: `--index` directory guard, resume-model filter,
    `--summarize DIR`, examiner-only Run 9 stats.
  - `src/vvrag/eval/metrics.py`: `verified_stats(..., examiner_only=...)`
    with `n_gold_examinable`, `n_examiner_unavailable`,
    `verified_precision_cp95`; coverage/source accuracy now use all gold rows
    (not only examiner-ran rows).
  - `tests/test_harness.py`; examiner-only metric test; full suite green.
- Live run:
  - Model A resumed to 81/81 with the corrected guard/resume path.
  - Hybrid arm completed 24/24 with `mode_used=hybrid` and cached query
    vectors; merged into the run of record (hybrid `idx` renumbered).
- Report (`scripts/make_test8_pdf.py`):
  - Run 9 detection/title/default `testRun9.pdf`; run-status/incomplete-cell
    warning; Run 9 feature list; section 4b examiner-only metrics with
    denominators; section 4c verified-but-wrong audit; section 4d
    PASS/FAIL verdict; gate refit note; hybrid section; honest notes.
- Gate refit attempt (runs 5-9, 233 labels) and retention of the previous
  18-col gate; refit artifacts kept for audit.
- Full pytest after the run: 173 collected, 172 passed, 1 skipped.

## Next (optional)

1. Resume model B when quota resets:
   `python scripts/run_test8.py --resume results/demo/testrun8_20261019_141118 --resume-models gemini-3.8-flash`
   then rebuild the report and `--summarize`.
2. Improve examiner robustness: remove the 404 fallback models from
   `verify.examiner_models` and aggregate per-model errors instead of keeping
   only the last one; refit tau when enough live labels exist.
3. Add a question-constraint check for semantic near-misses (the Alcatel
   slogan class); the current examiner is too lenient on on-page but
   non-answering text.
4. Optional: multi-page reader context for the multi-hop Nestlé items.

## Commands

```bash
# rebuild the report from the merged run
python scripts/make_test8_pdf.py results/demo/testrun8_20261019_141118 testRun9.pdf

# regenerate summary.json
python scripts/run_test8.py --summarize results/demo/testrun8_20261019_141118

# resume model B next quota day
python scripts/run_test8.py --resume results/demo/testrun8_20261019_141118 --resume-models gemini-3.8-flash

# full suite (no live run active)
python -m pytest -q
```

## Honesty notes

- `verified` requires an examiner that actually ran; examiner-unavailable rows
  are `not_confident` and must not be counted as verification failures.
- The 53 model-B rows are `PENDING`/quota errors; the cell is incomplete and
  must not be read as 0 performance.
- Gate artifacts are 18-col feature-order version-bound; do not mix with
  9/12/15-col artifacts. The refit at 5% was infeasible; the retained gate is
  the pre-Run-9 18-col artifact.
- Small n: 9 verified rows and 49 examiner-ran gold rows give wide CP
  intervals; treat the rates as directional.
