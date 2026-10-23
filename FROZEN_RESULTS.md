# Frozen deck-disjoint pilot: results and honest reading

Run date: 2026-10-22. Pre-registration: `PREREGISTRATION.md`. Report:
`frozen_eval.pdf`. Rows: `results/demo/testrun8_20261022_214946/rows.json`.

## What was run

- Frozen corpus: 6 new SlideVQA decks / 120 pages, disjoint from the 8 dev
  decks; `data/index_frozen2` (fake 32-d embeddings, keyword retrieval).
- Gold: 28 items (22 single-hop, 5 multi-hop, 1 image-only).
- Probes: 19 verified-absent near-miss/entity-swap/false-bridge/absent items.
- Reader: Gemini 3.5 Flash Lite, one sample per item, no rotation.
- Examiner: Gemini 3.5 Flash Lite, claim-level, page + one claim crop.
- Independence: **false** in `config.pilot.yaml` (same provider/family). This
  pilot measures the claim/verification pipeline, not independent verification.
- Gate loaded but display-only. No thresholds tuned after the run.
- 47/47 rows completed, 0 errors. The first process was killed by the 50-minute
  shell timeout after 29 rows; the 18 PENDING rows were completed with
  `--resume`. The final rows file is one complete 47-row run.

## Headline numbers

| Metric | Value |
|---|---|
| Gold answered (shown) | 19 / 28 (67.9%) |
| Nothing found | 9 / 28 (32.1%) |
| Strict verified | 6 / 28 (21.4%) |
| Verified precision | 5 / 6 (83.3%) |
| Shown-answer precision | 16 / 19 (84.2%) |
| Page hit @1 | 50.0% |
| MRR (found only) | 0.783 |
| Examiner ran on answered gold | 18 / 19 (94.7%) |
| Examiner ran on all gold | 18 / 28 (64.3%; 9 abstained) |
| Claim labels | 10 supported, 9 unsupported, 1 partial, 1 insufficient (21 claims) |
| Visual-verified | 0 (six region attempts, none passed) |
| Probes nothing-found | 15 / 19 (78.9%) |
| Probes answered (not confident + verified) | 4 / 19 (21.1%) |
| Probes verified | 1 / 19 (5.3%, one false-verified) |

By gold category (shown / verified / strict-correct):

- single-hop: 15 / 5 / 4 of 22; false abstention 31.8%
- multi-hop: 3 / 1 / 1 of 5; false abstention 40.0%
- image-only: 1 / 0 / 0 of 1

## The main failure

`frozen-unans-14` was **verified**:

- Question: "What is Old Act Section 4 in the New Act Section 12?"
- Answer: `11`
- Cited unit: table row `4 | Award | 11 | 23/30/.`
- The row maps Section 4 to Section 11, not Section 12. The answer is on the
  page and the claim/citation are self-consistent, but the answer does not
  satisfy the question's Section 12 constraint.

This is exactly the question-constraint grounding gap: citation-specific
verification plus answer support is not enough when the question contains a
qualifier that the cited evidence contradicts. The same-family examiner
accepted it. It is a real false-verified probe, not a label or parser error.

The three other answered probes were `not_confident`, so they were flagged but
still shown. Probe false-verified rate is therefore 5.3%, not zero.

## Second issue: partial answers are scored wrong

`1-150711120533-lva1-app6892_95-64` was verified with
"National Monitoring Committee", while gold is "National Monitoring Committee,
Oversight at Central Level for all projects". The answer is a correct fragment
of a multi-part gold answer; exact-equivalence scoring marks it wrong. This one
item is the entire gap between 5/6 and 6/6 verified precision. Report it as a
strict-score caveat, not as a fabrication.

## Examiner availability

18 of 19 answered gold rows had a working examiner call (94.7%). One row
(`...-41`) hit the free-tier Gemini quota and is `not_confident` with
`all API keys exhausted (quota)`. This meets the pre-registered 80%-of-answered
availability threshold, but it is not a clean run. The examiner is same-family
in this pilot, so no independent-verification claim is supported.

## Post-hoc fix after the pilot: Section-reference constraint

The false-verified probe exposed a missing question constraint: section
references were not checked against the cited unit. `question_details.py` now
extracts `Section N` / `Sec. N` / `§ N` from the question and requires every
section number to appear in the **cited sentence**, not the whole page.

- Offline check on all 47 frozen rows: exactly one row changes, the false-
  verified probe; none of the 28 gold rows is affected. On the 81-row dev run,
  zero rows are affected.
- Live post-fix check (`results/demo/testrun8_20261022_234533`): the same probe
  is now `not_confident` with reason "the question references section 12, but
  the cited evidence does not contain that section number."
- The independent Groq examiner had also verified the probe before the fix, so
  this is not a same-family artifact. The deterministic check blocks it for any
  examiner provider.
- This is a **post-hoc bug fix**, not a change to the pilot metrics. The pilot
  table above remains the pre-fix result (1/19 false-verified). Use a fresh
  frozen set for the next claim; do not report this same set as improved.

## Independent spot checks (not part of the pilot table)

- `idx=0` (Appendiceal distension) was re-run with the production config:
  Gemini reader + Groq `qwen/qwen3.8-27b` examiner. Result: `verified`, claim
  1/1 supported, citation-specific. This validates the independent claim-level
  path for one item (`results/demo/testrun8_20261022_230621`).
- The false-verified probe re-run with the production Groq examiner could not
  complete: Groq TPD returned 429 (`Used 198670`, `Requested 4877`), so it ran
  with no examiner and became `not_confident` (`..._230706`). It is not evidence
  that the independent examiner would catch the Section 12 mismatch; the
  independent probe check is still open.

## Presentation correction after the run

The old rows labelled six region attempts `tier="visual"` even when the
examiner rejected them. `tier="visual"` now requires `visual_verified`; the PDF
recomputes the visual count from `visual_verified` (0). This changes only the
tier label, not answers, states, claims or probe verdicts. It is reported as a
post-hoc presentation fix, not a protocol result.

## What this supports

Supported:

- The claim/schema/region/examiner pipeline runs end-to-end on a deck-disjoint
  corpus (47/47, 0 errors) with no crashes or unresolved citations.
- 16 of 19 shown answers are correct; 5 of 6 verified answers are correct.
- 15 of 19 probes are `nothing_found`; 18 of 19 are flagged (not verified).
- The production independent Groq examiner can verify a claim-level answer
  (`idx=0`) when quota allows.

Not supported:

- Any claim of independent verification at scale: the pilot examiner is
  same-family and the independent probe check is blocked by quota.
- Stable estimates: n=28 gold, n=19 probes, one run, fake embeddings and
  keyword retrieval only.
- Zero false-verified probes: one near-miss was verified.
- Visual grounding gains: no visual-verified rows; the image-only item was
  shown but not verified.

## Next, in order

1. Re-run the frozen pilot with an independent examiner when Groq quota
   permits; do not change the frozen labels or thresholds.
2. ~~Add question-constraint checking for the Section 12 class~~ **Done**
   post-hoc (above); re-evaluate on a fresh frozen set, not this one.
3. Do not expand the frozen set until the independent examiner path is stable;
   the report's 200-300 item target is still unmet.
4. Keep the report-parity column labelled as a counterfactual, not the original
   report code path.


---

## Frozen3 (built 2026-10-23; no live result yet)

`data/slidevqa_frozen3` is a fresh deck-disjoint set: **25 decks / 500 pages /
120 gold / 60 verified-absent probes**. It excludes all 8 dev decks and all 6
`frozen2` decks. Gold audit: 67 ok, 35 short-answer, 7 OCR-typo, 9 partial,
2 image-only, 0 page mismatch after 7 pre-run corrections. Probes: near-miss
15, absent-attribute 13, entity-swap 22, false-bridge 10; every check string was
verified absent from local OCR.

The local `data/index_frozen3_local` uses fake embeddings only for audit/probe
checks. The real ColQwen index and all 180 query vectors are built by the
self-sufficient `notebooks/frozen3_colab.ipynb`, which clones `main` and needs
no upload. The pre-registered protocol is `PREREGISTRATION_FROZEN3.md`; hashes
are in `data/slidevqa_frozen3/frozen_meta.json`.

**No live frozen3 result exists yet.** The numbers above this section are the
`frozen2` pilot (pre-fix, same-family examiner). Do not present frozen3 as an
improvement, and do not tune on it before its first run.
