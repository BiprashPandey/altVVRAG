# Pre-registration: frozen deck-disjoint pilot (2026-10-22)

Status: written before the live run. No tuning of thresholds, prompts, or
labels after this point. Any post-run change is reported as post-hoc.

## Question

Does the claim-level citation/verification contract preserve the report's
honest behaviour on documents never used for development: answers that are
shown are more often correct, unsupported claims/regions are withheld or
flagged, and absent attributes are not answered?

## Frozen corpus

- 6 SlideVQA decks, disjoint from the 8 development decks.
- 120 pages; 117 with OCR; 4,151 word boxes; 1,109 sentence/table units.
- Index: `data/index_frozen2`, manifest hash
  `e0e5ac327b131ba28ed920d4ef2564bd55fb1ee62116857a329b5cd7f87b67ad`.
- Embeddings: deterministic fake 32-d. Retrieval is keyword-only; this is a
  pipeline/verification test, not a visual-retrieval test.
- Dev decks excluded via `data/frozen_exclude.json`.

## Frozen items

- Gold: `data/slidevqa_frozen2/gold.json`, sha256
  `f600b2f8ce5777bb44bf12de18a620eb577ea8a1ba7bc10dcc84016942d88cbc`.
  28 items: single-hop 22, multi-hop 5, image-only 1. Audited pages and labels:
  `results/frozen_gold_audit3.json`.
- Probes: `data/slidevqa_frozen2/probes.json`, sha256
  `c15095e9931cc24cefa0210e182d4b3bc666c39ce1a3386268f1ec849520a125`.
  19 probes: near-miss 10, entity-swap 4, false-bridge 4, absent-attribute 1.
  Each probe string was checked absent from normalized OCR before the run.
- These sets are small (not the planned 200-300 gold / 100+ probes); results
  are a protocol-valid pilot, not a stable final estimate.

## Configuration

- Pilot config: `config.pilot.yaml`, sha256
  `b732821096e00baa48f984e2565bb21cdfdad8e9daa0bae40261da10b12a8614`.
  It differs from `config.yaml` only in:
  - `verify.examiner_provider: gemini`
  - `verify.examiner_models: [gemini-3.5-flash-lite]`
  - `verify.independence_required: false`
- Reader: Gemini 3.5 Flash Lite, one sample per item (`--n3 0`), no rotation.
- Examiner: Gemini 3.5 Flash Lite. This is SAME-FAMILY and NOT independent;
  the pilot measures the claim pipeline, not independent verification.
- Examiner evidence: `page_and_region`, one claim crop (`claim_level_images: 1`).
- Claim level: enabled, max 5 claims; citation-specific required.
- Gate: calibrated artifact loaded but display-only; the gate does not decide
  the three-state outcome.
- Retrieval: keyword-only, full corpus (`doc_id=None`).
- Code: git `4f23dec` plus uncommitted working-tree changes. The run is
  reproducible only with this working tree; record `git diff` separately.

## Primary outcomes (fixed before the run)

Gold (answerable, n=28):
1. Strict verified precision = correct verified / all verified.
2. Verified yield = verified / 28 (coverage of the strict arm).
3. False abstention = nothing_found / 28 (shown-answer arm).

Probes (unanswerable, n=19):
4. Probe false-verified rate = probes in state `verified` / 19.
5. Probe answer rate = probes not in `nothing_found` / 19 (secondary; a
   not_confident answer is still an answer shown to the user).

Secondary: shown correctness, claim support rate, examiner availability,
source accuracy, page hit @1 / MRR, per-category results, state counts.

## Decision rules

- Do not tune on this run. If it fails, report the failure and leave the code
  unchanged until a new frozen set is built.
- A result with examiner availability below 80% of answered gold rows is
  reported as "verification not exercised", not as evidence for the contract.
- A same-family examiner result cannot support a claim of independent
  verification. The independent-examiner check is reported separately and is
  not used to replace the primary pilot table.
- The report-parity arm is a strict-withholding counterfactual over the same
  enhanced rows, not the original report code path.

## Run command

```
VVRAG_CONFIG=config.pilot.yaml python scripts/run_test8.py   --index data/index_frozen2   --gold data/slidevqa_frozen2/gold.json   --probes data/slidevqa_frozen2/probes.json   --arms keyword --models gemini-3.5-flash-lite --n3 0
```

Report:

```
python scripts/make_frozen_report.py <run_dir> frozen_eval.pdf   --config config.pilot.yaml
```

## Known limits declared up front

- Small set; single run; keyword retrieval with fake embeddings.
- Same-family examiner; not independent.
- One claim crop only, for free-tier token budget.
- Image-only and multi-hop items remain hard; the one image-only gold item is
  expected to be a failure mode unless the visual tier fires.
- The pilot may be repeated only for a documented infrastructure error, not
  to obtain a better result.

## Outcome

Run complete 2026-10-22: `results/demo/testrun8_20261022_214946`; results and honest reading in `FROZEN_RESULTS.md`, report `frozen_eval.pdf`. The pilot used a same-family examiner, so it does not establish independent verification. One probe was falsely verified (question-constraint failure). A post-hoc deterministic Section-reference check fixes that probe, but the pilot metrics above are unchanged and this set must not be reused as evidence of improvement.
