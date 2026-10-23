# Research Guide: turning altVRAG into a defensible paper

This guide is the protocol for producing final metrics that can survive review.
It is intentionally conservative: a smaller, honestly measured result is worth
more than a large number obtained on data used to design the system.

## 1. The one rule

**The current 40 gold items and 41 probes are a development set.** They have
already influenced many fixes. Do not tune on them further and then present
them as a held-out result. Build a fresh, deck-disjoint test set, freeze the
code and configuration, and run it once.

Everything below follows from that rule.

## 2. Current state (development evidence)

Run 9 (`results/demo/testrun8_20261019_141118`) is useful for diagnosis, not
for the paper's main table:

- 186 merged rows, 133 completed, 53 resumable quota errors.
- Examiner ran on 49/92 completed gold rows; 43 were unavailable.
- 9 verified rows, 3 of them wrong; all 3 wrong rows are the ambiguous Alcatel
  title/tagline item.
- 20 examiner-confirmed rows were not verified; 19 of those were correct.
  The details check is the largest immediate blocker.
- Near-miss probes: 7/8 were answered (only 1 reached `nothing_found`), even
  though the aggregate "flag rate" looked like 1.0.
- Image-only answers are never correctly verified; they cite `sent_id=-1` and
  have no resolvable quote.
- The 18-column gate refit selected `tau=1.0` and accepted nothing. It is
  display-only and must not be presented as a contribution.

These observations justify the fix order in Section 5.

## 3. Test-set design

### 3.1 Split by deck, not by item

- Current 8 decks are dev.
- Draw 200-300 test questions from new SlideVQA decks.
- Keep decks disjoint between dev and test.
- If possible, build one frozen index containing both dev and test decks and
  split questions by deck. Otherwise report corpus-size differences as a
  limitation.

SlideVQA is large enough: 2,619 decks and 2,215 test QA pairs. Do not treat the
current 8-deck subset as representative.

### 3.2 Stratify

Preserve approximate proportions of:

- single-hop textual answers,
- multi-hop / bridged questions,
- visual-only or image-only answers,
- answer types (single-span, multi-span, non-span).

Define these categories in the paper. The current labels are project-specific
derivations, not official SlideVQA categories.

### 3.3 Probes

Build 100+ unanswerable probes from test decks:

- absent-attribute,
- near-miss,
- entity-swap,
- false-bridge.

Give near-miss at least 25 items; it is the weakest and most informative
category. Verify absence against the whole corpus, then spot-check the labels
manually.

### 3.4 Label audit

Audit labels blind to model outputs where possible. Record every change in
`labels_log.md`.

Known issues:

- Alcatel item: the question asks for the presentation title, the on-slide title
  is "ALCATEL-LUCENT OVERVIEW 2012", and the gold answer is the tagline
  "AT THE SPEED OF IDEAS". Accept both or rewrite the question.
- `scripts/audit_gold.py` flags 18 short answers and 3 page mismatches. Review
  them manually.
- If a label is changed after seeing outputs, report results both before and
  after the change.

### 3.5 Freeze

Before the test run, freeze:

- git commit,
- `config.yaml`,
- prompts,
- thresholds,
- retrieval settings,
- model IDs and API versions,
- decision rules,
- analysis plan.

Write `PREREGISTRATION.md` with primary and secondary metrics, exclusions, and
the exact test commands.

## 4. Metrics

### 4.1 Primary: selective prediction

Report these together; no single number is enough:

- **Verified precision**: correct / verified, with a 95% Clopper-Pearson CI.
- **Verified yield**: verified / all answerable gold items. This exposes
  examiner outages instead of hiding them in a restricted denominator.
- **Answer coverage**: shown / all answerable gold items.
- **Risk-coverage curve**: accuracy at `verified` only, and at
  `verified + not_confident`.
- **False abstention**: answerable items ending in `nothing_found`.
- **Probe answered rate**: probes with a shown answer / probes. This is the
  metric that exposes near-miss failures.
- **Probe false-verified rate**: probes reaching `verified` / probes. Target 0.
- **Unflagged-wrong rate**: wrong and verified / wrong.
- **Examiner leniency**: examiner-confirmed but wrong / examiner-confirmed.
- **Source accuracy**: shown answers with boxes or refs / shown answers, plus
  IoU and sentence F1 on evaluable rows only.

### 4.2 Secondary

- Retrieval: `page_hit@1`, `page_hit@3`, MRR.
- Verification: quote pass rate, citation-resolution rate, details-block rate,
  examiner availability.
- Grounding: sentence precision/recall/F1, IoU, word F1.
- Cost: reader calls, examiner calls, wall-clock latency.

### 4.3 Denominators

- Report examiner availability separately.
- Do not make "examiner-only precision" the headline.
- Report text-verified and visual-verified separately if a visual state exists.
- Always show the evaluable denominator for grounding metrics.
- Rename the current `probe_abstention` field: it currently means "not
  verified", not "nothing found". Report `probe_flag_rate`,
  `probe_nothing_found_rate`, and `probe_answered_rate` separately.

### 4.4 Sample size and intervals

- 50 verified with 0 errors gives a Clopper-Pearson 95% lower bound of about
  0.93; 49/50 gives about 0.89.
- 100 verified at 95% gives a lower bound of about 0.89.
- Target at least 50 verified with 0-1 errors, or 100+ verified at about 95%.
- With 200-300 test items and a 40-60% verified yield, this is achievable.
- Use cluster bootstrap by deck for CIs and paired comparisons.
- Use McNemar's test for paired system comparisons; report exact counts.

## 5. Fix order on dev

### P0 - Labels and protocol

Fix the Alcatel label and review short answers before changing the examiner.
The three verified-wrong rows are currently the whole precision failure.

### P1 - Examiner availability

- Remove dead Groq model IDs; verify every examiner model with
  `models.list()` before a run.
- Send only the region crop. If no region exists, use the cited-sentence or
  answer crop, not the full page.
- Keep verdicts cached by model + prompt + image hash, and resume missing
  verdicts across days.
- Record every attempt and every per-model error.
- Add a second cross-family provider when available.
- Target: examiner ran on at least 90% of answered gold rows.

### P2 - Details-check scope

- Hard-check only years, dates, numbers, and units, against the full evidence
  page, deck title, and examiner quote.
- Soft-check named entities against page/deck/corpus text, fold OCR variants,
  and skip generic business/tagline words.
- A `warn` details failure must not block VERIFIED unless the missing detail is
  a hard year/number/unit.
- Add `details_block_rate_on_correct`.
- Target: correct answers blocked by details <5%, with no probe regression.

### P3 - Short quotes and citations

- Normalize case, spacing, `%`, currency, separators, and OCR confusions.
- Try exact normalized containment first.
- For short quotes (<=4 tokens), use token-set containment rather than
  token-F1.
- Match the examiner quote against page OCR near the cited region.
- Fix table-cell citation so answers like `12.5 bn` get a resolvable quote.
- Target: exact-quote false negatives <5%.

### P4 - Examiner leniency

- Fix the ambiguous label first.
- Add a blind-examiner ablation: answer from the image, then compare.
- Keep claim-confirming as the default until dev shows a reason to switch.
- Target: Alcatel-class false confirmations = 0.

### P5 - Image-only path

- Add high-DPI OCR of the reader-declared region, producing a real quote and
  sentence/region reference.
- Or add a separate `verified_visual` state with its own precision and
  coverage. Never mix it with text-verified counts.
- Target: image-only correct-verified n > 0, probe false-verified still 0.

### P6 - Multi-hop retrieval

- Decompose the question and resolve bridge entities before retrieval.
- Read top-2/3 pages.
- Precompute query vectors for all dev and test questions on Colab and run
  hybrid on the full set.
- Target: `page_hit@1 >= 0.75`, `page_hit@3 >= 0.90`, MRR >= 0.85.
- Do not present the 24-item hybrid selection as a general retrieval result.

### P7 - Near-miss probes

- If a year/number/entity is provably absent from the whole corpus, do not
  answer from a neighboring fact. Say "No 2019 data found; nearest year is
  2011" and route to `nothing_found` or a defined constrained-abstention state.
- Report `probe_answered_rate` by category.
- Target: near-miss answered rate substantially reduced; probe false-verified 0.

### P8 - Gate

- Remove the gate from the decision path and from the paper's contributions.
- Report the `tau=1.0` refit as a negative result.
- If a display score is needed, label it display-only. A small calibrated model
  on `examiner_supported`, `details_ok`, and `quote_seen` is optional, but only
  if it passes proper cross-validation and improves the risk-coverage curve.

## 6. Baselines and ablations

Run on dev, then freeze and run on test:

- Reader only, no verification (always answer).
- Reader + self-consistency only.
- Reader + same-family self-check.
- Full system.
- Ablations: remove examiner, remove details, remove quote check, remove bridge
  pages, remove lexical rerank, keyword vs hybrid, remove multi-page reading,
  remove visual tier.

Use two reader models only if both cells are complete. Finish model B on paid
quota or drop it from the main table. A 28/81 cell cannot be compared to a
complete cell.

## 7. Paper structure and claims

Suggested structure:

1. Introduction: verification/abstention gap in document QA.
2. Related work: SlideVQA, selective prediction, LLM-as-judge.
3. System: retrieval, reader, verification, three-state contract.
4. Evaluation protocol: deck split, probes, metrics, CIs.
5. Results: main table, risk-coverage curve, probe table, ablations.
6. Error analysis and negative results.
7. Limitations and reproducibility.

Claims that can be supported with a proper test set:

- The three-state contract increases coverage while keeping probe
  false-verified at zero.
- Cross-family examination catches errors that same-family checks miss.
- Hybrid retrieval helps on hard items when evaluated on the full set.
- Source grounding is strong where evaluable.

Claims to avoid:

- "95% verified precision" from the current 9 verified rows.
- Presenting the failed gate as a contribution.
- Mixing visual and text verification.
- Generalizing the 24-item hybrid result.
- Comparing a partial model B cell.
- Relabeling after seeing outputs without disclosure.

Include an error taxonomy with counts: retrieval miss, OCR miss, table
segmentation, multi-hop, image-only, label ambiguity, examiner false confirm.

## 8. Reporting checklist

- Main table by category and state.
- Risk-coverage figure.
- Probe table by category with answered rate and false-verified rate.
- Retrieval table with page_hit@1/@3 and MRR.
- Ablation table.
- Examiner availability and leniency.
- Latency and API cost.
- Limitations: one benchmark, small dev set, OCR dependence, synthetic probes,
  free-tier quota, examiner leniency, label ambiguity, gate failure.
- Reproducibility: commit hash, config, prompts, run directory, model IDs and
  dates, cache policy, exact commands.

## 9. Execution order

1. Freeze dev and audit labels; write `PREREGISTRATION.md`.
2. Fix P1-P3 on dev; rerun model A keyword.
3. Add P5-P7; rerun dev; record ablations.
4. Build the test set/index on Colab; precompute vectors; freeze everything.
5. Run the test set once.
6. Analyze, write, and package the reproducibility artifacts.

The immediate goal is not to make the current rows look better. It is to make
the verification path correct on dev, then measure it honestly on new decks.
