# Label audit log

This log records label changes made before freezing the dev/test protocol. It
is intentionally separate from model results so changes can be audited and
reported before/after.

## Audit command

```bash
python scripts/audit_gold.py
```

Result on the 40-item dev gold set:

- `ok`: 19
- `short_answer`: 18
- `page_mismatch`: 3

The audit is a heuristic, not ground truth. Page mismatches and short answers
were reviewed manually.

## Changes applied

### Alcatel title/tagline items (2 items)

QIDs:

- `alcatel-lucenteesoverview-120403090342-phpapp01_95-17`
- `alcatel-lucenteesoverview-120403090342-phpapp01_95-18`

The question asks for "the title of the presentation ... for the OVERVIEW of
2012". The slide deck contains both:

- page 0 title: `ALCATEL-LUCENT OVERVIEW 2012`
- page 1 tagline: `AT THE SPEED OF IDEAS`

The original gold answer was the tagline. The audit flags the answer as a page
mismatch because it is not on the labeled page. This is a label ambiguity, not
an examiner error.

Change: keep the original gold answer and add both title variants to
`acceptable_answers`:

- `ALCATEL-LUCENT OVERVIEW 2012`
- `Alcatel-Lucent Overview 2012`

A `label_note` is stored on both items. This is an after-the-fact change made
after observing Run 9 outputs, so any paper table must report results with the
old and new equivalence rule.

Post-change dev check (Run 10, 2026-10-22): item 17 returned
`ALCATEL-LUCENT OVERVIEW 2012`, which is now counted correct and verified.
Item 18 returned the same answer, was counted correct, but the examiner was
unavailable (Groq 400 on a tiny crop), so it stayed `not_confident`.

## Reviewed but not changed

### `1shangxichinassubnationaldebts-issuesandsuggestions-160229201053_95-39`

The answer `Municipal-level government debts` is present on the gold page
(page 5), but OCR segmentation lowers the token coverage to 0.667. The same
phrase also appears on page 12. The gold page is defensible for the June 2013
table; this is an OCR/sentence-segmentation issue, not a clear page error.
Keep the label and report the evaluable denominator.

### Short answers

The audit flags 18 short answers. These are legitimate single-token or
single-number answers (years, counts, country/product names). They need
context-aware evaluation, not a blanket relabel. Do not change them without
manual adjudication.

## Rules for future changes

1. Audit blind to model outputs where possible.
2. Log every change with the old value, new value, reason, and date.
3. Report before/after results for any after-the-fact change.
4. Never change test labels after seeing test outputs.
