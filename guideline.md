# altVRAG Guideline: Answer, Verify, Show the Source

Status: written 2026-10-19, based on `PIPELINE.md` (rev 3) and `testRun8.pdf`.

---

## 0. The goal, in plain words

You ask a question about your documents. The system should:

1. **Always give an answer** (its best guess), never a blank "I abstain".
2. If it is not sure, write **(not confident enough)** right after the answer.
3. **Always show exactly where the answer came from**: deck, page, the highlighted spot, and the quoted text. The user can then check it with their own eyes.
4. Use **two different AI models**: one reads and answers (the student), one checks (the examiner). They must come from different companies, so the checker does not share the reader's blind spots. The examiner is **shown** the student's answer and checks it against the page.

Models you chose:

| Job | Model family | Config |
|---|---|---|
| Reader (finds the answer) | **Gemini** | `reader.provider: gemini` |
| Examiner (checks the answer) | **Groq** (a vision model hosted on Groq) | `verify.cross_family_provider: groq` |

> Note: you wrote "grok". The pipeline already has a Groq client (`GROQ_API_KEY`). I assumed that. If you meant xAI Grok, only the provider name and model in the config change; the design stays the same. Whatever you choose, the examiner must be a **vision** model, because many answers live in charts and tables.

Also: **answers can be any type**. A number, a name, a year, a sentence, a list. The system must not reject an answer because of its "type" (see section 4.1).

---

## 1. How the system should work (simple picture)

Think of a **student and an examiner**.

- The **student** (Gemini) reads the pages and writes an answer, with a quote and a pointer to where it found it.
- The **examiner** (Groq) is **shown the student's answer, quote, and location**, plus the same pages. Its only job is to check the student's claim against the page: "Is this quote really there? Does it really say this? Does it answer the question?"
- If the examiner confirms it, we trust it. If not, we still show the student's answer, but mark it **(not confident enough)** and say what the examiner found wrong.

**Why the examiner is shown the answer:** the student's biggest risk is making something up (an answer or a quote that is not on the page). The fastest way to catch that is to hand the examiner the exact claim and ask it to find it on the page. There is one design only: the examiner checks the student's claim. It does not answer the question separately.

### Step by step

1. **Find pages.** Use hybrid search (keywords + visual) and take the best 5 pages.
2. **Reader (Gemini)** reads the top pages and returns: answer, exact quote, where it found it (page + sentence or box), what kind of place (text / table / chart / image), and a confidence number.
3. **Quick automatic checks (no AI):**
   - Does the quote really exist on that page?
   - Do the years, dates, and names in the question appear in the evidence? (catches "as reported in 2021" when no 2021 data exists)
4. **Examiner (Groq)** is shown the question, the student's answer, quote, and location, plus the page images. It must say what it actually sees at that location and whether that supports the answer.
5. **Combine** the examiner's verdict with the automatic checks from step 3.
6. **Show the result** in one of three states (section 3), always with the source.

---

## 2. What the Run 8 test told us (and why we change things)

| What we saw | In simple words | What we do about it |
|---|---|---|
| About 60% of gold questions were "abstained" | The system said "I don't know" a lot, even when it had the right answer | Always show the best answer with a warning (section 3) |
| By my count, 5 of 6 abstentions that still had a candidate answer were correct (items 15, 16, 32, 38, 39 for gemini-3.8-flash) | The safety checks threw away good answers | Fix the checks that were too strict (section 4) |
| "Which … percentage" questions were rejected for "wrong answer type" | A rule guessed the answer must be a number, but the answer was a label | Remove the answer-type rule completely |
| Item 26: answer "Poland" was marked verified but wrong | The OCR glued cells from different table rows into one fake sentence, and the quote check trusted it | Read tables as rows and cells, and have the examiner check the quote against the page image |
| Item 18: marked verified but didn't match the gold | The check only asked "is this text on the page?", not "does it answer the question?" | The examiner must confirm the answer really answers the question, not just that the words exist |
| The visual checker rejected 8 answers that were actually correct, and never caught a wrong one | It is weaker than the reader on charts, and we never tested whether it can catch mistakes | Use a strong examiner on every answer, and test it with deliberately wrong answers |
| The usage log shows about 180 of 186 rows served by `gemini-3.5-flash-lite` | The "two models" comparison was mostly one model compared with itself | Never silently switch models; record which model really answered |
| The confidence gate has `tau = 0.0` | The confidence score currently does nothing, because there was almost no wrong-answer data to learn from | Rebuild the gate with real and synthetic wrong answers |
| Gold page was not rank 1 in about half the questions | The answer was often on a page the system never looked at | Make hybrid search the default and add re-ranking |
| `unans-02` ("as reported in 2021") got an answer | Nothing checked that "2021" exists in the documents | Add the question-details check |

---

## 3. What the user sees: three states

Every response has an answer or a clear "nothing found", and always a source.

| State | When | What is shown |
|---|---|---|
| **Verified** | The examiner confirms the answer, the quote exists on the page, and the question's details were found | The answer, with a green badge |
| **Not confident enough** | There is a reader answer but something is off: the examiner rejects it, the quote is not found, the examiner is unavailable, or a detail from the question is missing | The answer followed by **(not confident enough)**, a one-line reason, and what the examiner said was wrong |
| **Nothing found** | Neither model found any candidate answer | "I couldn't find this in the documents" and the closest passages |

Example reasons (short, human, specific):

- `(not confident enough: the examiner could not find this figure at the cited spot)`
- `(not confident enough: "June 2012" does not appear in the source; nearest is June 2013)`
- `(not confident enough: the examiner check could not be run)`

**Why the reason line matters:** a bare warning makes the user guess what is wrong. A reason tells them exactly what to look at. It also protects you in the cases where the question asks about something that is not in the documents. Without the reason line, the system would show a made-up answer for those questions with only a small warning.

### Source display (always)

- Deck name and page number.
- The page image with the exact words or region highlighted.
- The quoted text.
- For charts and images, the region box the model pointed to.
- If there are several sources, list them all and mark the main one.

**Why:** the user's trust comes from being able to check in five seconds. The source is the product as much as the answer.

---

## 4. The improvements (what, why, how)

### 4.1 Remove the answer-type check (your point 1)

- **What:** delete the rule that checks whether the answer is a number, a date, or a name. Set `verify.answer_type_check: false`. Remove `answer_type_match` from the gate features and refit the gate.
- **Why (simple):** the rule guessed what kind of answer the question wanted, and guessed wrong. "Which region has the highest percentage?" wants a region, not a percentage. Real questions have many answer shapes, and a guessing rule will always break on some of them.
- **What replaces it:** two better checks.
  1. The examiner checks that the shown answer really answers the question, so a wrong kind of answer gets rejected.
  2. The question-details check (4.4) makes sure the question's own details are present in the source.
- **Also soften the "both options present" rule** for "X or Y" questions. Make it a warning ("only 2015 was found in the cited text"), not a block. Table fragments often hold only one of the two values, which is how items 38 and 39 lost correct answers.

### 4.2 Always answer, with a confidence label

- **What:** remove the "UNANSWERABLE always abstains" shortcut. The reader prompt says: "Always give your single best answer with a quote and location. Use `no_evidence: true` only if there is truly nothing relevant." The system output becomes one of the three states in section 3.
- **Why:** the old design hid correct answers (see section 2). The new design never hides anything, but never overstates either.
- **Important:** a green "Verified" badge must be hard to get. Only an examiner confirmation plus the automatic checks earn it. Everything else is labeled.

### 4.3 Gemini reads, Groq examines, and they stay independent (your point 2)

**Rules:**

1. **The examiner is shown the student's work.** It gets the question, the student's answer, the student's quote, the claimed location, and the page images.
2. **The examiner must show what it sees.** It does not just say "yes". It copies the exact words it sees at the claimed spot (`quote_seen`). The code then compares `quote_seen` with the student's quote and with the OCR text. A "yes" with no matching evidence is thrown out.
3. **Attack first.** The prompt tells the examiner its job is to find what is wrong (a quote that is not there, a number from the wrong row, a wrong year), and to say "yes" only if it cannot find a problem.
4. **No outside knowledge.** The examiner may only use what is visible on the pages.
5. **No silent switching.** If Groq is down or out of quota, the row becomes **Not confident enough: examiner check could not be run**. It never falls back to a Gemini examiner. If Gemini is out of quota, the row waits or fails; it does not quietly use another model without logging it.
6. **Log everything.** Every row records `reader_served_by` and `examiner_served_by`. If both are the same family, mark `independence_violated: true` and exclude the row from "verified" counts.
7. **Strength.** The examiner should be at least as capable as the reader on charts. In Run 8 the Groq checker misread a bar chart. Use the largest Groq vision model you can, and test it on your chart items before trusting it.

**Why (simple):** the student can invent a quote or misread a table. An examiner holding the exact claim and the page can check it in one look. Using a different company's model means the examiner does not share the student's blind spots.

**Known risk, and how we handle it (no change of design):** an AI that is told "here is the answer" can be too agreeable. We keep the design and defend against this with the three safeguards above (show-what-you-see, attack-first, deterministic quote comparison), and we **measure** the risk with the wrong-answer test in 4.8. If the examiner accepts too many wrong answers in that test, we tighten its prompt and rules. We do not switch to a different scheme.

---

### 4.4 Check the question's own details

- **What:** pull years, dates, units, and named things out of the question (for example "2021", "June 2012", "20-year", "Zynga"). Check that each appears in the evidence text, or that the examiner confirms it from the image. If one is missing, label the answer "not confident enough" and name the missing detail.
- **Why:** it is cheap, needs no AI, and would have caught the one probe failure (`unans-02`) and the near-miss probes. It also gives the user a useful explanation.
- **Care:** treat this as a warning, not a ban. Sometimes a detail is in an image the OCR missed. The examiner can overrule the warning.

### 4.5 Fix tables and charts (the biggest source of wrong "verified" answers)

- **What:** at ingestion, detect tables and store rows and cells with their own boxes. Let citations point to a cell ("row Poland, column YoY growth"), not to a sentence that was built by gluing words left to right.
- **Why (simple):** a table is a grid. Reading it like a paragraph mixes up rows. That is how "Poland 126.1%" appeared, a fake phrase that was never on the slide. Once the grid is right, the quote check is trustworthy again, and the many `nli_low` rejections on tables go away.
- **For images and charts the OCR misses:** run the existing VLM page transcription (`--vlm-transcribe-empty`) so those pages also get text and boxes. Half the gold set is "image-only", and it scored the worst.

### 4.6 Better page finding (retrieval)

- **What:**
  1. Make `retriever.retrieval_mode: hybrid` the default.
  2. Replace the pre-computed query cache with an online query encoder, so new questions work without a Colab run.
  3. Add a re-ranker over the top 5 pages.
  4. For "the company that did X" questions, first resolve the bridge (which deck or entity), then answer.
- **Why:** if the right page is not in the top few, nothing downstream can help. Hybrid doubled correct page hits on the hard subset (7 to 14 of 24). Multi-hop was 0 of 5 under keyword search.

### 4.7 Rebuild the confidence score

- **What:** refit the gate using the ~186 labeled rows from Run 8 plus synthetic wrong answers (wrong year, swapped entity, neighbor-row value). Make the examiner's verdict (confirmed / partly / rejected, and whether its `quote_seen` matched) the strongest feature. Keep `tau` meaningful: the "Verified" badge needs a measured precision of at least 95%.
- **Why:** right now the score cannot tell good from bad, because the training data had 2 wrong examples. A score that does not separate good from bad makes the "not confident enough" label meaningless.
- **Also:** use `n_samples = 3` at least on the reader for questions where the examiner rejects the answer. In the live matrices n = 1, so there was no self-agreement signal at all.

### 4.8 Honest testing

- **Test the examiner with wrong answers.** Show it deliberately wrong answers (wrong year, neighbor-row value, made-up quote, wrong page) and measure how often it still says "yes". Because the examiner is shown the answer, this test is the most important one in the project. Do this before trusting the badge.
- **Fix the two-model comparison.** Run the experiment again with `served_by` logged and quota handled by waiting, not by switching.
- **Audit gold labels.** Check items like #18, where the "gold" and the slide text disagree, and add acceptable-page tolerance everywhere (as in item 10, whose evidence is on page 6 but gold says page 0).
- **Build the missing gold boxes** (many show "[image missing]"), so source-highlighting accuracy is measured on more than 5 to 6 items.
- **Report intervals.** With 40 items, differences of 1 or 2 questions are noise. Always show confidence intervals.
- **Fix report bugs:** the "X rows" placeholder in section 12, and truncated model names in the matrix headers.

---

## 5. Config changes (summary)

```yaml
reader:
  provider: gemini
  model: <pinned gemini vision model>
  fallback_provider: ""          # no silent fallback to another family
  retry_on_unanswerable: false   # reader always gives a best answer
  n_samples: 1                   # raise to 3 when the examiner rejects the answer

verify:
  mode: examiner_shown           # new: examiner is shown the reader's answer, quote, location
  cross_family_provider: groq    # the examiner; or xai if you meant Grok
  cross_family_model: <largest Groq vision model you can use>
  answer_type_check: false       # your point 1
  comparison_check: warn         # warning, not a block
  question_detail_check: warn    # new, section 4.4
  require_answer_support: false  # replaced by the examiner verdict + labels
  log_served_by: true

retriever:
  retrieval_mode: hybrid
  rerank_top_k: 5

output:
  always_show_answer: true
  low_confidence_suffix: "(not confident enough)"
  always_show_source: true
```

(Key names marked "new" do not exist yet. They are the settings the code changes in section 7 would add.)

---

## 6. Prompts (starting versions)

### Reader (Gemini)

```
You will see pages from documents and a question.
Give your single best answer, even if you are not fully sure.
Return JSON only:
{
  "answer": "...",                 // any form: number, name, year, sentence, list
  "quote": "...",                  // exact words from the page that support the answer
  "location": {"page": N, "sentence_ids": [..] or "box_frac": [x0,y0,x1,y1]},
  "found_in": "text|table|chart|image",
  "confidence": 0.0-1.0,
  "no_evidence": false             // true only if nothing relevant is on these pages
}
Do not invent a quote. If you cannot find a quote, say so in "quote" and lower the confidence.
```

### Examiner (Groq), shown the student's answer

```
You are an examiner. A student answered a question using the pages shown.

Question: <question>
Student's answer: <answer>
Student's quote: <quote>
Student's claimed location: page <N>, <sentence ids or box>

Your job is to find what is wrong with the student's claim.
Look at the claimed location on the page image. Use only what you can see.
Do not use outside knowledge. Say "yes" only if you cannot find a problem.

Return JSON only:
{
  "quote_found": true|false,          // is the student's quote really at that spot?
  "quote_seen": "...",                // the exact words you see there (empty if none)
  "answer_supported": "yes|partly|no",// do the words/figures you see back the answer?
  "answers_the_question": true|false, // is this the thing the question asked for?
  "problem": "..."                    // one short sentence, or empty
}
```

The code then checks `quote_seen` against the student's quote and the OCR text. If they do not match, the examiner's "yes" is ignored.

---

## 7. Decision rules (exact)

```
if student.no_evidence:
    state = NOTHING_FOUND                    # show closest passages
else:
    exam      = examiner(question, student.answer, student.quote, student.location, pages)
    confirmed = exam.quote_found
                and exam.answer_supported == "yes"
                and exam.answers_the_question
                and fuzzy_match(exam.quote_seen, student.quote)
    ok        = quote_exists_on_page            # OCR check
                and question_details_present
                and citation_resolved

    if confirmed and ok and examiner_ran and independence_ok:
        state = VERIFIED
    else:
        state = NOT_CONFIDENT                    # add reasons

# reasons come from: exam.problem, quote not found, missing question detail,
# examiner unavailable, independence violated
```

---

## 8. Metrics to track (replacing "abstention rate")

| Metric | Meaning | Target |
|---|---|---|
| Verified precision | Of answers shown as Verified, how many are correct | 95% or higher |
| Unflagged wrong rate | Wrong answers shown without the warning | As close to 0 as possible |
| Flag recall | Of wrong answers shown, how many were flagged | High (aim 90% or higher) |
| Answer coverage | Questions with a correct answer shown (any state) | Should rise from about 0.33 |
| Source accuracy | Shown source actually contains the answer (page hit and box overlap) | Measured on a larger gold-box set |
| Examiner leniency | Share of deliberately wrong answers the examiner confirms | Low (measured with the wrong-answer test in 4.8) |
| Probe flag rate | Impossible questions that end in NOTHING_FOUND or a clear reason | High, with a reason shown |

**Why these instead of abstention rate:** the old metric rewarded saying "I don't know". The new design shows answers, so what matters is: when we say "Verified", are we right, and when we are wrong, did we warn the user?

---

## 9. Work order (checklist)

**Phase 1: quick fixes**
- [ ] Remove the answer-type check and refit the gate without that feature.
- [ ] Make the comparison check a warning.
- [ ] Remove silent model fallback; log `reader_served_by` and `examiner_served_by`.
- [ ] Fix report placeholders and truncated model names.
- [ ] Add the question-details check (as a warning).

**Phase 2: the new behavior**
- [ ] Reader prompt: always best answer, `no_evidence` flag.
- [ ] Groq examiner (shown the student's answer) on every answer, not only as the visual rescue.
- [ ] Three output states with reason lines.
- [ ] Source display for every state.

**Phase 3: trust**
- [ ] Wrong-answer injection test on the examiner.
- [ ] Rebuild the gate on Run 8 labels plus synthetic wrong answers.
- [ ] Re-run the model matrix with served-by logging; no silent switching.
- [ ] Audit gold labels and build the missing gold boxes.

**Phase 4: recall**
- [ ] Hybrid as default; online query encoder; re-ranker.
- [ ] Multi-hop: resolve the bridge first.
- [ ] Table-aware ingestion (rows and cells) and VLM transcription for image-only pages.

Do Phases 1 and 2 first: they deliver your requested behavior and remove most of the lost correct answers. Phase 3 tells you whether to trust the badge. Phase 4 raises how many questions get a correct answer at all.

---

## 10. Plain glossary

- **Reader:** the AI that reads the pages and writes the answer (Gemini).
- **Examiner (verifier):** a different AI, from a different company, that is shown the student's answer, quote, and location and checks them against the page (Groq).
- **quote_seen:** the exact words the examiner says it sees at the claimed spot. Compared by code with the student's quote.
- **Citation / source:** the page, the highlighted spot, and the quoted words that show where the answer came from.
- **Abstain:** the old behavior of refusing to show an answer. In the new design the answer is shown with a warning instead.
- **Hybrid retrieval:** finding pages with both keyword matching and visual matching, then merging the two rankings.
- **Gate:** the rule that decides how much to trust an answer.
- **Calibrated:** the confidence number matches reality. If it says 90%, about 90% of such answers really are right.
- **Probe:** a trick question whose answer is not in the documents. A good system flags these.
- **Independence:** the reader and examiner do not share the same training family, so they do not make the same mistakes.