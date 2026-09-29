# Findings

Revision 2 (2026-09-28): Q10 false positives corrected; raw/Core and optional Q5-filtered train views are separate. Current decisions are in PLAN.md and DECISIONS.md. Historical spot-check statements below are not a new manual adjudication of every record.

This document has four parts:

- **Part A**: how the assignment's requirements were read, including ambiguities and how they were resolved.
- **Part B**: what the Phase 1 data analysis found. Every number comes from `make analyze`; the full evidence
  (per-record flags, examples) is in [reports/data_analysis.md](../reports/data_analysis.md) and
  `reports/quality_flags.jsonl`.
- **Part C**: how reliable the heuristic checks are.
- **Part D**: open questions.

Counts are written *train / val / test* unless stated otherwise.

---

## Part A - Requirements interpretation

### A.1 What must be delivered (Core)

| # | Deliverable | Requirement details | Where it will live |
|---|---|---|---|
| 1 | Data formatting pipeline | Chat/instruct format; a clear, consistent table serialization; tool-call examples teach the invocation pattern; answer-type distribution preserved across splits | `src/clinqa/formatting.py` (Phase 2) |
| 2 | Fine-tuning script | Runnable SFT (LoRA/QLoRA/full); free choice of base model, with the choice explained | `src/clinqa/train.py` (Phase 4) |
| 3 | Data quality analysis | >= 5 formatted examples (one per type + one more) showing exact input and expected output; split counts, type distribution, average note words, tool-call frequency, table type distribution; issues found and how they were handled | `reports/data_analysis.md` (statistics + issues done; formatted examples in Phase 2) |
| 4 | Evaluation script | Per-type metrics: extractive accuracy, numeric reasoning accuracy, tool selection accuracy, tool argument accuracy, uncertainty detection rate; limitations explained | `src/clinqa/evaluate.py` (Phase 5) |
| - | Report | Setup, exact commands, hardware and runtime, artifact paths, what was tried, what worked or did not, results, limitations, next steps | `reports/REPORT.md` (Phase 7) |
| - | Reproducibility | Pinned versions, documented seeds, deterministic where feasible, runnable with minimal manual steps, test set never used for training | `uv.lock`, `requirements.txt`, `Makefile`, configs |

### A.2 Tool requirements

- Implement **exactly two** tools with the specified signatures and formulas: `unit_convert(value, from_unit,
  to_unit, substance)` (9 listed conversions) and `calculate_bmi(weight_kg, height_cm)` rounded to 1 dp. Unsupported
  conversions return an error string. Implemented in `src/clinqa/tools.py`; all 662 gold tool calls reproduce
  exactly (Q4).
- Evaluation looks at four things: *did the model decide to call a tool*, *did it pick the right tool*, *are the
  arguments valid and do they match the note*, and *does the final answer use the result with brief clinical
  context*.
- The dataset deliberately records some notes in lb/in while gold arguments are in kg/cm. The model has to handle
  that conversion (see D-003 for how).

### A.3 Ambiguities and how they were resolved

| Topic | Ambiguity | Resolution |
|---|---|---|
| File names | Spec says `train.jsonl`/`val.jsonl`/`test.jsonl`; the provided files are `_data/train_4089.txt` etc. | Content is JSONL with the expected counts. Copied byte-for-byte to `data/{split}.jsonl` (the path used by the spec's Quick Start). Originals are untouched and verified with sha256 (D-010) |
| `requirements.txt` | Held the assignment text, but the spec asks for pinned requirements | Assignment moved verbatim to `docs/ASSIGNMENT.md`; `requirements.txt` is now exported from `uv.lock` (D-006) |
| "Preserving the answer_type distribution across splits" | Could mean "do not re-split" or "keep the proportions when filtering" | The provided splits are already stratified (40/20/25/15 in each). The formatter keeps every record; if P-012 filters records, the exact change in proportions is reported |
| "Do not modify originals" | Applies to data files | `_data/` and `data/*.jsonl` are never edited; cleaning is done with flags and filters at formatting time |
| Stretch B spec | An earlier reading (first ~200 lines only) looked truncated after the signature. **Correction**: the full text continues: "The reference file contains drug interaction data or clinical guidelines (we provide it, versioned in the repo). Show how the model learns when to consult the reference vs. answer from the note alone." | The Stretch B goal is the *decision* to consult the reference, not just the lookup. The file is `_data/reference_6223.txt`, copied to `data/reference.jsonl` |
| Stretch C | Not mentioned in the first plan | A third option exists (2-turn follow-ups, 10-20 curated examples). Not planned (see PLAN §2) |
| Evaluation criteria | Not mentioned in the first plan | Weights: formatting 10%, fine-tuning setup 40%, evaluation 10%, code quality 20% (incl. **version control**), going beyond 20%. Git was initialised in Phase 0 |
| Tool arguments for imperial notes | "The model must handle the implicit conversion" but gold has a single `calculate_bmi` call | Original single-call targets for Core; explicit sequential chains only in optional R3. Score every step and visible-input consistency, not just the final call |
| Standard reference ranges | Extractive gold answers sometimes cite a reference range that is not in the input (Q11) | Tracked as a system-prompt policy decision (P-003) |

---

## Part B - Data findings

### B.1 Basic facts

| Fact | Value |
|---|---|
| Records | 2000 / 250 / 400 (spec: 2000 / 250 / 400) |
| Answer types | Identical proportions in every split: 40.0% extractive, 20.0% numeric_reasoning, 25.0% tool_call, 15.0% uncertain (val: 24.8% / 15.2% because of rounding with n = 250) |
| Note length | Mean 274.5 words (train); range 217-377; about the same in all splits |
| Approx. tokens (chars / 4) | Prompt (note + table + question) mean 553, p95 618, max 704; answer mean 61, max 145. Exact counts come in Phase 3 |
| Tables | Labs 1393 / 184 / 288, vitals 607 / 66 / 112. Eight lab panels, each with a fixed set of rows and fixed reference ranges. Plus 3 atypical tables (Q14) |
| Tool calls | Exactly **one** call per tool_call example. train: `calculate_bmi` 410, `unit_convert` 90 |
| unit_convert usage | Only 3 of the 9 specified conversions ever occur: mg/dL -> mmol/L glucose (21 train), mg/dL -> mmol/L cholesterol (29), mg/dL -> μmol/L creatinine (40). Results are rounded to 2 dp |
| Reference file | 149 entries: 64 drug_interaction, 48 clinical_guideline, 18 clinical_reference, 11 contraindication, 8 dosing_guideline |
| Leakage | No duplicate `(note, question)` pairs and no reused notes within or across splits; the maximum word-5-gram Jaccard between a test note and any train note is 0.20 (Q3/Q18). The 7 test `(question, answer)` pairs that also occur in train are short templated extractive answers with coincidentally equal values, attached to different notes |

### B.2 Issues that affect training or evaluation

| # | Finding (evidence) | Impact | Proposed handling |
|---|---|---|---|
| F1 | **Ungrounded tool calls (Q5, error)**: 95 `calculate_bmi` examples (78 / 7 / 10) have gold arguments whose weight and height appear **nowhere** in the note, table or question. The questions say "based on her recorded weight and height" and the gold answer quotes numbers like "88.8 kg and 169.1 cm" that are not in the input. Checked by hand: the only matches in these notes are weight *changes* ("5 kg of unintentional weight loss") or plan text ("calculate BMI and reassess") | Training on them teaches the model to **make up tool arguments**, which is exactly what the uncertainty objective forbids. At test time, 10 test targets reward hallucination | R1 keeps all rows; optional R2 excludes train Q5 candidates only, with a review packet. Full test and Q5-grounded diagnostics are both reported |
| F2 | **Measurements only in the question (Q5)**: 67 BMI examples (50 / 4 / 13) state weight and height only in the question ("based on a weight of 77.4 kg and height of 161.5 cm") | Valid, but the question is part of the evidence. The system prompt must not say "use only the note" | P-003 wording: "use the note, table and question" |
| F3 | **Imperial conversions do not round-trip (Q6)**: among grounded BMI examples that involve lb/in (103 / 11 / 24), 51 inch heights differ from the gold cm value by up to 0.126 cm after conversion (the generator appears to have derived inches from cm). Using exact 2 dp `unit_convert` results changes BMI by +/-0.1 in 20 of 138 cases | Exact-match argument scoring would unfairly penalise correct behaviour; chained traces must carry the tool's own output forward | Separate strict gold agreement, input-derived conversion consistency and actual-result consistency. Rounding diagnostics do not justify a blanket tolerance |
| F4 | **Stated BMIs are unreliable (Q7)**: 709 / 89 / 139 notes state a BMI. Of 619 notes that also give weight and height, 119 disagree with their own numbers by > 0.5 (39 by 0.5-1, 33 by 1-2, 30 by 2-5, 17 by >= 5). No tool_call BMI note and no uncertain BMI note states a BMI | Stated BMIs are distractors. A model that copies them into answers would be wrong | Targets always come from the tool. P-003 can tell the model to recompute rather than trust a stated BMI |
| F5 | **BMI decision boundary (Q8)**: BMI questions are tool_call with both measurements available 439 times, tool_call with them missing 95 times (= F1), and uncertain with at least one missing 148 times. No uncertain example has both | Apart from F1, "both weight and height available" separates *call the tool* from *say what is missing* perfectly. It is the key behaviour to learn and to evaluate | Evaluate the call-vs-abstain decision as a confusion matrix on BMI questions (P-010) |
| F6 | **Corrected checker defect (Q10)**: the three prior flags were false positives. Gold contains `4.97x`, `2.69x` and `4.19×`; regex backtracking shortened those decimals. The boundary fix now yields zero Q10 flags on the supplied data | The checker, not the gold arithmetic, was wrong | Keep train_1890 and test_306; add positive and adversarial regression tests; do not widen numeric tolerance to hide the bug |
| F7 | **External reference ranges (Q11)**: 46 extractive answers (36 / 3 / 7) cite a standard range (for example "7-20 mg/dL" for BUN) when the table is a different panel and the value comes from the note text. All other extractive numbers can be found in the input | Mild use of outside knowledge; the ranges are the dataset's own standard ranges | Keep. P-003 decides whether the system prompt allows standard reference ranges |
| F8 | **Micro sign vs Greek mu (Q13)**: 56 tool_call examples write µ (U+00B5) in the question/answer but μ (U+03BC) in the tool arguments; tables and tool arguments always use μ | String-exact argument matching would fail on a harmless difference | Already handled: `normalize_unit` treats µ / μ / u as the same; the evaluator will reuse it |
| F9 | **Label boundary extractive vs numeric (Q15)**: 65.1% of extractive questions ask whether a value is "within the normal reference range" (a threshold check); no question template is used under both labels | "Extractive accuracy" includes simple threshold comparisons | Keep labels. Explain this in the report. The metric checks both the value and the direction (above/below/within) |
| F10 | **Atypical tables (Q14)**: train_278 (vitals + Weight + Height rows), test_004 (vitals + Weight), train_1905 (mixed labs with "Fasting Glucose" and a different creatinine range) | The table can also be a source of body measurements | The serializer is generic; grounding already searches table rows |
| F11 | **Empty unit cells (Q1)**: all 232 INR rows have `""` as unit | Must render cleanly (e.g. "-" or omitted) | P-002 |
| F12 | **Answer length (Q17)**: 35 gold answers have more than 3 sentences; numeric/uncertain answers average 51-57 words, max 92 words (about 145 tokens) | Sets `max_new_tokens` (about 256 covers everything) | Keep |

### B.3 Dataset characteristics (not errors)

| # | Finding | Implication |
|---|---|---|
| C1 | Uncertain sub-types from the gold answers: train allergy 56, dose 63, height 65, weight 51, timestamp 65 (all 300 classified; all non-timestamp cases verified against the note in Q8) | Balanced coverage of the 4 kinds of missing information the spec lists |
| C2 | Allergy documentation: no allergy statement 1528 / 187 / 289, NKDA 437 / 57 / 103, "Not documented" 35 / 6 / 8 | The model must tell "no allergy section" apart from "NKDA" |
| C3 | Note formats: about 40% markdown-style (`**CC:**`, 817 of 2000 train notes), the rest plain; section headers vary (`HPI` vs `HISTORY OF PRESENT ILLNESS`) | Keep notes verbatim; no normalisation |
| C4 | Weight/height units in notes are mixed: kg+cm, kg+in, lb+cm, lb+in, sometimes with a metric restatement ("66.0 in (167.6 cm)", "263.0 lb (119.3 kg)") | The chained trace skips conversions when a metric value is already written (D-003) |
| C5 | Distractor numbers: weight changes ("8 lbs of weight gain"), dimensions ("3 x 2 cm ulcer"), "5/5 in all extremities", dosing thresholds ("weight ≤60 kg") | The parser filters these (tested); the model must too |
| C6 | Synthetic-data artefacts (Q16): BPH/prostate for female patients 107, female-only conditions (e.g. "heavy menstrual bleeding") for male patients 20, Hct/Hgb ratio outside 2.4-3.6 in 92 CBC tables, 86 notes that point out their own chart discrepancy ("likely coding error") | Deliberate noise. Kept; the answers do not depend on it |
| C7 | Co-prescribed drugs of the same class (Q16): ACE/ARB 40, beta-blockers 27, anticoagulants 25, DPP-4 18, SGLT2 14, ... | Natural material for Stretch B interaction questions |
| C8 | Question templates: tool_call questions are highly templated (221 templates for 662 questions; the top template alone covers 116); uncertain and numeric_reasoning questions are nearly all unique | The tool-call *decision* must not be learned from question wording alone, because uncertain BMI questions look the same (see F5) |

### B.4 Reference file (Stretch B)

- Keys are unique and well-formed; categories match the key prefixes (Q19).
- Drug-pair keys are **not** in a canonical order (32 alphabetical, 32 not), and no pair appears twice. Lookup must
  be order-insensitive.
- 40 drugs appear in interaction keys; 630 notes (474 / 63 / 93) contain at least one covered pair. The most common
  are pairs with metformin (albuterol, spironolactone, sertraline, ...).
- Six frequent note drugs have **no** interaction entry (pantoprazole, rosuvastatin, montelukast, bisoprolol,
  hydrochlorothiazide, insulin lispro). They are natural "not in reference" negatives.
- The values spell micro units as ASCII `ug`/`uL`/`umol`, while the notes use μ.

---

## Part C - Reliability of the heuristic checks

The checks are regex-based. Each check that raises flags was spot-checked, and false positives found along the way
were fixed in the parser, not waved away:

| Check | Spot-check | Result / fixes made |
|---|---|---|
| Q5 ungrounded args | All 95 notes scanned for any weight/height/kg/lb/cm mention | Only weight-change phrases and plan text found; confirmed ungrounded |
| Q6 rounding | Confirmed arithmetically (68.1 in -> 172.97 cm vs gold 173.1) | Systematic: inch values are rounded restatements of cm |
| Q7 stated BMI | 17 flags read in context (7 + a random 10) | Found and fixed a parser bug ("BMI 30.7 kg/m²" was read as a 30.7 kg weight). After the fix, all sampled flags are genuine inconsistencies |
| Q8 label consistency | All flags reviewed across 3 iterations | Fixed: formula text ("BMI = weight in kg / height in m²") and generic "requires both height and weight" were read as "weight missing"; "dose of X" questions picked the wrong drug; dosing thresholds ("weight ≤60 kg") counted as measurements. Final: 0 contradictions, 398/398 uncertain examples classified |
| Q10 arithmetic | Previous manual-review conclusion was incorrect | Revision 2 reproduces and fixes decimal backtracking at x/× suffixes. Original three flags are withdrawn; regenerated Q10 has zero candidate issues. Heuristic coverage remains limited |
| Q11 extractive numbers | 10 random flags read | All are standard reference ranges not present in the input |
| Q12 note vs table | All 6 initial flags read | 3 false positives fixed ("PT/INR 17.9 / 1.8", "calcium 1.3 mg/dL above upper limit"); 3 are lower-precision restatements (ALT 133 vs 133.3), now reported separately. Final: 0 conflicts |
| Q16 plausibility | Samples read | Fixed: "cervical" (spine) and "outer" had matched female-only terms. Remaining samples are genuine |

Known limitations:

- The medication parser reads the "Medications" section only; drugs mentioned only in HPI/Plan are missed (this
  affects only Q19 coverage counts).
- The uncertain classifier uses the gold *answer* text, so it describes what the gold says is missing. For dose,
  allergy, height and weight it is also checked against the note; timestamp claims cannot be checked by machine.
- Approximate token counts assume 4 characters per token.

---

## Part D - Current processing and remaining work

Raw Core retains 2,000 rows. Optional Q5-filtered train view has 1,922 (800 extractive, 400 numeric, 422 tool, 300 uncertain), and keeps train_1890. val/test are not filtered. The view manifest records that candidates have not been newly human-adjudicated; inspect q5_review.jsonl before selecting the filtered model as preferred.

Strict JSON/schema checks run before feature extraction. Live input hashes are verified before generating reports or views. Overlap checks do not establish absence of semantic/template leakage. Test was already inspected for audit; future model/threshold selection uses val only.

Next work is native chat formatting and loss-mask auditing, followed by a real tool runner and SFT. Exact token counts, GPU pins and model results remain pending. RL is not part of the current plan.
