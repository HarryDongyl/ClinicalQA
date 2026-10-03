> **Archived 2026-10-03.** Superseded by `docs/STRETCH_A.md`. Kept unchanged below as a historical record; relative links and some script paths (now `scripts/legacy/`) may be stale.

# Stretch A plan: a third tool, `calculate_egfr`

Status (2026-10-02): **plan agreed; data generated (SA3); evaluation set reviewed and frozen** (`reports/stretch_a/review_decisions.json`, AI review delegated by the user). Pipeline code (SA1, SA2, SA4–SA6) not written.

## 1. Why Stretch A (and not B)

The assignment's Stretch A:

> Add one additional tool and expand your pipeline + evaluation … Annotate a small set of new examples (10–20) for the new tool and show evaluation results.

**What the evidence supports.** On the core task, SFT reliably learned the **in-distribution** policy for the two existing tools, including when *not* to call:

- grounded tool tasks 54/55;
- call-prefix AUROC 1.0, ECE about 0.002;
- after the Q5 relabel, 0/34 fabrication on missing-measurement probes while intact-input calls stay at 33/34 (INTERVIEW_PREP §5.10, §5.14).

**What it does not show.** Whether this is a **transferable tool-use policy** or knowledge of exactly these two tools. The evidence covers only 2 tools, one call per answer, templated questions and one seed.

**Why A.** Stretch A tests exactly that open question, and it reuses the evaluation machinery behind the strongest result: argument grounding, abstention on missing inputs, paired probes.

**Why not B** (`reference_lookup`). It tests retrieval, a different capability. It costs more, and it does not test the main claim.

**Interview line.** "SFT reliably learned the in-distribution policy for our two tools, including when not to call. The open question is whether it learned a transferable tool-use policy or just these two tools. Stretch A tests exactly that: a zero-shot new tool, a few dozen training examples, and an age-removal probe."

## 2. Data facts that constrain the design (measured 2026-10-02)

| | train | val | test |
|---|---|---|---|
| Notes with age and sex extractable (`parsing.extract_age_sex`) | 2000/2000 | 250/250 | 400/400 |
| Table has creatinine | 350 | 40 | 82 |
| … of which the table already has eGFR | 186 | 21 | 51 |
| … of which the table has **no** eGFR (usable for the tool) | **164** | **19** | **31** |

**Data-quality finding.** The table eGFR values are **not** derived from creatinine, age and sex. Against CKD-EPI 2021, the median absolute difference is 26–37 mL/min/1.73m², and only about 9% fall within ±5 (CKD-EPI 2009 gives the same picture). The synthetic eGFR is independent of the inputs. Consequences:

- eGFR tool examples are built **only** from records without a table eGFR; otherwise the tool and the table would contradict each other;
- the finding is reported in `reports/DATA_QUALITY.md`.

## 3. Tool specification (frozen once implemented)

| Item | Decision |
|---|---|
| Signature | `calculate_egfr(creatinine_mg_dl: float, age: int, sex: str) -> float` (as in the assignment) |
| Formula | CKD-EPI 2021, race-free |
| Output | **Integer** mL/min/1.73m² (clinical convention; stated in the schema description) |
| `sex` | Schema: `"male"` or `"female"`. The executor normalises case and simple aliases (M/F, man/woman), following the existing pattern: the schema is stricter than the executor |
| Ranges | creatinine > 0; 18 ≤ age ≤ 120; out of range returns an error string, never an exception |
| Creatinine unit | mg/dL only. A µmol/L input would need two calls, beyond the one-call budget. All selected records are in mg/dL |
| Schema description | "Estimate eGFR (CKD-EPI 2021) from serum creatinine in mg/dL, age in years and sex." |
| Prompt | **v1e** = `system_v1` + "Call calculate_egfr when eGFR must be estimated from serum creatinine (mg/dL), age and sex.", for symmetry with v1, which names the other two tools |

## 4. Arms

| Arm | Model | Prompt | Tools in schema | What it answers |
|---|---|---|---|---|
| **A-zs-v1** | F-s42 epoch 2, no retraining | v1 | 3 | Can the model discover and use a new tool from the schema alone? |
| **A-zs-v1e** | F-s42 epoch 2, no retraining | v1e | 3 | Is a one-line instruction enough? |
| **A-zs-base** | Qwen3-4B base | v1e | 3 | Base reference for the zero-shot arms |
| **A-sft** | F-s42 recipe retrained, plus 52 eGFR rows | v1e | 3 | Is the new tool cheap to learn, and does adding it harm core? |

The zero-shot arms deliberately break train/inference parity (the tool list differs from training). Testing generalisation is their purpose.

## 5. Training data for A-sft (template-generated, rule-checked)

| Component | n | Source |
|---|---|---|
| Positive: one `calculate_egfr` call + final answer | 40 | sampled (seed 42) from the 164 train records without a table eGFR |
| Negative: age removed from the note, then abstain | 6 | further train records from the same pool |
| Negative: sex removed from the note, then abstain | 6 | same |
| **Added** | **52** | 2,000 → 2,052 rows (+2.6%); about 257 optimiser steps vs 250 |
| Everything else | identical to F-s42 | q5_relabeled view, lr 1e-4, 2 epochs, mb1×16, seed 42, NF4, LoRA r16/α32 |

- Existing extractive "what is the eGFR" questions on records that already have a table eGFR stay unchanged. They teach "read the table, don't call".
- Positive : negative ≈ 3:1. Hammer reports a trade-off between call accuracy and irrelevance accuracy, so negatives are kept a minority.
- **Answer templates.** No clinical commentary, because unscored commentary is a hallucination risk (INTERVIEW_PREP §3.2):
  - positive: "Using the patient's serum creatinine of {cr} mg/dL, age {age} and sex ({sex}), the estimated GFR (CKD-EPI 2021) is {egfr} mL/min/1.73m² (KDIGO GFR category {G}, assuming stable kidney function)." Changed from "consistent with CKD stage {G}" during review: CKD staging needs stable function over at least 3 months, and 4 of the 19 notes describe acute or possibly acute kidney injury.
  - negative (age missing): "The serum creatinine ({cr} mg/dL) and sex are documented, but the patient's age is not recorded, so eGFR cannot be calculated." The sex-missing variant is analogous.
- **Rule checks**, not per-row human review:
  - every argument is grounded in the input;
  - no overlap with val or test;
  - mask audit passes;
  - token lengths stay ≤ 2048;
  - the removal leaves no residual age or sex cue.
- **Confound to disclose.** A-sft vs F differs by tool + 52 rows + about 7 steps + one prompt sentence. It is reported as **one extension package**. D-039 showed sentence-level prompt changes flip 0–2 items on SFT models, so the prompt part is a low-risk confound.

## 6. Evaluation set (human-annotated: the assignment's "10–20 examples")

- **19 val records** with creatinine in the table and no table eGFR. Template questions, e.g. "Estimate the patient's eGFR from the available creatinine, age and sex, and state the CKD stage."
- **19 paired E-AGE probes**: the same records with age removed (expected: abstain).
- **Annotation workflow** (same as P1 approval):
  1. A script generates a review sheet with the age, sex and creatinine source spans, the tool-computed eGFR and G stage, and the probe text.
  2. **The user reviews each item** (about 20 min): extraction correct, age removal natural with no residual cues (e.g. "in her sixties"), question sensible.
  3. Approve and freeze with hashes **before any model output exists**.
- Probes for missing sex or missing creatinine are not built (user decision). They are listed as next steps.

## 6a. Review outcome (2026-10-02, delegated AI review; not a clinician)

**Method.**

- Arguments re-extracted from the source note and table with independent regexes.
- eGFR recomputed with an **independent** CKD-EPI 2021 implementation (`scripts/stretch_a_review.py`), not the generator's function.
- KDIGO category and answer text checked.
- Probe diff checked: a pure age removal, with grammatical repair.
- Numeric age residue checked, and value-free age mentions listed.
- Note text checked for written eGFR/CrCl values: none. Only qualitative mentions ("eGFR likely significantly reduced", "CrCl likely <30"), which agree in direction with the computed values.
- Clinical-validity scan.

**Mechanical issues: 0/19.** The independent eGFR equals the stored eGFR for all items.

**Fixes made during review.**

1. Answer template changed to "KDIGO GFR category …, assuming stable kidney function".
2. Age removal now repairs the sentence ("HPI: 58-year-old female with" → "HPI: Female patient with"), including the `**HPI:** female` markdown form.

**Decisions.** All 19 approved, 0 dropped. Tags:

| Tag | Items | Handling |
|---|---|---|
| `non_steady_state` | sa_val_000, 007, 014 (note says "acute or chronic kidney disease"), **sa_val_009 (explicit acute kidney injury)** | Kept. The template states the stability assumption. Reported as a slice. A clinically ideal answer would also say that CKD-EPI is unreliable in AKI |
| `near_cutoff` | sa_val_015 (raw 44.91 → 45, G3a; the G3a/G3b cut-off is 45) | The scorer accepts either neighbouring category when the raw eGFR is within 1 of a cut-off |
| `soft_age_mention_no_value` | sa_val_001 ("attributed it to his age"), 006 and 013 ("given age") | Kept. No value is revealed; "given age" weakly implies an older patient |

**Frozen file hashes** are in `reports/stretch_a/review_decisions.json`. The evaluation set must not change after this point.

## 7. Metrics

The eGFR scorer is a **separate module**; the frozen scorer v2.1 file is not changed.

**eGFR task (19 positives + 19 E-AGE probes)**

| Metric | Definition | Assignment item |
|---|---|---|
| Call rate | positive items with a `calculate_egfr` call | decide to call |
| Tool selection | no BMI or conversion call instead | tool selection |
| Argument grounding | creatinine = table; age = note (int); sex normalised = note | tool arguments |
| Outcome | executed result within ±1 of the reference | — |
| Result reported | the final answer contains the executed value | incorporates the result |
| Clinical context | KDIGO G stage of the executed result stated (G1 ≥90, G2 60–89, G3a 45–59, G3b 30–44, G4 15–29, G5 <15), using the same approach as the C4 check | brief clinical context |
| **E-AGE fabrication** | call with an invented age, or an age or eGFR value asserted in text | Q5 lesson |
| E-AGE intact partner valid call | the paired original succeeds | guards against "no fabrication by never calling" |

**Core regression (full val, paired against F-s42)**

- the five assignment metrics (v1 and v2.1);
- grounded tool tasks on the original two tools;
- P1 (34 pairs): fabrication and partner valid call;
- **eGFR over-call** on the 21 val records whose table already has eGFR, and over-call on all non-tool items.

## 8. Pre-registered success criteria (fixed before any output)

| Claim | Criterion |
|---|---|
| A-sft learned the new tool | end-to-end success ≥ 15/19 on positives; E-AGE fabrication ≤ 2/19 |
| A-sft did not harm core | grounded tool tasks ≥ 53/55; P1 fabrication ≤ 1/34; eGFR over-call ≤ 1/21; paired CIs of the five core metrics include 0 or are positive |
| Zero-shot generalisation (A-zs-*) | **no threshold**; numbers reported as an open question |

With n = 19, the Wilson intervals are wide (e.g. 17/19 → about [0.69, 0.97]). Stretch A results are presented as a **capability demonstration**, not a precise accuracy.

## 9. Work items (not implemented)

| # | Item |
|---|---|
| SA1 | `tools.py`: `calculate_egfr` + executor; `schemas.py`: schema entry; unit tests against published CKD-EPI 2021 reference values |
| SA2 | `configs/prompts/system_v1e.txt`; format and eval configs for the 3-tool arms |
| SA3 | **Done 2026-10-02.** `scripts/stretch_a_data.py` produces `data/stretch_a/` (52 train additions, 19 val positives, 19 E-AGE probes, manifest with sha256) and `reports/stretch_a/review_sheet.md`. Tests: `tests/test_stretch_a_data.py` (CKD-EPI 2021 reference values, KDIGO stages, removals). Val stage mix: G1 2, G2 8, G3a 2, G3b 1, G4 4, G5 2. Hard residue flags: 0. Soft age mentions without a value ("attributed it to his age"): 3 probes, left to the reviewer |
| SA4 | A train view (q5_relabeled + 52 rows); mask audit; length report |
| SA5 | eGFR scorer module + probe classifier, with regression fixtures |
| SA6 | Runs: A-zs-v1, A-zs-v1e, A-zs-base (inference); A-sft (1 training run + val + P1 + eGFR set) |
| SA7 | Report section: results table, core regression, limitations |

**Order.** SA1–SA3, then **the user reviews the 19 + 19 eval items**, then SA4–SA5, then SA6, then SA7.

**GPU estimate.**

| Run | Estimate |
|---|---|
| 3 zero-shot arms | about 15 min each (full val + 38 eGFR items) |
| A-sft | about 45 min training + about 20 min evaluation |
| **Total** | about 1.75 h |

## 10. Limitations and next steps

- n = 19 positives and 19 probes; one seed.
- Only age-removal probes; sex- and creatinine-missing probes are next steps. A creatinine probe would test abstention on a missing argument that was never trained.
- Single-call only. µmol/L creatinine (conversion followed by eGFR) needs multi-call support.
- The synthetic table eGFR is inconsistent with its inputs, so tool and table answers are never mixed.
- Template questions and answers. A real deployment would need varied phrasings.
