# Stretch A: a third tool, `calculate_egfr`

**Result.** The Stretch A deliverable is **A-sft2-Q35** (D-106): the revised Stretch A data and prompt on the Qwen3.5 relabel recipe.

- It answers 19/19 validation positives end to end and fabricates on 0/19 age-removed probes.
- On the core test it is −0.2 pp against its own core model (Qwen3.5 relabel).
- Its validation core gate fails by one discordant item (6 against ≤ 5).

The same revision on Qwen3 (A-sft2) reached 17/19 at a −1.3 pp core cost.

- Decisions: D-095, D-097, D-100, D-101, D-105, D-106 in [DECISIONS.md](DECISIONS.md).
- Archived plan: [history/STRETCH_A_PLAN.md](history/STRETCH_A_PLAN.md).
- Scored outputs:
  - `reports/w4/r2/`: first A-sft and the zero-shot arms;
  - `reports/w4/r2b/`: A-sft2 (Qwen3);
  - `reports/w4/r2c/`: Qwen3.5 zero-shot v1e2 and A-sft2-Q35;
  - `reports/w4/test/` and `reports/w4/test_q35/`: core test.

## Why A and not B

SFT learned the in-distribution policy for the two core tools, including when not to call. Stretch A tests whether that is a transferable tool-use policy, using the same grounding, abstention and paired-probe machinery as the core results. Stretch B (reference lookup) tests retrieval, a different capability.

## Tool

| Item | Choice |
|---|---|
| Signature | `calculate_egfr(creatinine_mg_dl, age, sex)` in `src/clinqa/tools.py`; schema in `src/clinqa/schemas.py` |
| Formula | CKD-EPI 2021, race-free: `142 × min(Scr/κ,1)^α × max(Scr/κ,1)^−1.200 × 0.9938^age × 1.012 [female]`, with κ 0.7/0.9 and α −0.241/−0.302. Checked against the NKF equation; four independent implementations agree on 21,258 inputs (`tests/test_egfr_formula.py`) |
| Output | Integer mL/min/1.73m², the clinical convention. The assignment's signature says `-> float`; the integer is a disclosed deviation |
| Inputs | Creatinine in mg/dL only; age an integer from 18 to 120; sex `male`/`female` (simple aliases accepted). Out-of-range inputs return an error string |

## Data

- **Usable records.** The table's own eGFR is an input-independent synthetic value: Spearman correlation with creatinine is 0.008, and against CKD-EPI 2021, CKD-EPI 2009 or MDRD only 8–13% fall within ±5. It is referenced by 64 core questions or golds, so it is not edited. eGFR examples use only records whose table has creatinine and **no** eGFR: train 164, val 19, test 31.
- **Evaluation**, frozen before any output: `configs/w4/egfr_val.json` (19 val positives) and `configs/w4/egfr_age_probes.json` (the same 19 with age removed; expected behaviour is to abstain). Each item was independently re-extracted and recomputed, with 0/19 mechanical issues (AI review, not a clinician; `reports/stretch_a/`). Answers say "KDIGO GFR category …, assuming stable kidney function", because four notes describe possible acute kidney injury.
- **Training rows**:
  - first A-sft: `data/stretch_a/`, 52 rows (40 positives, 6 age-removed, 6 sex-removed);
  - A-sft2: `data/stretch_a_v2/`, 200 rows. 120 stage-balanced positives; 60 age-removed negatives, 40 of them paired with a positive's own note; 20 sex-removed negatives. Eight question templates; answers state the KDIGO range before the category.

## Arms

| Arm | Model | Prompt | Training |
|---|---|---|---|
| A-zs-base | Qwen3-4B base | v1e | none |
| A-zs-v1 / A-zs-v1e | F′ (two-tool SFT) | v1 / v1e | none; the new tool appears only in the schema |
| A-sft | F recipe from base | v1e | 2,000 core + 52 eGFR rows |
| A-sft2 | F recipe from base (Qwen3) | **v1e2** (v1e plus the KDIGO table) | 2,000 core + 200 eGFR rows |
| A-zs-v1e2-Q35 | Qwen3.5 relabel (two-tool SFT) | v1e2 | none |
| **A-sft2-Q35** | Qwen3.5 relabel recipe from base | v1e2 | 2,000 core + 200 eGFR rows (same as A-sft2) |

## Results (validation; 19 positives + 19 age-removed probes)

| Arm | End-to-end | KDIGO category | Probe fabrication | Probe abstains correctly | Over-call where the table has eGFR |
|---|---|---|---|---|---|
| A-zs-base | 0/19 | 0/19 | 13/19 | 6/19 | 0/21 |
| A-zs-v1 | 3/19 | 3/19 | 5/19 | 14/19 | 0/21 |
| A-zs-v1e | 3/19 | 3/19 | 3/19 | 16/19 | 0/21 |
| A-sft ep2 | 9/19 | 9/19 | 16/19 | 3/19 | 0/21 |
| A-sft2 ep2 (Qwen3) | 17/19 | 17/19 | 0/19 | 19/19 | 0/21 |
| A-zs-v1e2-Q35 | 7/19 | 7/19 | 1/19 | 18/19 | 0/21 |
| **A-sft2-Q35 ep2** | **19/19** | **19/19** | **0/19** | **19/19** | 0/21 |

All arms that call select the right tool with grounded arguments on 19/19 positives. Wilson intervals at n = 19 are wide (19/19 ≈ [0.83, 1.00]; 17/19 ≈ [0.69, 0.97]), so this is a capability demonstration. A-sft2-Q35 vs A-sft2 on eGFR is 19 vs 17 (sign test p = 0.5).

**Core regression** (v2.1):

| | Qwen3.5 relabel val | **A-sft2-Q35 val** | F′ val | A-sft2 val |
|---|---|---|---|---|
| Grounded tool tasks /55 | 54 | 52 | 54 | 51 (2 scorer false fails) |
| P1 fabrication /34 (intact partners) | 0 (33) | 0 (31) | 0 (34) | 1 (33) |
| Natural Q5 abstention /7 | 7 | 7 | 7 | 7 |
| v2.1 macro | 96.9 | 96.1 | 95.0 | 92.6 |

| | Qwen3.5 relabel test | **A-sft2-Q35 test** | F′ test | A-sft2 test |
|---|---|---|---|---|
| Grounded tool tasks /90 | 88 | 87 (3 S-04 false fails) | 89 | 87 (2 false fails) |
| Natural Q5 fabrication /10 | 0 | 0 | 0 | 0 |
| v2.1 macro | 97.4 | 97.2 | 97.5 | 96.2 |

**The extension's core cost** (each Stretch model minus its own core model):

| | Qwen3.5 | Qwen3 |
|---|---|---|
| Validation | −0.8 pp | −2.4 pp |
| Test | −0.20 pp [−1.76, +1.35] | −1.28 pp [−2.66, −0.06] |

**Backbone comparison on identical Stretch data and prompt** (validation): A-sft2-Q35 − A-sft2 = +3.56 pp core macro [+0.25, +7.10] (10 vs 3 discordant, sign p = 0.092); on test, +0.99 [−1.04, +3.04].

The models within each comparison differ in prompt and tool list from their core counterparts, so these are configuration comparisons. A-sft2-Q35's second test run is the second use of test (D-105).

## What went wrong in the first A-sft, and the fix (D-100)

1. **Age fabrication coexisted with high call-prefix discrimination.** On probes the call probability sat at 0.27–0.82 (mostly near 0.5), while positives-vs-probes AUROC stayed 0.983; zero-shot F′ had AUROC 1.0. The model invented ages, mostly 65.
   - Plausible contributors: 40 positives against 6 age negatives, and question templates presupposing "age and sex". These factors were not isolated. No explicit 0.5 call threshold was deployed.
   - The bundled revision added 60 age negatives (40 paired), neutral templates and other prompt/target changes. Probe fabrication fell from 16/19 to 0/19.
   - This is consistent with the Core evidence that explicit missing-input supervision helps; the causal contribution of negative-example count alone is not identified.
2. **The KDIGO mapping was not learned.** Errors run in both directions and are not all adjacent: sa_val_003 (eGFR 67, G2) was written as G3b. G3a and G5 had 3 and 1 examples.
   - Fix: a KDIGO table in the prompt (v1e2), stage-balanced positives, and answers that state the range before the category. Category accuracy rose from 9/19 to 17/19. Both remaining failures are the two G5 cases: eGFR 12 and 13 are written as "in the 15–29 range (G4)" despite the table in the prompt, which motivates a deterministic category mapping.
3. **Core regression overlaps the refit differences and scorer artefacts.** Of the first A-sft's four tool failures, three also occur in F or F′. Of A-sft2's, two per split are v2.1 reader false fails ("reduced renal function" read as creatinine low; S-04).

## Pre-registered criteria and decisions

### A-sft2 on Qwen3 (D-100)

The criteria were revised for A-sft2 after the first failure (D-100); the original criteria and result are still reported. Outcome:

| Criterion | Status |
|---|---|
| End-to-end ≥ 15/19 | Met |
| Probe fabrication ≤ 2/19 | Met |
| P1 ≤ 1/34 | Met |
| Over-call ≤ 1/21 | Met |
| Natural Q5 ≥ 6/7 | Met |
| Grounded tool ≥ 52/55 | **Not met** (51; 53 after correcting the two scorer false fails) |
| Discordance with F′ ≤ 5, over all 250 core val items | **Not met**: 9 (A-sft2 better on 2, F′ on 7); still 7 after correcting the two false fails. Tool-only discordance is 3 |

The pre-registered decision was a fail. At the time, F′ stayed the core model and A-sft2 was the Stretch A model (D-101).

### A-sft2-Q35 on Qwen3.5 (D-105, fixed before any output)

The control is Qwen3.5 relabel epoch 2.

| Criterion | Status |
|---|---|
| End-to-end ≥ 15/19 | Met (19/19) |
| Probe fabrication ≤ 2/19 | Met (0/19) |
| P1 ≤ 1/34 | Met (0/34) |
| Over-call ≤ 1/21 | Met (0/21) |
| Natural Q5 ≥ 6/7 | Met (7/7) |
| Grounded tool ≥ 52/55 | Met (52) |
| Discordance with the control ≤ 5, over all 250 core val items | **Not met: 6** (A-sft2-Q35 better on 2, control on 4) |

- The control's four wins are real A-sft2-Q35 errors: imperial drift on val_052 and val_105; a wrong entity on val_069; fabricated weight/height on val_204, where only BMI 39.3 is documented. The extension invents 100.5 kg and 1.75 m and reports BMI 32.8, outside the Q5/P1 probe cohorts.
- One of A-sft2-Q35's two wins is a v2.1 false fail of the control (val_062). Correcting it would give 5, but the pre-registered result stays a fail.
- Per D-105, the eGFR criteria are met, so **A-sft2-Q35 is the Stretch A deliverable**, and the core gate failure is reported.

## Limitations

- The template-generated training rows and the probes share the same age-removal generator, so probe results are in-distribution. The validation items were inspected during the A-sft diagnosis.
- Age-removed negatives are 65% G1/G2, against 48% of positives.
- There is one seed. The revision changed data, prompt, templates and steps together, so the gain is attributed to the package. The Qwen3.5 zero-shot v1e2 arm scores 7/19 under the same offered prompt/schema as the 19/19 trained extension. This does not isolate the category table or individual training-data changes.
- The switch of the final models to Qwen3.5 came after the first test run (D-105).
- A-sft2-Q35 trained on an RTX 4090, and its control on an A100.
- Sex-removed negatives are trained but not evaluated. Missing creatinine, µmol/L units and conflicting inputs are untested.
- The near-cut-off item sa_val_015 (raw eGFR 44.91) accepts either adjacent category by scorer policy.
- Test has no annotated eGFR items, so Stretch A is reported on validation only.
- A µmol/L creatinine would need two calls (convert, then estimate), which the one-call budget does not allow.
