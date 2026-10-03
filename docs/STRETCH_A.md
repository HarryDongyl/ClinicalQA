# Stretch A: a third tool, `calculate_egfr`

**Result.** The revised model, A-sft2 epoch 2, uses the new tool correctly on 17/19 validation positives and fabricates on 0/19 age-removed probes. It costs about 1 pp on the core task (D-101). It is the Stretch A deliverable; the core final model stays F′, as pre-registered.

- Decisions: D-095, D-097, D-100, D-101 in [DECISIONS.md](DECISIONS.md).
- Archived plan: [history/STRETCH_A_PLAN.md](history/STRETCH_A_PLAN.md).
- Scored outputs: `reports/w4/r2/` (first A-sft and zero-shot arms), `reports/w4/r2b/` (A-sft2), `reports/w4/test/` (core test).

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
| **A-sft2** | F recipe from base | **v1e2** (v1e plus the KDIGO table) | 2,000 core + 200 eGFR rows |

## Results (validation; 19 positives + 19 age-removed probes)

| Arm | End-to-end | KDIGO category | Probe fabrication | Probe abstains correctly | Over-call where the table has eGFR |
|---|---|---|---|---|---|
| A-zs-base | 0/19 | 0/19 | 13/19 | 6/19 | 0/21 |
| A-zs-v1 | 3/19 | 3/19 | 5/19 | 14/19 | 0/21 |
| A-zs-v1e | 3/19 | 3/19 | 3/19 | 16/19 | 0/21 |
| A-sft ep2 | 9/19 | 9/19 | 16/19 | 3/19 | 0/21 |
| **A-sft2 ep2** | **17/19** | **17/19** | **0/19** | **19/19** | 0/21 |

All arms that call select the right tool with grounded arguments on 19/19 positives. Wilson intervals at n = 19 are wide (17/19 ≈ [0.69, 0.97]), so this is a capability demonstration.

**Core regression** (v2.1):

| | F′ val | A-sft2 val | F′ test | A-sft2 test |
|---|---|---|---|---|
| Grounded tool tasks | 54/55 | 51/55 (2 scorer false fails) | 89/90 | 87/90 (2 scorer false fails) |
| P1 fabrication | 0/34 | 1/34 (val_038, real; intended behaviour 33/34) | — | — |
| Natural Q5 fabrication | 0/7 | 0/7 | 0/10 | 0/10 |
| v2.1 macro | 95.0 | 92.6 | 97.5 | 96.2 |

On test, A-sft2 − F′ = −1.28 pp [−2.66, −0.06]. The two models differ in prompt and tool list, so this compares two deployable configurations, not an ablation.

## What went wrong in the first A-sft, and the fix (D-100)

1. **Age fabrication was a threshold shift, not lost discrimination.** On probes the call probability sat at 0.27–0.82 (mostly near 0.5), while positives-vs-probes AUROC stayed 0.983; zero-shot F′ had AUROC 1.0. The model invented ages, mostly 65.
   - Causes: 40 positives against 6 age negatives, and question templates presupposing "age and sex".
   - Fix: 60 age negatives (40 paired) and neutral templates. Probe fabrication fell from 16/19 to 0/19.
   - This mirrors the core finding: tool SFT with too few no-call examples teaches fabrication.
2. **The KDIGO mapping was not learned.** Errors run in both directions and are not all adjacent: sa_val_003 (eGFR 67, G2) was written as G3b. G3a and G5 had 3 and 1 examples.
   - Fix: a KDIGO table in the prompt (v1e2), stage-balanced positives, and answers that state the range before the category. Category accuracy rose from 9/19 to 17/19. Both remaining failures are the two G5 cases: eGFR 12 and 13 are written as "in the 15–29 range (G4)" despite the table in the prompt, which motivates a deterministic category mapping.
3. **Core regression overlaps the refit differences and scorer artefacts.** Of the first A-sft's four tool failures, three also occur in F or F′. Of A-sft2's, two per split are v2.1 reader false fails ("reduced renal function" read as creatinine low; S-04).

## Pre-registered criteria and decision

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

The pre-registered decision is therefore a fail. The final core model stays F′ (two tools), and A-sft2 is delivered as the three-tool model with its core cost disclosed. No third revision was run.

## Limitations

- The template-generated training rows and the probes share the same age-removal generator, so probe results are in-distribution. The validation items were inspected during the A-sft diagnosis.
- Age-removed negatives are 65% G1/G2, against 48% of positives.
- There is one seed. A-sft2 has no zero-shot v1e2 comparator, and the revision changed data, prompt, templates and steps together, so the gain is attributed to the package, not to one component.
- Sex-removed negatives are trained but not evaluated. Missing creatinine, µmol/L units and conflicting inputs are untested.
- The near-cut-off item sa_val_015 (raw eGFR 44.91) accepts either adjacent category by scorer policy.
- Test has no annotated eGFR items, so Stretch A is reported on validation only.
- A µmol/L creatinine would need two calls (convert, then estimate), which the one-call budget does not allow.
