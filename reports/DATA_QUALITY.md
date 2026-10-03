# Data quality analysis

This is the assignment's deliverable 3.

**Sources.**

- Generated statistics and all 19 checks: [data_analysis.md](data_analysis.md) (`make analyze`).
- Every flag: [quality_flags.jsonl](quality_flags.jsonl).
- The formatted SFT examples: [formatted_examples.md](formatted_examples.md) (`make audit-masks`).

## Provenance

The provided files in `_data/` are copied byte for byte to `data/{train,val,test,reference}.jsonl` by `make data`, which records their sha256 in `data/MANIFEST.json`. They are verified before every use, and the originals are never modified.

Training uses train-only views in `data/processed/`. Each view records its policy, the IDs it changes and its own hash. The views are:

| View | Rows | What it is |
|---|---|---|
| `raw` | 2,000 | The training split unchanged |
| `q5_filtered` | 1,922 | The 78 Q5 records removed |
| `q5_relabeled` | 2,000 | The 78 Q5 records relabelled as reviewed abstentions |
| `q5_relabeled_egfr` | 2,052 | `q5_relabeled` plus 52 template-generated Stretch A rows |
| `q5_relabeled_egfr2` | 2,200 | `q5_relabeled` plus 200 template-generated Stretch A rows |

Validation and test are never edited.

## Formatted examples

[w3/relabel_mask_audit/formatted_examples.md](w3/relabel_mask_audit/formatted_examples.md) shows six examples as rendered for the final training view (`q5_relabeled`): train_000 (extractive), train_004 (numeric), train_003 (uncertain), train_046 (BMI), train_015 (unit conversion) and train_006 (imperial BMI). Each shows the exact rendered model input (native Qwen chat template, system prompt v1, tool schemas, generation prompt) and the exact supervised continuation. The wave-1 rendering of the same six examples is [formatted_examples.md](formatted_examples.md).

A reviewed Q5 relabel, before and after (`train_009`; the input documents no weight or height):

- Original gold: a `calculate_bmi` call with 88.8 kg and 169.1 cm, then "the patient's BMI is 31.1 kg/m², classifying her as obese …".
- Reviewed target: "Weight and height are not documented, so BMI cannot be calculated from the available information.", with no call. The review was AI-assisted grounding review with a recorded rationale (`configs/w3/q5_relabel_review.jsonl`), not clinician adjudication.

**Representation:**

- The user message is `## Encounter note`, then `## Table (<type>)` as a Markdown table (pipes escaped, empty units shown as `(no unit)`), then `## Question`.
- Tool calls use the native `<tool_call>{"name", "arguments"}</tool_call>` turn, followed by a `tool` message carrying the real executor result, then the final answer.
- Loss is applied to assistant tokens only. The mask audit passed on 2,000/2,000 conversations; 5.5% of tokens are supervised.

## Statistics

| | train | val | test |
|---|---|---|---|
| Records | 2,000 | 250 | 400 |
| Extractive / numeric / tool / uncertain | 40.0 / 20.0 / 25.0 / 15.0% | 40.0 / 20.0 / 24.8 / 15.2% | 40.0 / 20.0 / 25.0 / 15.0% |
| Mean note length (words) | 274.5 | 273.7 | 274.9 |
| Records with a tool call | 500 (25.0%) | 62 (24.8%) | 100 (25.0%) |
| `calculate_bmi` / `unit_convert` calls | 410 / 90 | 47 / 15 | 77 / 23 |
| Labs / vitals tables | 1,393 / 607 | 184 / 66 | 288 / 112 |

- Every tool record has exactly one call.
- The provided splits are already stratified by answer type. The formatter keeps every record, so the distribution is preserved by construction.
- Only three unit conversions occur (creatinine, cholesterol, glucose).
- Tokenised conversations (Qwen3): median 1,375 tokens, maximum 1,614, against a 2,048 limit.

## Findings and handling

Checks produce candidates, not proof that a gold label is wrong. Each check has an explicit keep, flag or exclude policy.

### Findings that changed training or evaluation

1. **Ungrounded BMI tool labels (Q5).**
   - 95 of 534 `calculate_bmi` gold calls (train 78, val 7, test 10) use weight and height that appear nowhere in the note, table or question.
   - A further 67 BMI records state the measurements only in the question. They are grounded, because the question is input.
   - **Handling.** The 78 training records were relabelled as abstentions after review (`configs/w3/q5_relabel_review.jsonl`). This deliberately changes the training type mix (422 tool, 378 uncertain). Validation and test keep their labels and are evaluated as expected abstentions.
   - Training on the raw labels taught argument invention: 32/34 on the P1 probes. Removing them moved the invention into prose: 25/34 on Qwen3 and 29/34 on Qwen3.5. Relabelling removed it: 0/34; on test, 0/10 against 6/10 for filter-only.
2. **A coverage gap after filtering.** Among BMI training questions:

   | Type | Count |
   |---|---|
   | Tool call with both measurements | 332 |
   | Abstention with height only | 43 |
   | Abstention with weight only | 65 |
   | **Abstention with both missing** | **1** |

   This distributional gap, found only from model behaviour, is why filtering was not enough.
3. **An allergy shortcut.** Train has 51 questions that mention allergy explicitly and 5 implicit ones, and every "is it safe" question is uncertain. Validation accuracy is 36/36 on explicit questions and 8/18 on implicit ones. This is reported as a slice; implicit-question augmentation is a next step.
4. **Imperial rounding (Q6).**
   - 51 inch heights do not round-trip to the gold centimetres.
   - Exact conversion changes the BMI by ±0.1 in 20 of 138 imperial records.
   - **Handling.** Argument tolerance ±0.05, and outcome-based BMI scoring (±0.15). The same drift explains the remaining tool errors: the wave-1 test errors, val_104 and val_052, and test_285.
5. **The table eGFR is an input-independent synthetic value.**
   - Its Spearman correlation with creatinine is 0.008 across 258 records. Against CKD-EPI 2021, CKD-EPI 2009 (with or without the race factor) and MDRD, the median absolute difference is 29–30 and only 8–13% fall within ±5 (`tests/test_egfr_formula.py`).
   - It appears in 64 core questions or golds, so it is not edited: answers grounded in the table are correct by the task's contract, though clinically inconsistent.
   - **Handling.** Stretch A uses only records without a table eGFR (train 164, val 19, test 31).

### Task design, kept as is

6. **"Most abnormal" has no defined scale.** In 15 ambiguous items, the gold uses relative deviation in 13; only 5 have a common unit, and val_202's gold is wrong under both scales. **Handling.** The scorer accepts either scale when units agree. The sample is too small to justify relabelling.
7. **External reference ranges (Q11).** 46 extractive golds cite a standard range absent from the input. This is task design: the standard range is required when the input gives none. A "use only supplied ranges" prompt contradicted 36 correct training golds and changed 0–2 items, so it was not adopted.
8. **Stated BMI disagrees with the measurements (Q7)** in 119 notes. The gold uses the computed value, and the records are kept.

### Gold errors found later

9. **Clinical-context errors in gold.** Three training golds call BMI 18.7–18.9 "underweight". The SFT models copy the same error on val_125, their only clinical-context miss (54/55).
10. **A test gold asserts a measurement the input lacks.** test_020 (uncertain) states a height of 162.8 cm that is not in the note. Test labels are left unchanged.
11. **Numeric gold versus input-derived keys.** Of 50 validation numeric items, 34 agree, 0 conflict and 16 cannot be fully checked. The known scorer false passes all sit in the agreeing stratum, so a correct gold does not guarantee a correct score.

### Checked and clean, or noise

- No overlap between splits (Q3, Q18).
- Gold tool results reproduce with our tools (Q4), and answer types are consistent with the inputs (Q8).
- The initial arithmetic flags (Q10) were parser false positives. The parser was fixed with regression tests; 0 issues remain.
- µ vs μ spelling (Q13; 56 records) is normalised.
- 219 clinically implausible synthetic values (Q16) are kept and documented.
- Test labels were read during this audit, so test is described as audited but unused for training, not as blind.

### Limitations of the data

- The data is synthetic and template-generated, with four answer types and one call per answer.
- Explanatory commentary in golds is unscored, and models inherit its style, including the gold's errors.
- Validation is small (250), so per-type minimum detectable effects are about 14–23 pp.
