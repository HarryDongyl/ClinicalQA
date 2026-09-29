# Data Card

Describes the provided data: files, schema, conventions and known issues, and how the data flows through the
pipeline. Statistics come from [reports/data_analysis.md](../reports/data_analysis.md) (regenerate with `make analyze`).

## 1. Provenance and files

The data was provided with the assignment as `.txt` files containing JSON Lines. `make data` copies them
byte-for-byte to canonical names and records checksums in `data/MANIFEST.json`. `_data/` is never modified.

| Canonical file | Source | Records | SHA-256 (prefix) | Role |
|---|---|---|---|---|
| `data/train.jsonl` | `_data/train_4089.txt` | 2000 | `acda49fd834b` | Training |
| `data/val.jsonl` | `_data/val_3768.txt` | 250 | `33562f1ee02c` | Model selection, hyperparameters, ablations |
| `data/test.jsonl` | `_data/test_4569.txt` | 400 | `15c4956402ae` | Audited test; frozen final model comparison |
| `data/reference.jsonl` | `_data/reference_6223.txt` | 149 | `2095e645a078` | Stretch B lookup table |

The assignment describes pre-generated data, and the audit shows strong synthetic/template characteristics. This repository audit does not independently establish the provenance of every clinical statement.

**Split policy.** Original splits remain unchanged. Test labels have already been inspected for data audit, so this is not an untouched blind test. Do not use test for training or future model/threshold selection. Q3/Q18 found no note duplicates under their implemented checks; that does not rule out semantic or template overlap.

## 2. QA record schema

| Field | Type | Description |
|---|---|---|
| `id` | string | `{split}_{n}`, unique and contiguous (`train_000` ... `train_1999`) |
| `note` | string | Encounter note, 217-377 words. Sections: CC, HPI, PMH, Medications, (Allergies), Vitals, Exam, (Labs), Assessment/Plan. About 40% use markdown bold headers |
| `table` | object | One structured table, see §3 |
| `question` | string | A single question, 6-50 words. **May itself contain evidence** (e.g. weight and height) |
| `answer` | string | Gold answer. Extractive answers average 17 words; the other types average 51-57 words (up to 92) |
| `answer_type` | string | `extractive` \| `numeric_reasoning` \| `tool_call` \| `uncertain` |
| `tool_calls` | array | Present **iff** `answer_type == "tool_call"`. Always exactly one element: `{"tool", "arguments", "result"}` |

Answer-type distribution: train/test are 40% extractive, 20% numeric_reasoning, 25% tool_call, 15% uncertain; val is 40%, 20%, 24.8%, 15.2% because of integer counts.

## 3. Tables

```json
{"type": "labs", "headers": ["Test", "Value", "Unit", "Reference Range"], "rows": [["Creatinine", "3.6", "mg/dL", "0.7-1.3"], ...]}
{"type": "vitals", "headers": ["Vital", "Value", "Unit"], "rows": [["Blood Pressure", "113/78", "mmHg"], ...]}
```

- All cells are strings. Values are numeric, except blood pressure (`"systolic/diastolic"`).
- INR has an empty unit (`""`) in every lab table (232 rows).
- Reference ranges have three forms: `a-b`, `<x`, `>x`. Most panels use fixed ranges; atypical tables can differ (see below).
- Panels (inferred from row names):

| Panel | Rows |
|---|---|
| metabolic | Glucose (fasting), HbA1c, Creatinine, BUN, Sodium, Potassium |
| lipid | Total Cholesterol, LDL, HDL, Triglycerides |
| cbc | WBC, Hemoglobin, Hematocrit, Platelets |
| liver | ALT, AST, Alkaline Phosphatase, Total Bilirubin, Albumin, Total Protein |
| thyroid | TSH, Free T4, Free T3 |
| coagulation | PT, INR, aPTT, D-dimer |
| iron | Serum Iron, TIBC, Ferritin, Transferrin Saturation |
| renal | Creatinine, BUN, eGFR, Uric Acid, Calcium, Phosphorus |
| vitals | Blood Pressure, Heart Rate, Temperature (°F), Respiratory Rate, SpO2 |

- Three atypical tables: `train_278` (vitals + Weight + Height rows), `test_004` (vitals + Weight),
  `train_1905` (mixed labs, "Fasting Glucose", creatinine range 0.6-1.2).
- The note text often repeats some table values. There are no conflicting values; three restate a value at lower
  precision (e.g. "ALT 133" vs 133.3) (Q12).

## 4. Tool calls

Gold format:

```json
"tool_calls": [{"tool": "calculate_bmi", "arguments": {"weight_kg": 104.2, "height_cm": 163.4}, "result": 39.0}]
```

| Tool | Occurrences (train/val/test) | Argument conventions |
|---|---|---|
| `calculate_bmi` | 410 / 47 / 77 | `weight_kg`, `height_cm` in metric, 1 dp. For imperial notes the gold value is the metric equivalent. Weight is `round(lb × 0.4536, 1)`; inch heights are lossy restatements of the gold cm (up to 0.126 cm apart after conversion) |
| `unit_convert` | 90 / 15 / 23 | Only `mg/dL -> mmol/L` (glucose, cholesterol) and `mg/dL -> μmol/L` (creatinine). Result rounded to 2 dp. `to_unit` always uses Greek mu (U+03BC) |

Every gold `result` reproduces exactly with `clinqa.tools` (Q4). Every gold answer states the tool result (Q9).

Where the arguments come from (Q5):

| Source of weight / height | train | val | test |
|---|---|---|---|
| note, metric (kg / cm) | 209 | 25 | 36 |
| note, involves lb and/or in | 73 | 11 | 18 |
| question only | 50 | 4 | 13 |
| **nowhere in the input (ungrounded)** | **78** | **7** | **10** |
| `unit_convert` value from table / note | 57 / 33 | 13 / 2 | 16 / 7 |

## 5. Uncertain examples

What the gold answer says is missing (rule-based, verified against the note except for timestamps):

| Missing | train | val | test |
|---|---|---|---|
| height (weight given) | 65 | 8 | 13 |
| weight (height given) | 51 | 8 | 11 |
| medication dose | 63 | 7 | 10 |
| allergy documentation | 56 | 9 | 14 |
| lab timestamp | 65 | 6 | 12 |

Allergy documentation in notes: no allergy statement 76%, "NKDA" 23%, "Not documented" 2%.

## 6. Reference file (`data/reference.jsonl`)

```json
{"key": "drug_interaction:warfarin:aspirin", "category": "drug_interaction", "value": "Concurrent use of ..."}
```

| Category | Key pattern | Entries |
|---|---|---|
| drug_interaction | `drug_interaction:<drug>:<drug>` (pair order not canonical) | 64 |
| clinical_guideline | `guideline:<topic>:<subtopic>` | 48 |
| clinical_reference | `reference:<topic>:<subtopic>` | 18 |
| contraindication | `contraindication:<drug>:<condition>` | 11 |
| dosing_guideline | `dosing:<drug>:<context>` | 8 |

Drug names are lowercase with `_` for spaces (`calcium_carbonate`, `insulin_glargine`). 630 notes contain at least
one medication pair covered by an interaction key.

## 7. Conventions to respect downstream

- **Units**: µ (U+00B5) and μ (U+03BC) both appear in text; tool arguments use μ. Compare units only after
  `clinqa.tools.normalize_unit`.
- **Numbers**: notes use en/em dashes in ranges (`8.5–10.5`); gold answers use ≈, ×, ÷, − (U+2212). Number
  extraction must handle these (`clinqa.parsing.extract_numbers`).
- **Stated BMI in notes** is not reliable (Q7) and is never used as a target.
- **Distractor numbers** (weight changes, lesion sizes, dosing thresholds) must not be taken as body measurements.

## 8. Known issues (summary)

| Issue | Records | Handling |
|---|---|---|
| Ungrounded BMI tool-call arguments (F1) | 78 / 7 / 10 | R1 keeps all; optional R2 excludes train Q5 candidates with an audit packet. Never filter val/test |
| Corrected Q10 parser false positives (F6) | train_1890, test_306 | Gold numbers were correct. Fix numeric-token backtracking, retain both records |
| Inconsistent stated BMI (F4) | 96 / 10 / 13 | Keep; documented distractor |
| Extractive answers citing ranges not in the input (F7) | 36 / 3 / 7 | Keep; policy P-003 |
| µ vs μ between question/answer and tool args (F8) | 37 / 5 / 14 | Normalised in tools and evaluator |
| Synthetic clinical artefacts (C6) | see Q16 | Keep |

Per-record flags: `reports/quality_flags.jsonl` (fields: `severity`, `check`, `split`, `id`, `detail`).

## 9. Lineage

```text
_data/*.txt  --make data-->  data/*.jsonl + MANIFEST.json  --make analyze-->  reports/{data_analysis.md, data_stats.json, quality_flags.jsonl}
                                   |
                                   +--make views--> data/processed/{raw,q5_filtered}/ (train raw-record selection + manifest + review packet)
                                   |
                                   +--make format (future)--> chat conversations + mask audit
```

## 10. Revision 2 validation and views

The JSONL reader rejects duplicate keys and non-finite numbers. Structural validation runs before downstream feature extraction. Preparation preflights all sources before copying, and analysis/view generation verifies current source and canonical checksums against MANIFEST.json.

`make views` generates both explicit variants; the CLI default is `raw`. R1 is byte-identical to the 2,000-row canonical train file. R2 is a heuristic Q5-filtered 1,922-row selection, not a relabeling or manual certification. Its types are 800/400/422/300 (extractive/numeric/tool/uncertain), so distribution changes are intentional and disclosed. Each manifest records input hashes, code/config hashes, exclusions and counts. Full excluded inputs and tool labels are in q5_review.jsonl for review. Neither variant creates filtered validation/test files.

These views are not SFT-ready chat data. Native template rendering, exact token counts and assistant-only loss-mask verification are future work in PLAN.md.
