> **Archived 2026-10-04.** Superseded by `reports/REPORT.md`. Kept unchanged below as a historical record.

# Revision 2 — changes and verification

Date: 2026-09-28. Scope: update the implementation plan and repair current data processing. No GPU training or model evaluation was performed. No git commit or push was made.

## Changes

- Fixed Q10 numeric-token backtracking at x/× suffixes, including signed values. Kept train_1890 and test_306; their gold arithmetic was not the defect.
- Added regression tests for complete decimals, multiplier suffixes, percentage expressions, negative differences, actual incorrect arithmetic and adjacent formulas.
- JSONL loading now rejects duplicate object keys, non-standard/non-finite numbers and overflow, with file/line context.
- Structural validation now precedes feature extraction: record fields, IDs, table dimensions, tool presence, argument keys/types and finite values/results. Empty INR units remain valid.
- Data preparation preflights all sources before copying any split. Analysis and view generation verify current source and canonical file hashes against the manifest.
- Added explicit train-only raw and Q5-filtered data views. Each includes input/code/config hashes, before/after counts, exclusions and full review packets. The default CLI policy is raw.
- Updated generated report language to avoid treating a heuristic flag as proof of truncation or an overlap check as proof of no leakage.
- Updated PLAN, DECISIONS, DATA, FINDINGS and README. Native chat formatting and model training remain future work. RL is explicitly out of current scope.

## Actual checks

- Full suite: **88 passed in 3.53 seconds** (previously 61 tests).
- Q10: **0 flags**, 133 explicit equations checked, 389/389 parsed “by D” amounts verified. This is heuristic coverage, not a proof of universal clinical/arithmetic correctness.
- Q4: all **662** supplied gold tool results reproduce.
- Q5 candidates remain **78 train / 7 val / 10 test**.
- Raw train view: **2,000** rows and byte-identical to the original canonical train file.
- Filtered train view: **1,922** rows — 800 extractive, 400 numeric_reasoning, 422 tool_call, 300 uncertain. train_1890 is retained.
- No filtered val/test files are created. All original and canonical file SHA-256 values remain unchanged, including the reference file.
- Generated data reports were reproduced byte-for-byte with a different PYTHONHASHSEED; the data-view idempotence check is also covered by tests.

## Environment limitation

The copied .venv/bin/python points to a missing interpreter and uv is not on the current PATH. This revision used the available **Python 3.13** and the existing copied **pytest 8.3.4 / PyYAML 6.0.2** packages through an explicit PYTHONPATH for CPU-only verification:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.venv/lib/python3.11/site-packages python3 -m pytest -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.venv/lib/python3.11/site-packages python3 -m clinqa.analyze --config configs/analysis.yaml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.venv/lib/python3.11/site-packages python3 -m clinqa.data_views --variant raw
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.venv/lib/python3.11/site-packages python3 -m clinqa.data_views --variant q5_filtered
```

These are the actual local verification commands, not the intended GPU setup. Recreate the target Python 3.11/uv environment on the GPU host, pin the training stack after a smoke test, and run the suite there. No CUDA/transformers/TRL compatibility claim is made by this revision.

## Review and data boundaries

Q5 filtering is an explicit heuristic experiment. The generated manifest states `manual_review_complete: false`, and q5_review.jsonl includes full input and gold calls. This revision does not claim a fresh human adjudication of all 78 exclusions. The unchanged raw view remains the Core baseline.

Test labels were already inspected for data audit. They are not used for training or model selection; do not describe the test as untouched blind data. Freeze the future model comparison before generating final test outputs.

## Next step

Implement the model-specific formatter and assistant loss-mask audit, then the real tool runner and tiny SFT save/reload smoke. Complete full SFT evaluation before considering further optimization. RL is unnecessary for the current assignment; reassess only if reliable execution rewards and measured residual SFT failures justify a separate experiment.
