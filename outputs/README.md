# Outputs

Every training run and every evaluation is kept here; nothing is deleted, so each claim in the report can be traced to its outputs. The report uses the groups below.

**Layout.**

- `outputs/<run>/`: a training run (`manifest.json`, `run_contract.json`, `data_policy.json`, `train_log.jsonl`).
- `outputs/<label>/<split>/`: an evaluation (`trajectories.jsonl`, `scored.jsonl` and `metrics.json` under v1, `run.json` with protocol and hashes, `error_analysis.md`). The split is `val`, `train` (train-fit), `p1_probes`, `egfr_val`, `egfr_age_probes` or `test`.
- Epoch checkpoints are `<run>_step000NNN`. Epoch 2 is the primary endpoint.
- v2.1 scores and analyses live in `reports/`.

## Final models

| Directory | What it is |
|---|---|
| `w4_q35_4b_relabel_lr1e4`, `…_step000125`, `…_step000250` | **Core final model**: Qwen3.5-4B on the reviewed relabel view; val, P1, train-fit |
| `q35_relabel_test` | Its frozen test run |
| `w4_q35_4b_relabel_egfr2_lr1e4`, `…_step000138`, `…_step000276` | **Stretch A final model** (A-sft2-Q35); val, P1, eGFR sets |
| `asft2_q35_test` | Its frozen test run (the second test use) |

## Controls that support the main findings

| Directory | Role |
|---|---|
| `w3_c_filtered_s42`, `c_filtered_test` | Qwen3, Q5 rows removed: fabricates in prose (P1 25/34; test 6/10) |
| `w3_relabel_lr1e4_s42_*`, `w4_q3_refit_relabel_lr1e4_s42*`, `f_refit_test` | Qwen3 relabel: F (adapter lost) and its refit F′; the pre-registered core model until D-105 |
| `w3_q35_4b_filtered_lr1e4*`, `w4_q35_4b_filtered_lr1e4_step*` | Qwen3.5, Q5 rows removed (P1 29/34), and its parity regeneration |
| `raw_lr1e4*`, `w4_raw_lr1e4_step000250` | Raw labels: the wave-1 model and P1-RAW, the label-only control (32/34) |
| `w3_r0_v1`, `r0_v1_test`, `w3_r0_q35_4b` | Base models with the same prompt (zero-shot) |
| `w4_q3_relabel_egfr_lr1e4*`, `w4_q3_relabel_egfr2_lr1e4*`, `asft2_test` | Qwen3 Stretch A: the first A-sft (failed) and A-sft2 |
| `w4_sa_zs_*` | Stretch A zero-shot arms (new tool in the schema, no training) |

## Alternatives that were ruled out

| Directory | Question answered |
|---|---|
| `w3_r0_v3`, `w3_r0_v3_fs4` | Can prompting replace SFT? No (grounded tool 9/55 and 29/55) |
| `w3_r0_8b`, `w3_r0_q35_9b` | Does a larger base fix grounding? No (8B fabricates 28/34) |
| `w2_filtered_lr*_mb1*`, `w2_filtered_lr1e4_mb4` | Learning rate (noise) and micro-batch 4 (slower) |
| `w2p_v1_*`, `w2p_v2_*` | An inference-only prompt change (0–2 items) |

## Early runs (wave 1)

| Directory | Role |
|---|---|
| `base`, `base_test`, `raw_lr5e5*`, `q5filtered_lr5e5*`, `raw_lr1e4_test` | The first end-to-end baseline and its single wave-1 test evaluation |
| `final_selection.json`, `runpod_preflight.json` | The wave-1 selection record and the pod preflight |
| `comparisons/` | Wave-1 and wave-2 paired comparison reports. `clinqa.evaluate compare` writes new ones to `outputs/` |
