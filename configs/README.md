# Configs

Configs are never moved or renamed. Their paths and sha256 are recorded in run manifests, evaluation `run.json` files and the frozen test list. Training configs inherit with `extends:`, and each child sets only what differs. `README.md` ("Add a new experiment") shows how to add one.

| Kind | Pattern | Holds |
|---|---|---|
| Data | `data.yaml`, `analysis.yaml` | Source and target paths of the provided files; data-quality check policy |
| Format | `format_*.yaml` | Tokenizer (pinned revision), tool list, system prompt, train views, output directory of the SFT conversations |
| Train | `train/*.yaml` | Run ID, format config, train view, LoRA and optimiser settings (`train/base.yaml` holds the shared recipe), checkpoint and output directories |
| Eval | `eval_*.yaml` | Format config, model and quantisation, generation budget, batch size, and the `runs:` that may be evaluated with it |
| Prompts | `prompts/system_*.txt` | System prompts |
| Frozen sets and gates | `w3/`, `w4/` | P1 probes, Q5 relabel review, train-fit IDs, gates, eGFR evaluation sets, test list |

## Final models

| Model | Train | Format | Eval (val, test) |
|---|---|---|---|
| **Core final model**: Qwen3.5-4B, relabel view, two tools, prompt v1 | `train/w4_q35_4b_relabel_lr1e4.yaml` (extends `train/w3_q35_4b_filtered_lr1e4.yaml`) | `format_w4_q35_4b.yaml` | `eval_w4_q35_4b.yaml` |
| **Stretch A final model, A-sft2-Q35**: three tools, prompt v1e2, `q5_relabeled_egfr2` view | `train/w4_q35_4b_relabel_egfr2_lr1e4.yaml` | `format_w4_q35_4b_egfr2.yaml` (`max_length` 2560) | `eval_w4_q35_4b_tools3_v1e2.yaml` |
| F′ (Qwen3 comparator; pre-registered core until D-105) | `train/w4_q3_refit_relabel_lr1e4_s42.yaml` (extends `train/w3_relabel_lr1e4_s42.yaml`) | `format_w3.yaml` | `eval_w3_v1.yaml` (val), `eval_w4_v1.yaml` (test) |
| A-sft2 (Qwen3 Stretch A comparator) | `train/w4_q3_relabel_egfr2_lr1e4.yaml` | `format_w4_q3_egfr2.yaml` | `eval_w4_q3_tools3_v1e2.yaml` |

## Comparators and ablations

| Arm | Train | Eval |
|---|---|---|
| R0 (base, prompt v1) | none | `eval_w3_v1.yaml`, `eval_w4_v1.yaml` |
| R0 with prompt v3, or v3 plus four shots | none | `eval_w3_v3.yaml`, `eval_w3_v3_fs4.yaml` (`format_w3_v3.yaml`, `w3/fewshot_v3.json`) |
| C-filtered (Q5 rows removed) | `train/w2_filtered_lr1e4_mb1.yaml` | `eval_w3_v1.yaml`, `eval_w4_v1.yaml` |
| F-s42 (original relabel run; adapter not preserved, D-097) | `train/w3_relabel_lr1e4_s42.yaml` | `eval_w3_v1.yaml` |
| Raw labels (wave 1; P1-RAW control) | `train/raw_lr1e4.yaml` | `eval_core.yaml`, `eval_w4_v1.yaml` |
| Qwen3-8B zero-shot | none | `eval_w3_8b.yaml` (`format_w3_8b.yaml`) |
| Qwen3.5-4B / 9B zero-shot | none | `eval_w3_q35_4b.yaml`, `eval_w3_q35_9b.yaml` |
| Qwen3.5-4B filtered / relabel | `train/w3_q35_4b_filtered_lr1e4.yaml`, `train/w4_q35_4b_relabel_lr1e4.yaml` | `eval_w4_q35_4b.yaml` (`format_w4_q35_4b.yaml`) |

## Stretch A

| Item | Config |
|---|---|
| Zero-shot arms (three tools; prompt v1 or v1e) | `format_w4_q3_tools3_v1{,e}.yaml`, `eval_w4_q3_tools3_v1{,e}.yaml` (Qwen3.5 variants: `*_q35_4b_tools3_*`) |
| First A-sft (52 rows; failed its criteria, D-100) | `train/w4_q3_relabel_egfr_lr1e4.yaml`, `format_w4_q3_egfr.yaml` |
| A-sft2, A-sft2-Q35 | see Final models; the Qwen3.5 zero-shot v1e2 arm uses `eval_w4_q35_4b_tools3_v1e2.yaml` |
| Frozen evaluation sets | `w4/egfr_val.json` (19 positives), `w4/egfr_age_probes.json` (19 age-removed probes) |

## Prompts

| Prompt | Use |
|---|---|
| `system_v1.txt` | Training and evaluation of every core model |
| `system_v1e.txt` | v1 plus one sentence naming `calculate_egfr` (Stretch A zero-shot, first A-sft) |
| `system_v1e2.txt` | v1e plus the KDIGO category table (A-sft2) |
| `system_v2_reference.txt` | Wave-2 inference-only ablation ("use only supplied ranges"); not adopted |
| `system_v3.txt` | Wave-3 strict prompt baseline; not adopted (D-091) |

## Historical

These are kept for reproducibility and are not used by current runs:

- `eval_core.yaml`, `format_core.yaml`, `train/raw_*.yaml`, `train/q5filtered_lr5e5.yaml`, `train_raw.yaml`, `train_grounded.yaml`, `final_eval.yaml` (wave 1);
- `eval_w2_*.yaml`, `format_w2_reference.yaml`, `train/w2_*.yaml`, `wave2_decisions.yaml` (wave 2);
- `train/w3_relabel_lr1e4_s43.yaml` and `_s44.yaml` (seeds, not run);
- `train/w3_8b_filtered_lr1e4.yaml` (not run);
- `smoke.yaml`, `train/gpu_smoke.yaml`, `train/*_smoke.yaml` (smoke tests);
- `w3/approvals.json`, `w3/gates.yaml`, `w4/gates_q35.yaml`, `w4/gate_q35.json` (reviewed gates).

**Frozen; do not edit:** `w3/p1_probes.json`, `w3/q5_relabel_review.jsonl`, `w3/trainfit_ids.json`, `w4/egfr_val.json`, `w4/egfr_age_probes.json`, `w4/final_test.json`, `w4/final_test_q35.json`.
