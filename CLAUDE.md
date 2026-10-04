# CLAUDE.md

Guidance for AI coding assistants working in this repository. Start with `README.md`, then `reports/REPORT.md`.

## What this is

This repository applies QLoRA SFT to 4B instruction models (Qwen3-4B-Instruct-2507 and Qwen3.5-4B) for clinical QA over one note and one table, with tool calls (`unit_convert`, `calculate_bmi`, and in Stretch A `calculate_egfr`) and abstention when information is missing.

The final models are on Qwen3.5 (D-106): `w4_q35_4b_relabel_lr1e4` (core, checkpoint 250) and `w4_q35_4b_relabel_egfr2_lr1e4` (Stretch A, checkpoint 276). The Qwen3 models F′ and A-sft2 are reported comparators.

## Layout

| Path | Contents |
|---|---|
| `src/clinqa/` | Library: `formatting.py`, `train.py`, `evaluate.py` (rollout, generate, score), `tools.py`, `schemas.py`, `data_views.py`, `training_data.py`, `metrics.py` (v1 scorer), `scorer_v2.py` (v2.1 scorer, frozen) |
| `scripts/` | Runners and analysis; index in `scripts/README.md`; superseded scripts in `scripts/legacy/` |
| `configs/` | Format, training (`extends:`), evaluation and prompt configs; index in `configs/README.md` |
| `data/`, `_data/` | `_data/` holds the provided originals; `data/` holds checksum-verified copies plus train views |
| `outputs/` | `outputs/<run>/` (training) and `outputs/<label>/<split>/` (evaluation) |
| `reports/`, `docs/` | Results and documentation; `docs/history/` is archived |

## Commands

```bash
make setup-cpu        # analysis environment (CPU); make setup on GPU pods (train + qwen35 extras)
make test             # unit tests; must pass before any commit
make data analyze views-all
make w4-score OUT=reports/new_dir  # CPU: final Q35 validation evidence
make help
```

| Where | Commands |
|---|---|
| GPU only | `make train`, `make eval`, `scripts/run_w3_round.sh`, `scripts/run_w4_round.sh`, `scripts/w3_epochs.py generate`, `scripts/w4_test.py run` |
| CPU | Everything else, including all scoring (`scripts/score_v21_val.py`, `scripts/score_v2.py`, `scripts/w3_analyze.py`, `scripts/stretch_a_score.py`) |

Run commands from the repository root. Never run a bare `uv sync`, because it drops the `train` extra; use `make setup`. Shell loops over label lists need bash, because zsh does not word-split `$VAR`.

## Rules that must not be broken

1. **No selection on test.** Train is used for training and val for every decision. Test runs only through `scripts/w4_test.py` from a list frozen and committed beforehand (`configs/w4/final_test.json`). Test has been used; do not run it again without `--rerun-reason`, which is disclosed.
2. **Never modify the provided data**: `_data/`, `data/{train,val,test,reference}.jsonl` (sha256 in `data/MANIFEST.json`). Training changes happen only through train views in `data/processed/`, with a recorded policy.
3. **Frozen files:**
   - `src/clinqa/scorer_v2.py` (v2.1; sha256 in `reports/scorer_v2/scorer_v2.1.sha256`);
   - `configs/w3/p1_probes.json`, `configs/w3/q5_relabel_review.jsonl`, `configs/w3/trainfit_ids.json`;
   - `configs/w4/egfr_val.json`, `configs/w4/egfr_age_probes.json`, `configs/w4/final_test.json`;
   - `data/stretch_a*/` (hash-checked against their manifests).
4. **Protocol hashes cover every `src/clinqa/*.py` file**, plus format configs, prompts, the analysis config and tool schemas.
   - Any change makes new outputs non-comparable with old ones until a parity regeneration (`scripts/compare_rollouts.py`) shows identical outputs.
   - Do not move or rename configs; their paths and hashes are recorded in manifests and in the test list.
5. **Never overwrite outputs.** Use a new label or output directory; the scripts refuse to overwrite.
6. **After editing `data_views.py` or `training_data.py`**, rebuild with `make views-all` and confirm that every `train_sha256` in `data/processed/*/manifest.json` is unchanged.
7. **Scoring policy.**
   - Report v2.1 together with v1.
   - v2.1 measures the asked part of an answer and misses false extra claims ([docs/SCORER_V2_1_KNOWN_ISSUES.md](docs/SCORER_V2_1_KNOWN_ISSUES.md)).
   - Do not interpret differences of 1–2 pp or a few items. A single same-recipe refit (F vs F′) moved the macro by 1.7 pp and changed 114/250 outputs; that is one observation, not a variance estimate.
   - Read every case before citing a fabrication count.

## Adding an experiment

Follow `README.md` → "Add a new experiment". Write the decision and its pre-registered criterion in `docs/DECISIONS.md` (next ID after D-107) before generating results. Report a failed criterion as a fail; never move a threshold after seeing results.

## Documentation conventions

- All repository files are in English.
- Decisions go in `docs/DECISIONS.md` (D-xxx); run narratives go in `docs/EXPERIMENT_JOURNAL.md`.
- Archived documents in `docs/history/` are never edited, apart from the archive banner at the top.
- Keep `README.md` short; details belong in `reports/` and `docs/`.
