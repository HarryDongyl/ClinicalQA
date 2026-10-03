# Clinical QA fine-tuning

This project applies supervised fine-tuning (QLoRA) to Qwen3-4B-Instruct-2507. The model answers a clinical question from one encounter note and one structured table. It extracts facts, does simple numeric reasoning, calls tools (`unit_convert`, `calculate_bmi`, and in Stretch A `calculate_egfr`), and says what is missing instead of inventing it. The assignment is in [docs/ASSIGNMENT.md](docs/ASSIGNMENT.md).

**Start with [reports/REPORT.md](reports/REPORT.md).** It covers the approach, trade-offs, results, limitations and key findings.

## Results (test, run once from a frozen list)

| Model | Natural-Q5 fabrication | Grounded tool tasks | v2.1 macro |
|---|---|---|---|
| **F′**, core final model (two tools) | **0/10** | **89/90** | **97.5** |
| **A-sft2**, Stretch A model (three tools) | 0/10 | 87/90 | 96.2 |
| Filter-only control | 6/10 | 87/90 | 95.8 |
| Base model, same prompt | 1/10 | 48/90 | 79.3 |

The adapters are public on the Hugging Face Hub:

- [`Harrydongyl/clinqa-w4_q3_refit_relabel_lr1e4_s42`](https://huggingface.co/Harrydongyl/clinqa-w4_q3_refit_relabel_lr1e4_s42) (F′, checkpoint-250);
- [`Harrydongyl/clinqa-w4_q3_relabel_egfr2_lr1e4`](https://huggingface.co/Harrydongyl/clinqa-w4_q3_relabel_egfr2_lr1e4) (A-sft2, checkpoint-276).

## Quickstart

```bash
make setup-cpu && make test            # CPU: analysis environment and unit tests
make data analyze views-all            # copy the provided data, quality report, train views
```

On a GPU (RTX 4090 24 GB or larger, Linux; pod setup in [docs/RUNBOOK.md](docs/RUNBOOK.md)):

```bash
make setup
STAGES=refit UPLOAD=1 bash scripts/run_w4_round.sh     # train F′, evaluate both epochs on val and P1
STAGES=r2b   UPLOAD=1 bash scripts/run_w4_round.sh     # train A-sft2, evaluate on val, P1 and the eGFR sets
```

To evaluate the published F′ adapter without training:

```bash
uv run --frozen hf download Harrydongyl/clinqa-w4_q3_refit_relabel_lr1e4_s42 --include "checkpoint-250/*" \
  --local-dir checkpoints/w4_q3_refit_relabel_lr1e4_s42
make eval EVAL_CONFIG=configs/eval_w3_v1.yaml RUN=w4_q3_refit_relabel_lr1e4_s42 \
  ADAPTER=checkpoints/w4_q3_refit_relabel_lr1e4_s42/checkpoint-250 LABEL=my_fprime_val
uv run python scripts/score_v21_val.py --out reports/my_fprime --labels my_fprime_val     # CPU
```

Training F′ takes about 45 min and 7.8 GiB on an RTX 4090; evaluating 250 validation items takes about 20 min. `make help` lists every target.

## Add a new experiment

1. **Training config.** Create `configs/train/<run>.yaml` with `extends: configs/train/w3_relabel_lr1e4_s42.yaml` (the F recipe). Set `run_id`, `checkpoint_dir` and `output_dir` to `<run>`, and override only what changes (for example `seed` or `training.learning_rate`).
2. **New training data, if any.**
   - Add a train view in `src/clinqa/data_views.py` (train-only, hash-checked; see `EGFR_VIEWS`).
   - Build it with `uv run python -m clinqa.data_views --variant <view>`.
   - Point a new `configs/format_<name>.yaml` at it (tokenizer, tools, system prompt).
   - Set `format_config`, `train_view` and `expected_train_examples` in the training config.
3. **Eval config.** Add `<run>: {adapter: checkpoints/<run>/final}` under `runs:` in the eval config of the same format, for example `configs/eval_w3_v1.yaml`. Adding a run does not change the evaluation protocol hash.
4. **Train and evaluate (GPU).**

   ```bash
   make train RUN=<run>
   uv run python scripts/w3_epochs.py generate --run <run> --config <eval config>
   ```

   The second command evaluates both epochs on val, and epoch 2 on the P1 probes. Add `--p1-all-epochs`, `--trainfit` or `--records-files configs/w4/egfr_val.json …` as needed.
5. **Score (CPU).**

   ```bash
   uv run python scripts/score_v21_val.py --out reports/<new dir> --labels <run>_step000<N>
   uv run python scripts/w3_analyze.py p1 --out reports/<new dir>/p1 --labels <run>_step000<N>
   ```

6. **Record the decision.** Add a numbered entry to [docs/DECISIONS.md](docs/DECISIONS.md) before looking at results, stating the comparison and the criterion. Never select on test.

Outputs are never overwritten: use a new label or output directory. [configs/README.md](configs/README.md) and [scripts/README.md](scripts/README.md) index every config and script.

## Repository layout

| Path | Contents |
|---|---|
| `_data/` | The provided files, unmodified |
| `data/` | Canonical copies (`make data`), train views (`processed/`), Stretch A rows |
| `src/clinqa/` | Formatting, training, rollout and evaluation, tools and schemas, scorers (v1 `metrics.py`, v2.1 `scorer_v2.py`) |
| `configs/` | Format, training, evaluation and prompt configs; frozen probe sets and the test list |
| `scripts/` | Experiment runners, scoring and analysis (`legacy/` holds superseded scripts) |
| `tests/` | Unit tests (`make test`) |
| `outputs/` | Training manifests and logs, and every evaluation's trajectories, scores and metrics |
| `reports/` | [REPORT.md](reports/REPORT.md), [DATA_QUALITY.md](reports/DATA_QUALITY.md), scorer validation and per-round scoring |
| `docs/` | Decisions, journal, walkthrough, scorer, Stretch A, runbook; `history/` holds archived reviews |

## Documentation

| Document | Use |
|---|---|
| [reports/REPORT.md](reports/REPORT.md) | The report |
| [reports/DATA_QUALITY.md](reports/DATA_QUALITY.md) | Formatted examples, statistics, data issues and their handling |
| [docs/PROJECT_WALKTHROUGH.md](docs/PROJECT_WALKTHROUGH.md) | The project in order, with evidence and decisions |
| [docs/DECISIONS.md](docs/DECISIONS.md), [docs/EXPERIMENT_JOURNAL.md](docs/EXPERIMENT_JOURNAL.md) | Decision log (D-001 to D-104) and experiment journal |
| [docs/SCORER.md](docs/SCORER.md), [docs/SCORER_V2_1_KNOWN_ISSUES.md](docs/SCORER_V2_1_KNOWN_ISSUES.md) | How answers are scored, its validation, and known defects |
| [docs/STRETCH_A.md](docs/STRETCH_A.md) | The `calculate_egfr` tool, data, diagnosis and results |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | RunPod setup and reproduction |

## Reproducibility

- Dependencies are pinned in `uv.lock`. Every run records its git commit, package versions, hardware and the sha256 of its data, prompt, chat template, tool schemas and configs.
- Seed 42; greedy decoding.
- The provided files are checksum-verified before use, and validation and test are never edited.
- The test split was used once for the final comparison, from a list frozen before any output.
