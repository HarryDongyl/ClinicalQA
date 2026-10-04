# Runbook: reproducing the final models on RunPod

GPU work runs on a RunPod pod (RTX 4090 24 GB with a CUDA-13 host driver ≥ 580.95.05, or an A100). Scoring and analysis run on any CPU machine. The archived round-by-round runbook is [history/RUNBOOK_W4.md](history/RUNBOOK_W4.md).

## 1. Pod setup (once per volume)

```bash
cd /workspace
git clone https://github.com/HarryDongyl/ClinicalQA.git Clinical && cd Clinical   # scripts expect /workspace/Clinical
source scripts/runpod_env.sh        # every new terminal: PATH, uv and HF caches on /workspace
bash scripts/setup_runpod.sh        # uv 0.12.19, Python 3.11.13; checks the host driver
make setup                          # uv sync --frozen --extra train --extra qwen35 (installs torch, transformers, hf)
uv run --frozen pytest -q
```

- Do not run a bare `uv sync`: it removes the `train` extra, including `hf`.
- **Hugging Face.** `uv run --frozen hf auth login` with a write token, needed to upload adapters.
- **GitHub.** The runners commit and push after every step. Use a fine-grained personal access token (Contents: read and write) as the password; an account password is rejected.

  ```bash
  git config --global credential.helper "store --file /workspace/.git-credentials"
  git push --dry-run origin <branch>      # enter the token once
  chmod 600 /workspace/.git-credentials
  export GIT_TERMINAL_PROMPT=0            # a failed push warns instead of hanging inside tmux
  ```

## 2. Final models

Run each in `tmux`. Every runner refuses to start with modified tracked files and resumes finished steps: training is skipped when its manifest exists, and evaluations reuse verified outputs.

| Model | Command | Measured |
|---|---|---|
| **Core: Qwen3.5 relabel** | `make train RUN=w4_q35_4b_relabel_lr1e4`, then `scripts/w3_epochs.py generate --run w4_q35_4b_relabel_lr1e4 --config configs/eval_w4_q35_4b.yaml --p1-all-epochs --trainfit` (or `STAGES=r1`, which adds its controls) | 56.6 min training, 11.3 GiB (A100) |
| **Stretch A: A-sft2-Q35** | `STAGES=r2c UPLOAD=1 bash scripts/run_w4_round.sh` | 87.0 min training, 12.1 GiB (RTX 4090) |
| F′ (Qwen3 comparator) | `STAGES=refit UPLOAD=1 bash scripts/run_w4_round.sh` | 42.5 min, 7.8 GiB (RTX 4090) |
| A-sft2 (Qwen3 comparator) | `STAGES=r2b UPLOAD=1 bash scripts/run_w4_round.sh` | 59.7 min, 8.4 GiB (RTX 4090) |

Qwen3.5 needs `flash-linear-attention` to import (installed by `make setup`). `causal-conv1d` is not installed. The first steps are slow while Triton compiles kernels; `scripts/runpod_env.sh` keeps that cache on `/workspace`.

Without GPU training, download the published adapter and evaluate only:

```bash
uv run --frozen hf download Harrydongyl/clinqa-w4_q35_4b_relabel_lr1e4 --include "checkpoint-250/*" \
  --local-dir checkpoints/w4_q35_4b_relabel_lr1e4
uv run --frozen python -m clinqa.evaluate --config configs/eval_w4_q35_4b.yaml generate --label my_q35_val \
  --run w4_q35_4b_relabel_lr1e4 --adapter checkpoints/w4_q35_4b_relabel_lr1e4/checkpoint-250 --split val
```

### Stretch A on Qwen3.5 and the second test use (D-105; done, do not repeat)

The Stretch A model was produced in one RTX 4090 session:

```bash
STAGES=r2c UPLOAD=1 bash scripts/run_w4_round.sh 2>&1 | tee -a outputs/w4_r2c.console.log
```

This runs, in order:

1. a zero-shot v1e2 arm on the Qwen3.5 relabel adapter;
2. the v1e2 mask audit (`reports/w4/mask_audit_q35_4b_egfr2.json` must pass);
3. A-sft2-Q35 training (2,200 rows, 276 steps);
4. both epochs on val, P1 and the eGFR sets.

flash-linear-attention must import. Then freeze and run the second, disclosed test use:

```bash
uv run --frozen python scripts/w4_test.py freeze --frozen configs/w4/final_test_q35.json \
  --entry asft2_q35_test configs/eval_w4_q35_4b_tools3_v1e2.yaml w4_q35_4b_relabel_egfr2_lr1e4 \
          checkpoints/w4_q35_4b_relabel_egfr2_lr1e4/checkpoint-276
git add configs/w4/final_test_q35.json && git commit -m "w4: freeze second test list (D-105)" && git push
STAGES=test TEST_LIST=configs/w4/final_test_q35.json bash scripts/run_w4_round.sh 2>&1 | tee -a outputs/w4_test_q35.console.log
```

Take the epoch-2 step from `outputs/w4_q35_4b_relabel_egfr2_lr1e4/manifest.json`; 276 is expected for 2,200 rows.

## 3. Score locally (CPU)

```bash
git pull
uv run python scripts/score_v21_val.py --out reports/<new_dir>/v21 --labels w4_q3_refit_relabel_lr1e4_s42_step000250
uv run python scripts/w3_analyze.py p1 --out reports/<new_dir>/p1 --labels w4_q3_refit_relabel_lr1e4_s42_step000250
uv run python scripts/stretch_a_score.py score --out reports/<new_dir>/stretch_a --labels w4_q3_relabel_egfr2_lr1e4_step000276
```

Run these under bash, because zsh does not word-split `$VAR` label lists. Output directories are never overwritten; always choose a new one.

## 4. The test run (done once; do not repeat)

Test was run once on 2026-10-03 from the frozen list `configs/w4/final_test.json`, at commit `7ae0034` on an RTX 4090 (D-098, D-101), and once more for A-sft2-Q35 only from `configs/w4/final_test_q35.json` at `724e707` (D-105, D-106). Results: `outputs/*_test/test/` and `reports/w4/test/TEST_REPORT.md`.

The protocol, for the record:

1. `scripts/w4_test.py freeze --entry LABEL EVAL_CONFIG RUN ADAPTER …` writes the list. It hashes each eval config, its protocol and each adapter, and is committed before any test output exists.
2. `STAGES=test bash scripts/run_w4_round.sh` calls `w4_test.py run`. It refuses a dirty tree (untracked files count), any changed hash, or existing test outputs.
3. A rerun needs `--rerun-reason`, which is written to `RERUN.txt` and disclosed.

## 5. Other rounds

The wave-3 round is `make w3-round` (`scripts/run_w3_round.sh`). Wave-4 stages are `r1` (Qwen3.5 relabel; requires flash-linear-attention) and `r2` (first Stretch A). Wave 2 is `make w2-round` (`scripts/legacy/run_w2_round.sh`). Each round's commands, labels and expected outputs are recorded in [EXPERIMENT_JOURNAL.md](EXPERIMENT_JOURNAL.md) and [history/RUNBOOK_W4.md](history/RUNBOOK_W4.md).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Tracked files are modified; commit first.` | `make views` regenerated `data/processed/*/manifest.json`. If `train_sha256` is unchanged (`git diff data/processed \| grep train_sha256` prints nothing), commit the manifests |
| `Failed to spawn: hf` | `make setup` was not run, or a bare `uv sync` removed the `train` extra |
| The runner hangs after a step | `git push` is waiting for credentials. Set up the token and `export GIT_TERMINAL_PROMPT=0` |
| `hf download` 404 | The adapter repository is private, or was never uploaded. Check with `HfApi().list_models(author='Harrydongyl')` |
| `flash-linear-attention not importable` | Needed only for Qwen3.5 (`r1`). Qwen3 stages warn and continue |
