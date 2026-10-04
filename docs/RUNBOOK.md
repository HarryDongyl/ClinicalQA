# Runbook: inspect, evaluate, or reproduce the final models

Run commands from the repository root. The three paths below have different output identities: shipped evidence, fresh-label adapter evaluation, and fresh-run training. None requires deleting historical outputs or rerunning test. Historical orchestration is documented in [history/RUNBOOK_W4.md](history/RUNBOOK_W4.md).

## 1. Inspect and score the shipped evidence (CPU)

Install uv and Python 3.11, then:

```bash
make setup-cpu
make test
make data analyze views-all
make w4-score OUT=reports/final_validation_review
```

`w4-score` defaults to the delivered Qwen3.5 Core epoch 2 and A-sft2-Q35 epoch 2. It reads their saved outputs and writes full v1 metrics, frozen v2.1 summaries, P1 reports, and the Stretch eGFR report. It neither generates model answers nor selects a new checkpoint. Use a new `OUT` on a later invocation; existing report directories are rejected.

The saved final test results are in `reports/w4/test/` and `reports/w4/test_q35/`. The final selection and evaluation chronology, including prior exposure, remains in D-105/D-106 and the experiment journal.

## 2. Prepare a GPU pod

Use Linux with an RTX 4090 24 GB or larger; the pinned CUDA stack requires a compatible host driver (the recorded RunPod setup checks CUDA 13 / driver 580.95.05 or later).

```bash
cd /workspace
git clone https://github.com/HarryDongyl/ClinicalQA.git Clinical
cd Clinical
source scripts/runpod_env.sh
bash scripts/setup_runpod.sh
make setup
make test
make data analyze views-all
```

`make setup` installs the training and Qwen3.5 extras. Do not run `make setup-cpu` or bare `uv sync` inside an active GPU environment: that removes the optional training dependencies. `causal-conv1d` is not installed; the recorded short-convolution path is PyTorch. Initial steps include Triton compilation.

The direct commands below do not upload artifacts or push to GitHub. A GPU smoke run is useful on a new image, but the existing `make gpu-smoke` is a Qwen3 check, not proof of the Qwen3.5 path.

## 3. Evaluate an existing adapter with a fresh label

The recorded Core repository is `Harrydongyl/clinqa-w4_q35_4b_relabel_lr1e4`. Anonymous access returned 401 during release review; confirm access or use a supplied local checkpoint. If access is intentionally private, a read token is sufficient (`uv run --frozen hf auth login`). No write token is needed for evaluation.

For a Hub download, set `CLINQA_HUB_REVISION` to the immutable revision supplied with the adapter before running this block:

```bash
: "${CLINQA_HUB_REVISION:?Set the verified immutable Hub revision first}"
uv run --frozen hf download Harrydongyl/clinqa-w4_q35_4b_relabel_lr1e4 \
  --revision "$CLINQA_HUB_REVISION" --include "checkpoint-250/*" \
  --local-dir checkpoints/w4_q35_4b_relabel_lr1e4
uv run --frozen python -c 'from clinqa.run_info import adapter_sha256; assert adapter_sha256("checkpoints/w4_q35_4b_relabel_lr1e4/checkpoint-250") == "b28d7cc22bd8251c8a92abe97e1abc47f3ff4d2b255d83efd6303b854204a3ff", "adapter hash mismatch"'
```

Once that local checkpoint is available:

```bash
make eval EVAL_CONFIG=configs/eval_w4_q35_4b.yaml RUN=w4_q35_4b_relabel_lr1e4 \
  ADAPTER=checkpoints/w4_q35_4b_relabel_lr1e4/checkpoint-250 LABEL=my_q35_val
uv run --frozen python scripts/score_v21_val.py --out reports/my_q35_val --labels my_q35_val
```

Choose a fresh label and report directory for another evaluation. Historical labels are hash-checked against their original protocol; downloading weights does not authorize overwriting them. The immutable public revision remains an artifact-release requirement, not an invented value in this runbook.

## 4. Reproduce the training recipe under fresh run IDs

The supplied reproduction configs inherit the final training recipe and change only run ID, checkpoint directory, and output directory. Evaluation configs are self-contained because the evaluation CLI does not support `extends`. New outputs remain under `outputs/`, as required by the existing scoring helpers.

### Core: Qwen3.5 relabel

```bash
make train RUN=reproduce_core_q35
uv run --frozen python scripts/w3_epochs.py generate --run reproduce_core_q35 \
  --config configs/eval_reproduce_core_q35.yaml --p1-all-epochs --trainfit
```

This trains from the pinned base on 2,000 reviewed records. Expected epoch checkpoints are 125 and 250. Both are evaluated on validation; epoch 2 is the fixed primary endpoint. To score this Core run alone:

```bash
make score-v21 LABELS=reproduce_core_q35_step000250 OUT=reports/reproduce_core_v21
uv run --frozen python scripts/w3_analyze.py p1 --out reports/reproduce_core_p1 \
  --labels reproduce_core_q35_step000250
```

### Optional Stretch: A-sft2-Q35 recipe

```bash
make train RUN=reproduce_stretch_q35
uv run --frozen python scripts/w3_epochs.py generate --run reproduce_stretch_q35 \
  --config configs/eval_reproduce_stretch_q35.yaml --p1-all-epochs \
  --records-files configs/w4/egfr_val.json configs/w4/egfr_age_probes.json
make w4-score OUT=reports/reproduced_final_validation \
  CORE_LABEL=reproduce_core_q35_step000250 \
  STRETCH_LABEL=reproduce_stretch_q35_step000276
```

The extension is another fresh-base training run, using 2,200 records and prompt v1e2. Expected epoch checkpoints are 138 and 276. `--records-files` is required to produce the eGFR inputs for scoring. Verify actual steps against each new training manifest. The combined scoring command assumes both reproduction paths have completed; it preserves all historical gate decisions.

These names are fresh in the submission. For another full rerun, copy the corresponding train/eval configs and change the run ID and both output paths consistently. Resume only an interrupted run with complete matching state:

```bash
make train RUN=reproduce_core_q35 RESUME=latest
```

A completed run must get a new name. Current-source reproduction need not produce byte-identical outputs to a historical run; retain the new manifest and source hashes.

## 5. Resources and historical runners

- Recorded Core training: 56.6 minutes and 11.3 GiB peak allocated memory on A100.
- Recorded Stretch training: 87.0 minutes and 12.1 GiB on RTX 4090.
- Validation generation: approximately 20 minutes for 250 items; runtime depends on output length and hardware.

These are measured historical jobs, not guaranteed timings for every pod.

`run_w3_round.sh` and `run_w4_round.sh` retain the original experiment orchestration. They use historical run names and may commit, push, or upload. They are not the fresh-clone quickstart. `configs/w4/final_test.json` and `final_test_q35.json` preserve the frozen test lists; do not refreeze or rerun them as a reproduction setup step. Test reruns require an explicit recorded reason.

## Troubleshooting

- **Run artifacts already exist:** choose a new reproduction run ID; resume only an incomplete matching run. Do not delete the shipped evidence.
- **Historical protocol mismatch:** use a fresh evaluation label and preserve the new manifest; do not bypass the hash check.
- **Missing P1/eGFR files:** generate the required probe sets before scoring; use the exact commands above.
- **Report directory exists:** choose a new `OUT`; reports are not overwritten.
- **Missing hf / training packages:** run `make setup` in the GPU environment.
- **Hub 401/404:** check the repository name, revision and access; public availability must be verified separately.
- **FLA import failure:** the final Core and Stretch configurations both require the Qwen3.5 dependency stack.
