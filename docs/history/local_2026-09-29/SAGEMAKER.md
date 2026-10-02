# Running v1 on AWS SageMaker (single 24GB GPU)

This runbook replaces the RunPod runbook in the README (D-018) for SageMaker. The pipeline does not change;
only the host setup does. Written 2026-09-29 against the v1 code. Nothing below has been run on a GPU yet.

## 0. Choose the environment

| Option | Instance | Persistent path | Notes |
|---|---|---|---|
| **SageMaker Studio JupyterLab space** (recommended) | `ml.g5.2xlarge` (A10G 24GB, 32GB RAM) or `ml.g6.2xlarge` (L4 24GB) | `/home/sagemaker-user` | Set space storage to **100 GB** before starting (the default is 5 GB). Turn idle shutdown off, or set it longer than 6 h. |
| SageMaker notebook instance | same types | `/home/ec2-user/SageMaker` only | Set the EBS volume to 100 GB. Anything outside `SageMaker/` is lost on stop. |

`ml.g5.xlarge` (16 GB RAM) also works. Use `2xlarge` for extra RAM headroom while the 4B weights load.
Both A10G and L4 support bf16. The domain needs outbound internet to reach huggingface.co, download.pytorch.org,
astral.sh and PyPI. VPC-only domains need a NAT gateway.

Below, `ROOT` is the persistent path from the table.

## 1. Get the code onto the instance

The repo has **no commits yet**, and `make final-eval` refuses to run without a clean, committed tree (D-036).
Commit locally first:

```bash
# local machine
cd ~/Desktop/Personal/ClinicalQA
git checkout -b main
git add -A && git commit -m "v1: data, formatting, training, evaluation pipeline"
git remote add origin git@github.com:<you>/clinqa.git   # private repo (D-031)
git push -u origin main
```

`_data/` and `data/*.jsonl` are committed (they are small). `checkpoints/`, `data/sft/`, `.venv` and `_document/`
are gitignored.

## 2. One-time host setup (SageMaker terminal)

```bash
export ROOT=/home/sagemaker-user            # notebook instance: /home/ec2-user/SageMaker
cd $ROOT
nvidia-smi                                  # note GPU and "Driver Version" -- see step 3

# caches on the persistent disk
cat >> ~/.bashrc <<EOF
export ROOT=$ROOT
export HF_HOME=$ROOT/hf_cache
export UV_CACHE_DIR=$ROOT/.uv_cache
export UV_PYTHON_INSTALL_DIR=$ROOT/.uv_python
export PATH=\$HOME/.local/bin:\$PATH
EOF
source ~/.bashrc

curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
which make tmux || (sudo apt-get update && sudo apt-get install -y make tmux)   # notebook instance: sudo yum install -y make tmux

git clone git@github.com:<you>/clinqa.git clinqa && cd clinqa   # or https + token
conda deactivate 2>/dev/null; unset LD_LIBRARY_PATH             # keep conda's CUDA libs out of the venv
make setup                                                      # uv sync --frozen --extra train (Python 3.11 downloaded by uv)
```

## 3. CUDA check (blocking issue in v1)

`uv.lock` pins `torch==2.14.0` from PyPI. That build bundles **CUDA 13** runtime libraries (`nvidia-*-cu13`,
`cuda-toolkit 13.0.3`), which need NVIDIA driver **>= 580**. Many SageMaker images still ship a 535/550/570
driver (CUDA 12.x).

```bash
uv run python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

- Prints `True` and a GPU name: continue to step 4 and use plain `make ...`.
- Prints `False`, or errors with "driver too old": install the CUDA 12.6 build of the **same** torch version
  (checked: `torch-2.14.0+cu126` for cp311 exists on download.pytorch.org). From then on, stop uv from re-syncing
  back to the locked cu13 build:

  ```bash
  uv pip install --reinstall "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu126
  echo 'export UV_NO_SYNC=1' >> ~/.bashrc && export UV_NO_SYNC=1   # every `uv run` / make target now skips re-sync
  uv run python -c "import torch;print(torch.version.cuda, torch.cuda.is_available())"
  ```

  Record this in the report: the stack is then "uv.lock + torch cu126 override". Whether bitsandbytes 0.50.2 works
  with it is checked by `make smoke` in the next step.

`UV_NO_SYNC=1` is inherited by make and its nested `$(MAKE)` calls, so the commands below are the same on both paths.

## 4. CPU pipeline + GPU smoke (about 10 min)

```bash
cd $ROOT/clinqa
make all            # copy/validate data (sha256), analysis report, raw + Q5 views, 162 unit tests
make audit-masks    # native Qwen chat formatting -> data/sft/, mask audit, formatted examples
git status --short          # expect empty (generated reports are deterministic); inspect any diff before continuing
make smoke          # Qwen3-0.6B tiny overfit + adapter save/reload + rollout; every check must be true
```

Stop if `make smoke` fails. If a pin has to change, run `make lock` on the instance and commit the new `uv.lock`
(D-032).

## 5. Core run: train R1, eval R0, eval every epoch, select, compare (about 3-4.5 h on A10G/L4)

```bash
tmux new -s core            # survives a browser disconnect; reattach with: tmux attach -t core
cd $ROOT/clinqa && mkdir -p outputs
make core 2>&1 | tee outputs/core.log
```

`make core` runs, in order:

| Step | Command it runs | Output |
|---|---|---|
| format + audit | `python -m clinqa.formatting` / `clinqa.audit_masks` | `data/sft/`, `reports/` |
| train R1 | `python -m clinqa.train --config configs/train_raw.yaml` | `checkpoints/r1/checkpoint-125`, `checkpoint-250`, `final/`; `outputs/r1/{manifest.json,train_log.jsonl,tb/}` |
| eval R0 | `python -m clinqa.evaluate generate --run r0 --split val --label r0` | `outputs/r0/val/` |
| eval epochs + select | `python -m clinqa.evaluate epochs --run r1` | `outputs/r1-ep1/val/`, `outputs/r1-ep2/val/`, `outputs/r1/selection.json` |
| paired compare | `python -m clinqa.evaluate compare --a r0 --b <selected> --split val` | `outputs/compare_r0_vs_<selected>_val_full.{json,md}` |

Or run the steps one at a time (same result, easier to resume):

```bash
make train RUN=r1
make eval RUN=r0
make epochs RUN=r1
make compare A=r0 B=$(uv run python -c "import json;print(json.load(open('outputs/r1/selection.json'))['selected'])")
```

Quick check before the full eval (optional, a few minutes):

```bash
uv run python -m clinqa.evaluate generate --run r0 --split val --label r0-probe --limit 16
```

Monitor from a second terminal:

```bash
tail -f $ROOT/clinqa/outputs/core.log
tail -f $ROOT/clinqa/outputs/r1/train_log.jsonl
watch -n 5 nvidia-smi
uv run tensorboard --logdir outputs --port 6006   # Studio: open <studio-url>/jupyterlab/default/proxy/6006/
```

## 6. Optional R2 ablation (Q5-filtered train view, about 2.5-3.5 h)

```bash
make train RUN=r2
make epochs RUN=r2
make compare A=r1-ep2 B=$(uv run python -c "import json;print(json.load(open('outputs/r2/selection.json'))['selected'])")
```

(Replace `r1-ep2` with the label selected for R1.)

## 7. Freeze and run the test set once (about 30-45 min)

```bash
cat outputs/r1/selection.json          # "selected": "r1-epN", "checkpoints": {"r1-epN": "checkpoints/r1/checkpoint-XXX"}
# edit configs/final_eval.yaml: set the r1-final adapter to checkpoints[selected] exactly
git add outputs configs reports && git commit -m "R0/R1 val results; freeze final comparison" && git push
git status --short                     # must be empty, or final-eval refuses (D-036)
make final-eval                # writes outputs/r0-final/test/, outputs/r1-final/test/
git add outputs && git commit -m "Frozen test results" && git push
```

A second test run needs `uv run python -m clinqa.evaluate final --rerun-reason "<why>"`, and the rerun
must be disclosed in the report (D-030).

## 8. Save artifacts before stopping the instance

```bash
# adapters are gitignored: push them to a private HF repo or to S3
uv run huggingface-cli login
uv run huggingface-cli upload <user>/clinqa-r1 checkpoints/r1/checkpoint-XXX --private
# or
aws s3 sync checkpoints/ s3://<bucket>/clinqa/checkpoints/ && aws s3 sync outputs/ s3://<bucket>/clinqa/outputs/
```

Stop the Studio space or notebook instance when you are done. Billing continues while it is running.

## Runtime and memory estimates (not measured; replace with the logged values)

| Step | A10G / L4 estimate | Basis |
|---|---|---|
| `make setup` | 5-10 min | about 4 GB of wheels |
| model download | 2-5 min | Qwen3-4B about 8 GB bf16 into `HF_HOME` |
| R1 training | 2-3.5 h | 2,000 x 2 epochs, about 1.4k tokens per sequence, micro-batch 1, grad checkpointing, NF4 |
| one val eval (250) | 8-15 min | batch 16, greedy, at most 2 x 256 new tokens |
| test eval (2 x 400) | 30-45 min | |
| VRAM | about 8-12 GB peak | NF4 4B plus LoRA r16 plus activations at 1.6k tokens |

The measured runtime and VRAM go into REPORT section 3 from `outputs/*/run.json` and `outputs/r1/manifest.json`.
