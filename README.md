# Clinical QA Fine-Tuning

Supervised fine-tuning of a language model that answers clinical questions grounded in one encounter note
plus one structured table. The model has to:

- **extract** facts from the note or table,
- do **simple numeric reasoning** over those values,
- **call tools** (`unit_convert`, `calculate_bmi`) with correct arguments and use the results,
- **state uncertainty**: say what is missing instead of making it up.

The assignment text is in [docs/ASSIGNMENT.md](docs/ASSIGNMENT.md) (moved there word for word from the original
`requirements.txt`).

## Current scorer review and navigation

The supplied v2.1 results are **diagnostic only**, pending evaluation repair. Start with [the audit and next experiments](docs/SCORER_V2_1_AUDIT.md), then [the maintained journal](docs/EXPERIMENT_JOURNAL.md).

- **Historical v1:** `src/clinqa/metrics.py`; original run evidence in `outputs/`. Retained for reproducibility.
- **Bounded-contract v2:** `src/clinqa/scoring_v2.py`, `scripts/rescore_validation_v2.py`, [design](docs/SCORER_V2.md), `reports/scorer_v2_validation/`.
- **Candidate v2.1:** `scripts/score_v2.py` and `reports/scorer_v2/`. Its core `src/clinqa/scorer_v2.py` is installed and hash-matched; 16 tests pass and 1,750 validation scores reproduce. Known false-pass defects still block semantic checkpoint selection.
- **Command routing:** `make score-v2` runs the bounded-contract scorer. `make prompt-ablation` and `make w2-round` only generate, and run the legacy scorer on the pod. `make w2-score` (CPU) runs candidate v2.1 through the guarded `scripts/score_v21_val.py` wrapper, then bounded v2, then the legacy paired comparisons and rollout diffs (D-047). Legacy model-selection commands still do not implement semantic release gates.
- **Project history:** [decisions](docs/DECISIONS.md), [data findings](docs/FINDINGS.md), [repository design review](docs/REPOSITORY_DESIGN_REVIEW.md). These document the reasoning and are preserved.

## Status

Current development policy: **validation only; declare no winner until the scorer is repaired** (D-038, D-049).
Wave one is complete. The frozen rule selected raw_lr1e4 step125. Its single test evaluation scored 84.72% legacy grounded macro vs 45.28% for base; see [Experiment journal](docs/EXPERIMENT_JOURNAL.md) section 3 and the Q5 regression noted there.

Wave two is **in progress** (journal W2-PLAN-006 and W2-RUN-007, DECISIONS D-041 to D-052). One pod round covers two experiments:
- the v1/v2 prompt ablation on four fixed wave-one checkpoints plus base;
- a Q5-filtered LR ladder: 1e-4, then 1.5e-4, then 2e-4.

Generation uses batch 4 throughout. The ladder trains at micro-batch 1 with accumulation 16, as in wave one (D-052). Micro-batch 4 was measured slower and is kept only as `MB=mb4`.

```bash
make w2-round            # on the pod, after make setup and hf auth login; CROSS=1 adds prompt v2 on the ladder; resumes
make w2-score            # locally on CPU, after pulling outputs: v2.1 + bounded v2 + legacy compare + rollout diffs
```

Do not use the legacy `select`/`freeze` commands to announce a winner. Q5 relabel proposals, the pre-call sentence experiment and generation batch 8 are deferred (D-048).

| Phase | Scope | Status |
|---|---|---|
| 0-1b | Data copy + checksums, tools, 19 quality checks, data report, train views | done |
| 2 | Native Qwen chat formatting, assistant-only loss masks, token lengths, formatted examples | done (CPU-verified) |
| 3 | Legacy scorer, rollout state machine, evaluation/compare/select | historical v1; retained for reproducibility |
| 4 | QLoRA training script + smoke test | first-wave GPU runs completed |
| 5 | Base + three QLoRA runs and their downloaded artifacts | reviewed; see experiment journal |
| 6 | Validation-only contract scorer v2 and wave-two configurations | implemented; semantic adjudication pending |
| 7 | Wave-two round: prompt ablation + filtered LR ladder at mb1, generation bs4 | E1 generated; ladder running |

Plan: [docs/PLAN.md](docs/PLAN.md). Decisions: [docs/DECISIONS.md](docs/DECISIONS.md) (revision 4: D-037 onward covers the wave-one outcome and wave two).
Target: one 24GB NVIDIA GPU, Qwen3-4B-Instruct-2507 QLoRA SFT. RL is not required; see PLAN section 11.

## Quickstart

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.11 (uv installs Python if needed).

```bash
make setup        # uv sync --frozen --extra train (torch/transformers/peft/bitsandbytes pinned in uv.lock)
make all          # data + analyze + views + test (CPU)
make audit-masks  # format + mask audit (raw and q5_filtered views) -> reports/
make smoke        # CPU/GPU pipeline smoke with Qwen3-0.6B (not a result)
make preflight    # RunPod: data policy, one >=20 GiB CUDA GPU, disk, token-prefix checks
make gpu-smoke    # RunPod: 4B NF4 overfit + reload + both tools; required before formal training
```

Experiments (configs in `configs/train/`; all use Qwen3-4B-Instruct-2507 QLoRA, 2 epochs, effective batch 16):

| RUN | Train view | LR | Purpose |
|---|---|---|---|
| `q5filtered_lr5e5` | Q5-filtered (1,922) | 5e-5 | mitigated candidate |
| `raw_lr5e5` | raw (2,000) | 5e-5 | filtering ablation vs `q5filtered_lr5e5` |
| `raw_lr1e4` | raw (2,000) | 1e-4 | learning-rate ablation vs `raw_lr5e5` |

Individual steps:

```bash
make eval RUN=base                                  # base model + tools on val -> outputs/base/val/
make train RUN=q5filtered_lr5e5                     # full resumable checkpoints every 50 steps and at each epoch end
make train RUN=q5filtered_lr5e5 RESUME=latest       # resume after an interruption (same code/config/packages only)
make epochs RUN=q5filtered_lr5e5                    # eval epoch checkpoints on val -> outputs/<run>/selection.json
make compare A=base B=q5filtered_lr5e5_step000121
make freeze RUNS="q5filtered_lr5e5 raw_lr5e5 raw_lr1e4"   # cross-run selection on val -> configs/final_eval.yaml
make final-eval                                     # frozen test comparison, runs once
uv run python -m clinqa.evaluate score --label <label> --split val   # rescore without a GPU
```

## Running on RunPod (24GB GPU)

The GPU steps are run by hand on a RunPod pod (D-018). Pod disks are ephemeral: code, caches, checkpoints and
outputs live on a network volume mounted at `/workspace`.

### 0. Before RunPod (local machine)

The pod gets the code with `git clone`, so the repository must be committed and pushed first.

```bash
make test
git add -A && git status        # no .claude/, .venv/, checkpoints/ or data/sft/ should appear
git commit -m "..." && git push -u origin main   # private GitHub repo
```

Create two tokens: a GitHub fine-grained token (Contents read/write on this repo only) for pushing results from
the pod, and a Hugging Face write token for uploading adapters to private repos.

### 1. Create the pod

- Network volume: at least 60 GB (preflight refuses less than 25 GiB free; each run keeps several resumable
  checkpoints, plus about 8 GB of model cache).
- Pod: exactly one 24 GB GPU (RTX 4090 recommended; A5000 fine; L4 works but is slow), PyTorch/CUDA 12 template,
  container disk about 30 GB, volume mounted at `/workspace`, SSH exposed (HTTP 6006 optional for TensorBoard).
- Prefer a recent CUDA driver: torch 2.14 ships its own CUDA runtime, and `make preflight` catches a mismatch.

### 2. Pod setup (once)

```bash
apt-get update && apt-get install -y tmux git
tmux new -s clinqa                  # reattach after a disconnect: tmux attach -t clinqa
cat >> ~/.bashrc <<'RC'
export HF_HOME=/workspace/hf_cache
export UV_CACHE_DIR=/workspace/uv_cache
export PATH=$HOME/.local/bin:$PATH
RC
source ~/.bashrc
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env

cd /workspace
git clone https://<github-user>:<GITHUB_TOKEN>@github.com/<github-user>/<repo>.git clinqa
cd clinqa
git config user.name "<name>" && git config user.email "<email>"

make setup
uv run --frozen hf auth login       # Hugging Face write token
nvidia-smi && uv run --frozen python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### 3. Gates (in order; each must pass before the next)

```bash
make all
make audit-masks
make preflight                      # -> outputs/runpod_preflight.json
make gpu-smoke                      # every check must be true; if not, stop and fix before training
git add outputs reports && git commit -m "RunPod preflight + GPU smoke" && git push
```

### 4. Train and validate

The base model is evaluated on val once; each run then trains and evaluates its epoch checkpoints. Use explicit
steps rather than `make core`, which re-runs the base evaluation every time.

```bash
set -o pipefail
( make eval RUN=base && \
  for r in q5filtered_lr5e5 raw_lr5e5 raw_lr1e4; do
    make train RUN=$r && make epochs RUN=$r && \
    git add outputs && git commit -m "val: $r" && git push || exit 1
  done ) 2>&1 | tee outputs/runs.log
```

Monitor from a second tmux window (`Ctrl-b c`):

```bash
tail -f outputs/<run>/train_log.jsonl    # loss, lr, grad_norm, sec_per_step, peak_vram_gb
watch -n 5 nvidia-smi
uv run --frozen tensorboard --logdir outputs --host 0.0.0.0 --port 6006   # optional
```

Each run is about 242-250 optimizer steps. Wave one measured about 2,520-2,621 s per run at micro-batch 1 on an RTX 4090, with a peak of 7.81 GiB.

**Interrupted run.** Start a pod on the same volume, then:

```bash
cd /workspace/clinqa
make train RUN=<run> RESUME=latest      # restores adapter, optimizer, scheduler and RNG state
make epochs RUN=<run>                   # already-scored checkpoints are reused
```

Resuming is refused unless config, data, source code and package versions match the original run (the run
contract in `outputs/<run>/run_contract.json`). Do not `git pull` or edit code during a run; if code must change,
use a new `run_id`.

### 5. Freeze, then test once

```bash
git status                              # must be clean
make freeze RUNS="q5filtered_lr5e5 raw_lr5e5 raw_lr1e4"
cat configs/final_eval.yaml outputs/final_selection.json
git add configs/final_eval.yaml outputs && git commit -m "Freeze final selection" && git push
make final-eval                         # refuses a dirty tree, a changed adapter hash or an existing test run
git add outputs && git commit -m "Final test evaluation" && git push
```

`make freeze` writes the selected checkpoint into `configs/final_eval.yaml` automatically; do not edit it by hand.

### 6. Save adapters and stop billing

```bash
for r in q5filtered_lr5e5 raw_lr5e5 raw_lr1e4; do
  uv run --frozen hf upload <hf-user>/clinqa-$r checkpoints/$r --repo-type model --private
done
```

Stop or terminate the pod. The network volume is billed until deleted; delete it only after the uploads and
`git push` have succeeded.

## Artifacts

| Path | Contents | Committed |
|---|---|---|
| `reports/` | data analysis, formatted examples, mask audit, token lengths, REPORT | yes |
| `data/sft/` | formatted conversations + gold sidecars | no (regenerate with `make format`) |
| `outputs/<run>/` | training manifest, `train_log.jsonl`, TensorBoard events | yes (small) |
| `outputs/<label>/<split>/` | `trajectories.jsonl`, `scored.jsonl`, `metrics.json`, `error_analysis.md`, `run.json` | yes |
| `checkpoints/<run>/` | LoRA adapters per epoch | no (private HF Hub) |

## Repository layout

```text
_data/                  provided files, untouched (source of truth)
data/                   byte-identical copies under canonical names + MANIFEST.json (sha256)
  train.jsonl val.jsonl test.jsonl reference.jsonl
configs/
  data.yaml             source -> target mapping and expected split sizes
  analysis.yaml         thresholds for the quality checks
  format_core.yaml      tokenizer/template pin, system prompt, max length
  prompts/system_v1.txt frozen system prompt (hashed into every run)
  train_raw.yaml        R1 QLoRA config; train_grounded.yaml (R2) and smoke.yaml extend it
  eval_core.yaml        runs, budgets, selection rule, bootstrap
  final_eval.yaml       frozen test comparison list
src/clinqa/
  data_io.py            copy/validate/load raw data (strict JSON)
  validation.py         structural record validation (types, IDs, table shape, tool args)
  data_views.py         raw / Q5-filtered train-only record views + manifests
  tools.py              unit_convert, calculate_bmi, execute_tool (deterministic)
  parsing.py            regex extractors: weight/height, stated BMI, allergies, medications, table panels
  seed.py               global seeding
  analyze.py            Phase 1 entry point
  analysis/             features, statistics, checks Q1-Q19, Markdown report
  schemas.py            tool JSON schemas, strict call validation, <tool_call> wire parsing
  formatting.py         records -> native Qwen conversations, assistant-only labels (prefix-diff masks)
  audit_masks.py        mask audit + formatted examples report
  metrics.py            deterministic per-example scoring rules
  infer.py              rollout state machine (generate -> parse -> validate -> execute -> answer)
  modeling.py           tokenizer/model loading (NF4 on CUDA)
  train.py              Trainer + PEFT QLoRA training with JSONL/TensorBoard logging
  smoke.py              pre-training smoke checks
  evaluate.py           generate / score / compare / select / final
  run_info.py           provenance (git, versions, hardware, hashes)
tests/                  pytest suite
reports/                generated: data_analysis.md, data_stats.json, quality_flags.jsonl
docs/                   assignment, plan, findings, decisions, data card
```

## Documentation

| Document | Contents |
|---|---|
| [docs/ASSIGNMENT.md](docs/ASSIGNMENT.md) | The assignment specification (verbatim) |
| [docs/PLAN.md](docs/PLAN.md) | Phased plan, priorities, gates, deliverables, status |
| [docs/FINDINGS.md](docs/FINDINGS.md) | How the requirements were interpreted, data findings, limits of the heuristics |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Current decisions, superseded choices and resolutions of P-xxx |
| [docs/DATA.md](docs/DATA.md) | Data card: files, schema, conventions, known issues, lineage |
| [reports/data_analysis.md](reports/data_analysis.md) | Generated statistics, quality checks and examples |

## Reproducibility

- Dependencies are pinned in `uv.lock`, and `requirements.txt` is exported from it.
- `_data/` is never modified. `data/MANIFEST.json` records the SHA-256 of every copied file, and a test checks it.
- `make analyze` gives identical output every run, including across `PYTHONHASHSEED` values. The seed comes from
  `configs/analysis.yaml`.
- Test labels have already been inspected for data auditing. Test is not used for training or model selection; final model comparison is frozen before generating test outputs.
- The raw view keeps all 2,000 train rows; the optional Q5-filtered view keeps 1,922 heuristic-selected rows and includes review/provenance files. val/test stay unchanged.
- Seeds: 42 everywhere (`configs/*.yaml`); training uses deterministic kernels where torch allows; decoding is greedy.
- Every training and eval run records git commit/dirty state, package versions, hardware, data/prompt/template hashes and runtime.
- Measured wave-one results are recorded in the experiment journal. Wave-two entries are plans until their outputs exist; no planned result is reported as measured.
