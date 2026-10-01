#!/usr/bin/env bash
# Wave-three round on one 24 GB RunPod GPU (EXPERIMENTS_WAVE3; DECISIONS D-075 to D-083). Validation and
# train only; no test. Generation batch 2 for every arm. STAGES selects work (default: "core 8b"):
#   core   R0-v1, C-filtered-s42 (wave-two filtered 1e-4 epoch two, regenerated), R0-v3, R0-v3-FS4,
#          F-s42 (relabel view) -- each on val + P1 probes; D-TRAINFIT on the control and F-s42 epoch two
#   8b     8B mask audit, R0-8B (val + P1), 8B GPU smoke, A-8B (filtered 1e-4) both epochs on val + P1 on epoch two
#   seeds  F-s43, F-s44 -- only after configs/w3/gate_fs42.json records an approved pass (D-081)
# Arms whose inputs are not approved (scripts/w3_prep.py check) are skipped with a warning and the script
# exits 2 at the end, so a partial round is never mistaken for a complete one. Scoring, gates and C10 run
# later on CPU (make w3-score). Results are committed and pushed after every step; re-running resumes.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
test -f scripts/runpod_env.sh && source scripts/runpod_env.sh
exec 9>.training.lock
flock -n 9 || { echo "Another experiment runner is active."; exit 1; }
test -z "$(git status --porcelain --untracked-files=no)" || { echo "Tracked files are modified; commit first."; exit 1; }
STAGES=${STAGES:-core 8b}
HF_USER=${HF_USER:-Harrydongyl}
PY="uv run --frozen python"
SKIPPED=()

push() {
  git add outputs reports
  git diff --cached --quiet || git commit -q -m "w3: $1"
  git push -q || echo "WARNING: push failed after '$1'; results are committed locally."
}
approved() { $PY scripts/w3_prep.py check --require "$@" >/dev/null 2>&1; }
skip() { echo "WARNING: skipping $1 ($2 not approved; see scripts/w3_prep.py check)"; SKIPPED+=("$1"); }
gen() {  # config label [generate args...]
  local config=$1 label=$2; shift 2
  $PY -m clinqa.evaluate --config "$config" generate --label "$label" "$@"
}
arm() {  # config label run [--adapter path]: val + P1 probes
  local config=$1 label=$2 run=$3; shift 3
  gen "$config" "$label" --run "$run" --split val "$@"
  gen "$config" "$label" --run "$run" --split val --records-file configs/w3/p1_probes.json "$@"
}
train() {
  local run=$1
  if test -f "outputs/$run/manifest.json"; then echo "Training already completed: $run"; return 0; fi
  local resume=""
  test -f "outputs/$run/run_contract.json" && resume="RESUME=latest"
  make train RUN="$run" $resume 2>&1 | tee -a "outputs/$run.console.log"
  $PY scripts/w2_epochs.py stable --run "$run" || echo "WARNING: $run numerically unstable; evaluating anyway."
}

# 0. Gates. Views are rebuilt because their manifests hash the view code; P1 and train-fit are needed by every arm.
make views
$PY scripts/w3_prep.py check
approved p1 trainfit || { echo "P1 probes and train-fit IDs must be frozen first (make w3-prep)."; exit 1; }
if approved relabel; then
  $PY -m clinqa.data_views --variant q5_relabeled
  $PY -m clinqa.formatting --config configs/format_w3.yaml
fi
make preflight
test -f outputs/gpu_smoke/smoke_report.json || make gpu-smoke
$PY -c 'import json; p=json.load(open("outputs/gpu_smoke/smoke_report.json")); assert p["passed"], "GPU smoke must pass first"'
push "preflight"

if [[ " $STAGES " == *" core "* ]]; then
  # 1. R0-v1: base 4B, parity prompt.
  arm configs/eval_w3_v1.yaml w3_r0_v1 base
  push "R0-v1"

  # 2. C-filtered-s42: wave-two filtered 1e-4 epoch two (step 242), regenerated under the wave-three protocol.
  ctl=checkpoints/w2_filtered_lr1e4_mb1/checkpoint-242
  test -f "$ctl/adapter_model.safetensors" ||
    uv run --frozen hf download "$HF_USER/clinqa-w2_filtered_lr1e4_mb1" --revision "${HF_REV_CONTROL:-main}" \
      --include "checkpoint-242/*" --local-dir checkpoints/w2_filtered_lr1e4_mb1
  test "$($PY scripts/w3_epochs.py ckpt --run w2_filtered_lr1e4_mb1 --epoch 2)" = "$ctl"  # hash-checked
  arm configs/eval_w3_v1.yaml w3_c_filtered_s42 w2_filtered_lr1e4_mb1 --adapter "$ctl"
  gen configs/eval_w3_v1.yaml w3_c_filtered_s42 --run w2_filtered_lr1e4_mb1 --adapter "$ctl" --split train \
    --ids-file configs/w3/trainfit_ids.json
  push "C-filtered-s42"

  # 3. Stronger prompted baselines (inference only).
  if approved prompt_v3; then arm configs/eval_w3_v3.yaml w3_r0_v3 base; push "R0-v3"; else skip R0-v3 prompt_v3; fi
  if approved prompt_v3 fewshot; then
    arm configs/eval_w3_v3_fs4.yaml w3_r0_v3_fs4 base; push "R0-v3-FS4"
  else skip R0-v3-FS4 "prompt_v3/fewshot"; fi

  # 4. F-s42 on the reviewed relabel view; both epochs on val, epoch two on P1 and D-TRAINFIT.
  if approved relabel; then
    train w3_relabel_lr1e4_s42
    $PY scripts/w3_epochs.py generate --run w3_relabel_lr1e4_s42 --config configs/eval_w3_v1.yaml --trainfit
    push "F-s42"
  else skip F-s42 relabel; fi
fi

if [[ " $STAGES " == *" 8b "* ]]; then
  # 5. Qwen3-8B: template audit, zero-shot comparator, smoke, then one SFT run on the locked filtered recipe.
  $PY scripts/w3_prep.py audit-8b
  $PY -m clinqa.formatting --config configs/format_w3_8b.yaml
  arm configs/eval_w3_8b.yaml w3_r0_8b base
  push "R0-8B"
  test -f outputs/w3_8b_smoke/smoke_report.json || $PY -m clinqa.smoke --config configs/train/w3_8b_smoke.yaml
  $PY -c 'import json; p=json.load(open("outputs/w3_8b_smoke/smoke_report.json")); assert p["passed"], "8B smoke failed"'
  push "8B smoke"
  train w3_8b_filtered_lr1e4
  $PY scripts/w3_epochs.py generate --run w3_8b_filtered_lr1e4 --config configs/eval_w3_8b.yaml
  push "A-8B"
fi

if [[ " $STAGES " == *" seeds "* ]]; then
  # 6. Seed replication, only after the F-s42 gate was reviewed and passed (D-081).
  $PY -c 'import json; g=json.load(open("configs/w3/gate_fs42.json")); assert g["approved"] and g["decision"] == "pass", g' ||
    { echo "F-s42 gate not approved as a pass; seeds not run."; exit 1; }
  for s in 43 44; do
    train "w3_relabel_lr1e4_s$s"
    $PY scripts/w3_epochs.py generate --run "w3_relabel_lr1e4_s$s" --config configs/eval_w3_v1.yaml
    push "F-s$s"
  done
fi

if test "${UPLOAD:-0}" = 1; then
  for run in w3_relabel_lr1e4_s42 w3_relabel_lr1e4_s43 w3_relabel_lr1e4_s44 w3_8b_filtered_lr1e4; do
    test -d "checkpoints/$run/final" || continue
    uv run --frozen hf upload "$HF_USER/clinqa-$run" "checkpoints/$run" --repo-type model --private \
      --exclude "*/optimizer.pt" || echo "WARNING: upload failed for $run"
  done
fi
if ((${#SKIPPED[@]})); then echo "Round incomplete; skipped: ${SKIPPED[*]}"; exit 2; fi
echo "Wave-three stages finished: $STAGES. Stop the pod, pull, then run make w3-score locally."
