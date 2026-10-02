#!/usr/bin/env bash
# Wave-four round on one A100 80GB (docs/EXPERIMENTS_WAVE4.md). Validation, P1, train-fit and Stretch A sets only;
# the test run is a separate, frozen step (scripts/w4_test.py). STAGES selects work (default "r1"):
#   r1     A-Q35-relabel: mask audit, training (PARALLEL=0), both epochs on val + P1 (Wave4-Q1), epoch two train-fit;
#          P1 on A-Q35-filter epoch one; P1-RAW (wave-one raw_lr1e4 epoch two); parity regeneration of
#          A-Q35-filter epoch two on val under the wave-four code (must match the wave-three output item by item)
#   r2     Stretch A on the gated backbone: configs/w4/gate_q35.json decides Qwen3.5 (pass) or Qwen3 (fail -> F-s42).
#          Mask audits; zero-shot arms A-zs-v1 / A-zs-v1e on the relabel adapter and A-zs-base; A-sft training and
#          evaluation on val, P1 and both eGFR sets
#   test   scripts/w4_test.py run (configs/w4/final_test.json must be committed)
# Results are committed and pushed after every step; re-running resumes. Scoring runs later on CPU (make w4-score).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
test -f scripts/runpod_env.sh && source scripts/runpod_env.sh
exec 9>.training.lock
flock -n 9 || { echo "Another experiment runner is active."; exit 1; }
test -z "$(git status --porcelain --untracked-files=no)" || { echo "Tracked files are modified; commit first."; exit 1; }
STAGES=${STAGES:-r1}
HF_USER=${HF_USER:-Harrydongyl}
PY="uv run --frozen python"
P1=configs/w3/p1_probes.json
EGFR="configs/w4/egfr_val.json configs/w4/egfr_age_probes.json"

push() {
  git add outputs reports
  git diff --cached --quiet || git commit -q -m "w4: $1"
  git push -q || echo "WARNING: push failed after '$1'; results are committed locally."
}
gen() {  # config label [generate args...]
  local config=$1 label=$2; shift 2
  $PY -m clinqa.evaluate --config "$config" generate --label "$label" "$@"
}
train() {
  local run=$1
  if test -f "outputs/$run/manifest.json"; then echo "Training already completed: $run"; return 0; fi
  local resume=""
  test -f "outputs/$run/run_contract.json" && resume="RESUME=latest"
  make train RUN="$run" $resume 2>&1 | tee -a "outputs/$run.console.log"
  $PY scripts/w2_epochs.py stable --run "$run" || echo "WARNING: $run numerically unstable; evaluating anyway."
}
fetch() {  # run step: download one checkpoint from the private Hub if absent
  local run=$1 step=$2
  test -f "checkpoints/$run/checkpoint-$step/adapter_model.safetensors" ||
    uv run --frozen hf download "$HF_USER/clinqa-$run" --include "checkpoint-$step/*" --local-dir "checkpoints/$run"
}

# 0. Views and the fla kernel check. Views are rebuilt because their manifests hash the view code.
make views
$PY -m clinqa.data_views --variant q5_relabeled
$PY -m clinqa.data_views --variant q5_relabeled_egfr
$PY -c 'import fla.ops.gated_delta_rule' 2>/dev/null && echo "flash-linear-attention kernels available" ||
  { echo "flash-linear-attention not importable; wave four requires the same kernel as A-Q35-filter (make setup)"; exit 1; }

if [[ " $STAGES " == *" r1 "* ]]; then
  # 1. A-Q35-relabel: same recipe as A-Q35-filter, reviewed relabel view; training alone on the GPU (clean timing).
  $PY scripts/w3_prep.py audit --format configs/format_w4_q35_4b.yaml --view q5_relabeled \
    --out reports/w4/mask_audit_q35_4b_relabel.json
  $PY -m clinqa.formatting --config configs/format_w4_q35_4b.yaml
  train w4_q35_4b_relabel_lr1e4
  $PY scripts/w3_epochs.py generate --run w4_q35_4b_relabel_lr1e4 --config configs/eval_w4_q35_4b.yaml \
    --trainfit --p1-all-epochs
  push "A-Q35-relabel"

  # 2. P1 on A-Q35-filter epoch one (diagnostic: does fabrication grow with training?).
  fetch w3_q35_4b_filtered_lr1e4 121
  ep1=$($PY scripts/w3_epochs.py ckpt --run w3_q35_4b_filtered_lr1e4 --epoch 1)
  gen configs/eval_w4_q35_4b.yaml w4_q35_4b_filtered_lr1e4_step000121 --run w3_q35_4b_filtered_lr1e4 \
    --adapter "$ep1" --split val --records-file "$P1"
  push "P1 A-Q35-filter epoch one"

  # 3. Parity: A-Q35-filter epoch two on val under the wave-four code, same hardware and kernel as wave three.
  fetch w3_q35_4b_filtered_lr1e4 242
  ep2=$($PY scripts/w3_epochs.py ckpt --run w3_q35_4b_filtered_lr1e4 --epoch 2)
  gen configs/eval_w4_q35_4b.yaml w4_q35_4b_filtered_lr1e4_step000242 --run w3_q35_4b_filtered_lr1e4 \
    --adapter "$ep2" --split val
  push "parity A-Q35-filter epoch two"

  # 4. P1-RAW: wave-one raw_lr1e4 epoch two (2,000 raw rows, 250 steps, seed 42), label-only control for F-s42.
  fetch raw_lr1e4 250
  raw=$($PY scripts/w3_epochs.py ckpt --run raw_lr1e4 --epoch 2)
  gen configs/eval_w4_v1.yaml w4_raw_lr1e4_step000250 --run raw_lr1e4 --adapter "$raw" --split val --records-file "$P1"
  push "P1-RAW"
fi

if [[ " $STAGES " == *" r2 "* ]]; then
  # 5. Stretch A. The reviewed A-Q35-relabel gate decides the backbone (Wave4-Q2 stopping rule).
  decision=$($PY -c 'import json; g=json.load(open("configs/w4/gate_q35.json")); assert g["approved"], g; print(g["decision"])') ||
    { echo "configs/w4/gate_q35.json missing or not approved; run make w4-score and review the gate first."; exit 1; }
  if test "$decision" = pass; then
    fam=q35_4b; adapter_run=w4_q35_4b_relabel_lr1e4; sft=w4_q35_4b_relabel_egfr_lr1e4
  else
    fam=q3; adapter_run=w3_relabel_lr1e4_s42; sft=w4_q3_relabel_egfr_lr1e4
    fetch w3_relabel_lr1e4_s42 250
  fi
  echo "Stretch A backbone: $fam (gate decision: $decision)"
  for pv in v1 v1e; do $PY -m clinqa.formatting --config "configs/format_w4_${fam}_tools3_$pv.yaml"; done
  $PY scripts/w3_prep.py audit --format "configs/format_w4_${fam}_egfr.yaml" --view q5_relabeled_egfr \
    --out "reports/w4/mask_audit_${fam}_egfr.json"
  $PY -m clinqa.formatting --config "configs/format_w4_${fam}_egfr.yaml"
  ck=$($PY scripts/w3_epochs.py ckpt --run "$adapter_run" --epoch 2)
  stretch_arm() {  # config label run [--adapter path]: core val + both eGFR sets
    local config=$1 label=$2 run=$3; shift 3
    gen "$config" "$label" --run "$run" --split val "$@"
    for f in $EGFR; do gen "$config" "$label" --run "$run" --split val --records-file "$f" "$@"; done
  }
  stretch_arm "configs/eval_w4_${fam}_tools3_v1.yaml" "w4_sa_zs_v1_${fam}" "$adapter_run" --adapter "$ck"
  stretch_arm "configs/eval_w4_${fam}_tools3_v1e.yaml" "w4_sa_zs_v1e_${fam}" "$adapter_run" --adapter "$ck"
  stretch_arm "configs/eval_w4_${fam}_tools3_v1e.yaml" "w4_sa_zs_base_${fam}" base
  push "Stretch A zero-shot ($fam)"
  train "$sft"
  $PY scripts/w3_epochs.py generate --run "$sft" --config "configs/eval_w4_${fam}_tools3_v1e.yaml" --records-files $EGFR
  push "Stretch A A-sft ($fam)"
fi

if [[ " $STAGES " == *" test "* ]]; then
  # 6. One confirmatory test run from the committed freeze (option B).
  $PY scripts/w4_test.py run
  push "test (frozen list)"
fi

if test "${UPLOAD:-0}" = 1; then
  for run in w4_q35_4b_relabel_lr1e4 w4_q35_4b_relabel_egfr_lr1e4 w4_q3_relabel_egfr_lr1e4; do
    test -d "checkpoints/$run/final" || continue
    uv run --frozen hf upload "$HF_USER/clinqa-$run" "checkpoints/$run" --repo-type model --private \
      --exclude "*/optimizer.pt" || echo "WARNING: upload failed for $run"
  done
fi
echo "Wave-four stages finished: $STAGES. Stop the pod, pull, then run make w4-score locally."
