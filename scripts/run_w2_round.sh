#!/usr/bin/env bash
# Wave-two round on one 24 GB RunPod GPU (EXPERIMENT_JOURNAL W2-PLAN-006; DECISIONS D-041 to D-051). Validation only.
#   0. preflight + GPU smoke (the smoke report is not committed, so a fresh pod re-runs it)
#   1. W2-E1: prompt v1/v2 on fixed wave-one weights (8 generations, batch 4)            D-043, D-044
#   2. filtered view, mb4/ga4: LR 1e-4 -> 1.5e-4 -> 2e-4, both epoch checkpoints, prompt v1  D-042, D-045
#   3. optional (CROSS=1): prompt v2 on the new epoch checkpoints                         D-050
# The pod generates and runs the legacy scorer only; scorer v2.1 runs later on CPU (make w2-score, D-047).
# Results are committed and pushed after every step, so an interruption loses at most one step; re-running
# the script resumes (finished trainings are skipped, verified generations are reused).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
test -f scripts/runpod_env.sh && source scripts/runpod_env.sh
exec 9>.training.lock
flock -n 9 || { echo "Another experiment runner is active."; exit 1; }
test -z "$(git status --porcelain --untracked-files=no)" || { echo "Tracked files are modified; commit first."; exit 1; }

push() {
  git add outputs reports
  git diff --cached --quiet || git commit -q -m "w2: $1"
  git push -q || echo "WARNING: push failed after '$1'; results are committed locally."
}

# 0. Gates.
make preflight
test -f outputs/gpu_smoke/smoke_report.json || make gpu-smoke
uv run --frozen python -c 'import json; p=json.load(open("outputs/gpu_smoke/smoke_report.json")); assert p["passed"], "GPU smoke must pass first"'
push "preflight"

# 1. W2-E1 prompt ablation on fixed wave-one weights. The adapters come from the private HF repos at the
# revisions recorded in EXPERIMENT_JOURNAL (needs `hf auth login`); their hashes must match the wave-one manifests.
HF_USER=${HF_USER:-Harrydongyl}
fetch() {  # run checkpoint revision
  test -f "checkpoints/$1/$2/adapter_model.safetensors" ||
    uv run --frozen hf download "$HF_USER/clinqa-$1" --revision "$3" --include "$2/*" --local-dir "checkpoints/$1"
  uv run --frozen python - "$1" "$2" <<'PY2'
import json, sys
sys.path.insert(0, "src")
from clinqa.run_info import adapter_sha256
run, ck = sys.argv[1:]
want = {c["step"]: c["adapter_sha256"] for c in json.load(open(f"outputs/{run}/manifest.json"))["checkpoints"]}
got = adapter_sha256(f"checkpoints/{run}/{ck}")
assert got == want[int(ck.split("-")[1])], f"{run}/{ck}: adapter hash differs from the wave-one manifest"
print(f"{run}/{ck}: adapter hash matches the wave-one manifest")
PY2
}
fetch raw_lr1e4 checkpoint-125 cce3b5be36a4358d3e212bff9ea1639f4092343f
fetch raw_lr5e5 checkpoint-125 1f764bcbe4e6fbdc172d9cbf2cf93a174ae2d75d
fetch q5filtered_lr5e5 checkpoint-121 7f8a04ffe0acafbce13fc4418259143aac0f9036
for arm in v1 v2; do
  for r in base raw_lr1e4 raw_lr5e5 q5filtered_lr5e5; do  # D-044
    make eval EVAL_CONFIG="configs/eval_w2_prompt_$arm.yaml" RUN="$r" LABEL="w2p_${arm}_$r"
  done
  push "E1 prompt $arm"
done

# 2. LR ladder at micro_batch 4. If the first mb4 run hits CUDA OOM, the whole ladder switches to the
# documented mb2/ga8 fallback (same effective batch 16) so the rungs stay comparable (D-042).
MB=mb4
test -f outputs/w2_mb4_oom.txt && MB=mb2
train() {
  local run=$1
  if test -f "outputs/$run/manifest.json"; then echo "Training already completed: $run"; return 0; fi
  local resume=""
  test -f "outputs/$run/run_contract.json" && resume="RESUME=latest"
  make train RUN="$run" $resume 2>&1 | tee -a "outputs/$run.console.log"
}
LADDER=()
for lr in lr1e4 lr1p5e4 lr2e4; do
  run="w2_filtered_${lr}_${MB}"
  if test "$lr" = lr2e4 && ! uv run --frozen python scripts/w2_epochs.py stable --run "w2_filtered_lr1p5e4_${MB}"; then
    echo "Skipping $run: 1.5e-4 was numerically unstable (D-045)."
    continue
  fi
  if ! train "$run"; then
    if test "$MB" = mb4 && test "$lr" = lr1e4 && grep -qiE "out of memory|OutOfMemoryError" "outputs/$run.console.log"; then
      grep '"peak_vram_gb"' "outputs/$run/train_log.jsonl" 2>/dev/null | tail -1 > outputs/w2_mb4_oom.txt || true
      echo "CUDA OOM at micro_batch 4; see outputs/$run.console.log" >> outputs/w2_mb4_oom.txt
      push "mb4 OOM, switching to mb2 fallback"
      exec "$0" "$@"   # restart: the lock is released on exec and the marker selects mb2
    fi
    echo "Training failed: $run"; exit 1
  fi
  uv run --frozen python scripts/w2_epochs.py stable --run "$run" || echo "WARNING: $run numerically unstable; evaluating anyway."
  uv run --frozen python scripts/w2_epochs.py generate --run "$run" --prompt v1
  push "val $run (prompt v1)"
  LADDER+=("$run")
done

# 3. Optional prompt-v2 cross on the new checkpoints (D-050).
if test "${CROSS:-0}" = 1; then
  for run in "${LADDER[@]}"; do
    uv run --frozen python scripts/w2_epochs.py generate --run "$run" --prompt v2
    push "val $run (prompt v2)"
  done
fi
echo "Wave-two generation finished (ladder: ${LADDER[*]}). Upload new adapters to HF, stop the pod, then run make w2-score locally."
