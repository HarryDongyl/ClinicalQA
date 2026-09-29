#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Clinical
source scripts/runpod_env.sh
# Advisory lock is released by the OS on process exit, including an unexpected kill.
exec 9>/workspace/Clinical/.training.lock
flock -n 9 || { echo "Another experiment runner is active."; exit 1; }
uv run --frozen python -c 'import json; p=json.load(open("outputs/gpu_smoke/smoke_report.json")); assert p["passed"], "GPU smoke must pass first"'
make eval RUN=base
for run in raw_lr1e4 raw_lr5e5 q5filtered_lr5e5; do
  if test -f "outputs/$run/manifest.json"; then
    echo "Training already completed: $run; validating existing artifacts below."
  elif test -f "outputs/$run/run_contract.json"; then
    make train RUN="$run" RESUME=latest
  else
    make train RUN="$run"
  fi
  make epochs RUN="$run"
done
echo "Validation experiments finished. Run make freeze, commit the frozen config, then make final-eval."
