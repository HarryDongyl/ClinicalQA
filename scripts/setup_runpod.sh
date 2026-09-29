#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Clinical
source scripts/runpod_env.sh
test "$(uname -s)" = Linux
test "$(uname -m)" = x86_64
command -v nvidia-smi >/dev/null
for tool in curl git make python3; do
  command -v "$tool" >/dev/null || { echo "Missing $tool; install via apt-get first."; exit 1; }
done
# The existing Linux lock uses CUDA 13.0.96. Check the HOST driver before downloading wheels.
python3 -c 'import subprocess; rows=subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True).splitlines(); assert rows and all(tuple(map(int,r.strip().split("."))) >= (580,95,5) for r in rows), "Need host NVIDIA driver >=580.95.05 for the existing CUDA 13 lock; select another Pod host, not a different Python package set."'
mkdir -p /workspace/.clinqa-bin /workspace/.cache/uv /workspace/.local/share/uv/python /workspace/.cache/huggingface
if ! command -v uv >/dev/null || ! uv --version | grep -q '^uv 0.12.19'; then
  curl --fail --location --silent --show-error https://astral.sh/uv/0.12.19/install.sh | env UV_INSTALL_DIR=/workspace/.clinqa-bin sh
fi
uv python install 3.11.13
uv sync --frozen --extra train --python 3.11.13
uv run --frozen python -c 'import torch, transformers, peft, bitsandbytes; print("torch",torch.__version__,"cuda",torch.version.cuda,"transformers",transformers.__version__,"peft",peft.__version__,"bnb",bitsandbytes.__version__); assert torch.cuda.is_available(), "CUDA unavailable; check the Pod host driver"'
echo "Environment installed. Next: make data-check, make test, make audit-masks, make preflight."
