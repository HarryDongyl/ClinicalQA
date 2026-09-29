#!/usr/bin/env bash
# Source this in every new terminal. All bulky caches and Python installs persist.
export PATH="/workspace/.clinqa-bin:$PATH"
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/.local/share/uv/python
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
