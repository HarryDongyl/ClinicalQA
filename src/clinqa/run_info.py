"""Provenance recorded with every training/eval run (PLAN section 9, D-029)."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib import metadata
from typing import Any

from clinqa.config import PROJECT_ROOT, resolve

_PACKAGES = ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "numpy")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def package_versions() -> dict[str, str | None]:
    out = {}
    for p in _PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import torch

        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            p = torch.cuda.get_device_properties(0)
            info.update(gpu=p.name, gpu_memory_gb=round(p.total_memory / 2**30, 1), cuda_version=torch.version.cuda,
                        bf16=torch.cuda.is_bf16_supported())
    except ImportError:
        info["cuda"] = False
    return info


def sha256_file(path: Any) -> str | None:
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def adapter_sha256(adapter_dir: Any) -> str | None:
    from pathlib import Path

    return sha256_file(Path(adapter_dir) / "adapter_model.safetensors") if adapter_dir else None


def data_hashes(view: str | None) -> dict[str, Any]:
    manifest = json.loads(resolve("data/MANIFEST.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {k: v["sha256"] for k, v in manifest["files"].items()}
    if view:
        vm = resolve(f"data/processed/{view}/manifest.json")
        if vm.exists():
            m = json.loads(vm.read_text(encoding="utf-8"))
            live = sha256_file(resolve(f"data/processed/{view}/train.jsonl"))
            if m.get("train_sha256") and live != m.get("train_sha256"):
                raise ValueError(f"view {view}: train.jsonl does not match its manifest; rerun `make views`")
            out["view"] = {"variant": view, "train_sha256": live, "excluded_ids": m.get("excluded_ids", [])}
    return out
