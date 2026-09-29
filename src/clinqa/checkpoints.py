"""Checkpoint discovery and resumability checks, independent of GPU imports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clinqa.run_info import adapter_sha256


def discover(root: Path) -> list[dict[str, Any]]:
    found = []
    for path in root.glob("checkpoint-*"):
        state_path = path / "trainer_state.json"
        if not state_path.exists() or not adapter_sha256(path):
            continue
        state = json.loads(state_path.read_text())
        epoch = state.get("epoch") or 0.0
        found.append({"step": state["global_step"], "epoch": epoch, "path": str(path),
                      "adapter_sha256": adapter_sha256(path),
                      "epoch_end": epoch > 0 and abs(epoch - round(epoch)) < 1e-6})
    return sorted(found, key=lambda c: c["step"])


def resume_path(root: Path, requested: str | None) -> Path | None:
    if requested is None:
        return None
    candidates = [Path(c["path"]) for c in discover(root)] if requested == "latest" else [Path(requested)]
    for path in reversed(candidates):
        required = ("optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth")
        if path.resolve().parent == root.resolve() and all((path / name).is_file() for name in required):
            return path
    raise ValueError("no complete resumable checkpoint in this run; adapter-only saves cannot resume training")
