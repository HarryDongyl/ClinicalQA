"""Project paths and YAML config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    """Resolve a path relative to the project root (absolute paths pass through)."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    with resolve(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_run_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving an optional `extends:` chain (deep merge, child wins)."""
    cfg = load_yaml(path)
    parent = cfg.pop("extends", None)
    return _merge(load_run_config(parent), cfg) if parent else cfg
