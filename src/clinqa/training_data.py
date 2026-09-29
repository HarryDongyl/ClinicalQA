"""Fail-closed validation of train-only views before loading model weights."""
from __future__ import annotations

import json
from typing import Any

from clinqa.analysis.checks import Context, q5_arg_grounding
from clinqa.analysis.features import compute_features
from clinqa.config import load_yaml, resolve
from clinqa.data_io import load_split, read_jsonl, sha256_file, verify_manifest


def grounding_flags(records: list[dict[str, Any]], analysis_config: str = "configs/analysis.yaml") -> list[Any]:
    cfg = load_yaml(analysis_config)
    ctx = Context(data={"train": records}, feats={"train": [compute_features("train", r) for r in records]},
                  reference=[], cfg=cfg)
    return q5_arg_grounding(ctx).flags


def audit_train_view(cfg: dict[str, Any], fmt: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Check hashes, untouched targets, exact train membership and Q5 policy.

    Q5 is a heuristic quarantine, not clinical adjudication. Raw controls must
    explicitly acknowledge unsupported supervision; filtered runs cannot waive it.
    """
    canonical = verify_manifest(fmt["data_config"])
    view = cfg["train_view"]
    if view not in {"raw", "q5_filtered"}:
        raise ValueError(f"unsupported training view: {view}")
    path = resolve(fmt["train_views"][view])
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("training view manifest missing; run make views")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("variant") != view or manifest.get("train_sha256") != sha256_file(path):
        raise ValueError("training view hash/variant mismatch; run make views")
    if manifest.get("input_sha256", {}).get("train") != canonical["files"]["train"]["sha256"]:
        raise ValueError("training view was built from a different canonical train file")
    for name, digest in manifest.get("implementation_sha256", {}).items():
        if sha256_file(resolve(name)) != digest:
            raise ValueError(f"view implementation changed: {name}; run make views")
    analysis_config = cfg.get("analysis_config", "configs/analysis.yaml")
    if manifest.get("analysis_config_sha256") != sha256_file(resolve(analysis_config)):
        raise ValueError("analysis policy changed; run make views")
    original = load_split("train", fmt["data_config"])
    flags = grounding_flags(original, analysis_config)
    excluded = {f.id for f in flags} if view == "q5_filtered" else set()
    expected = [r for r in original if r["id"] not in excluded]
    records = read_jsonl(path)
    if records != expected:
        raise ValueError("view is not the exact, ordered, unmodified canonical train selection")
    if set(manifest.get("excluded_ids", [])) != excluded:
        raise ValueError("view exclusions disagree with current train-only Q5 check")
    if len(records) != cfg.get("expected_train_examples", len(records)):
        raise ValueError("unexpected training count; review data policy before training")
    remaining = grounding_flags(records, analysis_config)
    if remaining and (view != "raw" or not cfg.get("allow_ungrounded_targets", False)):
        raise ValueError(f"{len({f.id for f in remaining})} Q5 candidates remain: use q5_filtered, "
                         "or explicitly set allow_ungrounded_targets for a disclosed raw control")
    report = {
        "view": view, "canonical_train_count": len(original), "train_count": len(records),
        "excluded_ids": [r["id"] for r in original if r["id"] in excluded],
        "remaining_q5_ids": sorted({f.id for f in remaining}),
        "raw_control_acknowledged": bool(cfg.get("allow_ungrounded_targets", False)),
        "policy": "heuristic_Q5_quarantine_no_relabeling" if excluded else "raw_control_unchanged",
        "clinical_adjudication": False, "val_test_modified": False,
        "train_sha256": sha256_file(path), "view_manifest_sha256": sha256_file(manifest_path),
    }
    return records, report
