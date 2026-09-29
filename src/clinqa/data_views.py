"""Auditable train-only selections. These are NOT chat-formatted SFT data."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from clinqa.analysis.checks import Context, q5_arg_grounding
from clinqa.analysis.features import compute_features
from clinqa.config import load_yaml, resolve
from clinqa.data_io import load_split, sha256_file, split_path, verify_manifest


def build_view(variant: str, output_dir: str | Path = "data/processed",
               analysis_config: str = "configs/analysis.yaml") -> dict[str, Any]:
    if variant not in {"raw", "q5_filtered"}:
        raise ValueError("variant must be raw or q5_filtered")
    cfg = load_yaml(analysis_config)
    manifest = verify_manifest(cfg["data_config"])
    records = load_split("train", cfg["data_config"])
    feats = [compute_features("train", r) for r in records]
    # Never read val/test labels to choose training exclusions. Recompute Q5
    # from current train bytes rather than trusting a potentially stale report.
    ctx = Context(data={"train": records}, feats={"train": feats}, reference=[], cfg=cfg)
    flags = q5_arg_grounding(ctx).flags
    reasons: dict[str, list[str]] = {}
    for flag in flags:
        reasons.setdefault(flag.id, []).append(flag.detail)
    excluded = set(reasons) if variant == "q5_filtered" else set()
    kept = [r for r in records if r["id"] not in excluded]
    destination = resolve(output_dir) / variant
    # Protect canonical data from configurable output paths.
    raw_paths = {split_path(s, cfg["data_config"]).resolve() for s in ("train", "val", "test")}
    if (destination / "train.jsonl").resolve() in raw_paths:
        raise ValueError("output would overwrite a canonical split")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "train.jsonl"
    if variant == "raw":
        shutil.copyfile(split_path("train", cfg["data_config"]), target)
    else:
        target.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in kept), encoding="utf-8")
    audit = []
    for record, feature in zip(records, feats):
        if record["id"] in reasons:
            audit.append({
                "id": record["id"], "excluded": record["id"] in excluded,
                "check": "Q5", "reasons": reasons[record["id"]],
                "review_status": "heuristic_candidate_not_human_adjudicated",
                "input_measurements": [asdict(m) for m in feature.measurements],
                "note": record["note"], "table": record["table"], "question": record["question"],
                "gold_tool_calls": record["tool_calls"],
            })
    (destination / "q5_review.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in audit), encoding="utf-8")
    summary = {
        "variant": variant, "format": "raw_records_not_sft_conversations",
        "policy": "keep_all" if variant == "raw" else "exclude_train_Q5_candidates_only",
        "selection_uses_splits": ["train"], "manual_review_complete": False,
        "input_sha256": {k: v["sha256"] for k, v in manifest["files"].items()},
        "analysis_config_sha256": sha256_file(resolve(analysis_config)),
        "implementation_sha256": {
            name: sha256_file(resolve(name)) for name in (
                "src/clinqa/data_views.py", "src/clinqa/analysis/checks.py",
                "src/clinqa/analysis/features.py", "src/clinqa/parsing.py",
                "src/clinqa/tools.py", "src/clinqa/validation.py")
        },
        "before": {"count": len(records), "answer_types": dict(sorted(Counter(r["answer_type"] for r in records).items()))},
        "after": {"count": len(kept), "answer_types": dict(sorted(Counter(r["answer_type"] for r in kept).items()))},
        "excluded_ids": [r["id"] for r in records if r["id"] in excluded],
        "train_sha256": sha256_file(target),
        "evaluation": {s: {"source": manifest["files"][s]["target"], "filter": "none"} for s in ("val", "test")},
    }
    (destination / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("raw", "q5_filtered"), default="raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--analysis-config", default="configs/analysis.yaml")
    args = parser.parse_args(argv)
    result = build_view(args.variant, args.output_dir, args.analysis_config)
    print(f"{args.variant}: {result['before']['count']} -> {result['after']['count']}; "
          f"excluded={len(result['excluded_ids'])}; val/test unfiltered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
