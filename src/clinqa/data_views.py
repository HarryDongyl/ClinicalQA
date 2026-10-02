"""Auditable train-only selections. These are NOT chat-formatted SFT data.

Variants: raw (all 2,000), q5_filtered (Q5 candidates excluded) and q5_relabeled (D-075):
the unchanged q5_filtered rows plus the Q5 candidates a reviewer accepted in
configs/w3/q5_relabel_review.jsonl, relabelled as uncertain with the reviewed answer. IDs,
notes, tables and questions are never changed; rejected candidates stay excluded.
q5_relabeled_egfr (Stretch A, docs/STRETCH_A_PLAN.md section 5): q5_relabeled followed by the 52 frozen,
template-generated calculate_egfr rows from data/stretch_a/train_additions.jsonl (hash-checked against
data/stretch_a/manifest.json). Canonical rows are untouched; the additions carry new sa_train_* IDs.
"""

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


RELABEL_REVIEW = "configs/w3/q5_relabel_review.jsonl"
RELABEL_POLICY = "reviewed_Q5_relabel_to_uncertain_inputs_unchanged"
STRETCH_A_ADDITIONS = "data/stretch_a/train_additions.jsonl"
STRETCH_A_MANIFEST = "data/stretch_a/manifest.json"
STRETCH_A_POLICY = RELABEL_POLICY + "+stretch_a_egfr_template_rows"
VARIANTS = ("raw", "q5_filtered", "q5_relabeled", "q5_relabeled_egfr")


def stretch_a_additions(path: str = STRETCH_A_ADDITIONS, manifest: str = STRETCH_A_MANIFEST) -> list[dict[str, Any]]:
    """The frozen Stretch A train rows in canonical record form (metadata stripped; no tool_calls key when none)."""
    expected = json.loads(resolve(manifest).read_text(encoding="utf-8"))["sha256"][Path(path).name]
    if sha256_file(resolve(path)) != expected:
        raise ValueError(f"{path} differs from {manifest}; Stretch A additions are frozen")
    out = []
    for line in resolve(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rec = {k: r[k] for k in ("id", "note", "table", "question", "answer", "answer_type")}
        if r.get("tool_calls"):
            rec["tool_calls"] = r["tool_calls"]
        if not rec["id"].startswith("sa_train_"):
            raise ValueError(f"{rec['id']}: Stretch A rows must use sa_train_* IDs")
        out.append(rec)
    return out


def load_relabel_review(flagged: set[str], path: str = RELABEL_REVIEW) -> dict[str, dict[str, Any]]:
    """Reviewed decisions for exactly the current Q5 candidates; every row must be decided."""
    rows = [json.loads(line) for line in resolve(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    review = {r["id"]: r for r in rows}
    if len(review) != len(rows) or set(review) != flagged:
        raise ValueError(f"{path}: review IDs must be exactly the {len(flagged)} current train Q5 candidates")
    for r in rows:
        if r.get("accepted") not in (True, False) or not r.get("reviewer"):
            raise ValueError(f"{path}: {r['id']} is not reviewed (accepted true/false and reviewer are required)")
        if r["accepted"] and not str(r.get("answer") or "").strip():
            raise ValueError(f"{path}: {r['id']} is accepted without an answer")
    return review


def relabeled_record(record: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    """Same id/note/table/question; uncertain target with the reviewed answer and no tool call."""
    out = {k: v for k, v in record.items() if k not in ("answer", "answer_type", "tool_calls")}
    out.update(answer=decision["answer"].strip(), answer_type="uncertain")
    return out


def build_view(variant: str, output_dir: str | Path = "data/processed",
               analysis_config: str = "configs/analysis.yaml", review_path: str = RELABEL_REVIEW) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
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
    review: dict[str, dict[str, Any]] = {}
    additions: list[dict[str, Any]] = []
    if variant in ("q5_relabeled", "q5_relabeled_egfr"):
        review = load_relabel_review(set(reasons), review_path)
        excluded = {i for i, d in review.items() if not d["accepted"]}
        kept = [relabeled_record(r, review[r["id"]]) if r["id"] in review and r["id"] not in excluded else r
                for r in records if r["id"] not in excluded]
        if variant == "q5_relabeled_egfr":
            additions = stretch_a_additions()
            if {r["id"] for r in additions} & {r["id"] for r in kept}:
                raise ValueError("Stretch A IDs collide with canonical train IDs")
            kept = kept + additions
    else:
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
                "relabeled": bool(review.get(record["id"], {}).get("accepted")),
                "check": "Q5", "reasons": reasons[record["id"]],
                "review_status": review.get(record["id"], {}).get("review_type", "heuristic_candidate_not_human_adjudicated"),
                "input_measurements": [asdict(m) for m in feature.measurements],
                "note": record["note"], "table": record["table"], "question": record["question"],
                "gold_tool_calls": record["tool_calls"],
            })
    (destination / "q5_review.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in audit), encoding="utf-8")
    summary = {
        "variant": variant, "format": "raw_records_not_sft_conversations",
        "policy": {"raw": "keep_all", "q5_filtered": "exclude_train_Q5_candidates_only",
                   "q5_relabeled": RELABEL_POLICY, "q5_relabeled_egfr": STRETCH_A_POLICY}[variant],
        "selection_uses_splits": ["train"], "review_complete": variant in ("q5_relabeled", "q5_relabeled_egfr"),
        "manual_review_complete": bool(review) and all(d.get("review_type") == "human_clinical_adjudication" for d in review.values()),
        "clinical_adjudication": bool(review) and all(d.get("review_type") == "human_clinical_adjudication" for d in review.values()),
        "review_types": sorted({d.get("review_type", "unspecified") for d in review.values()}),
        "input_sha256": {k: v["sha256"] for k, v in manifest["files"].items()},
        "analysis_config_sha256": sha256_file(resolve(analysis_config)),
        "implementation_sha256": {
            name: sha256_file(resolve(name)) for name in (
                "src/clinqa/data_views.py", "src/clinqa/analysis/checks.py",
                "src/clinqa/analysis/features.py", "src/clinqa/parsing.py",
                "src/clinqa/tools.py", "src/clinqa/validation.py")
        },
        **({"relabel_review": review_path, "relabel_review_sha256": sha256_file(resolve(review_path)),
            "relabeled_ids": [r["id"] for r in records if r["id"] in review and r["id"] not in excluded]}
           if variant in ("q5_relabeled", "q5_relabeled_egfr") else {}),
        **({"stretch_a_additions": STRETCH_A_ADDITIONS, "stretch_a_additions_sha256": sha256_file(resolve(STRETCH_A_ADDITIONS)),
            "stretch_a_ids": [r["id"] for r in additions]} if variant == "q5_relabeled_egfr" else {}),
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
    parser.add_argument("--variant", choices=VARIANTS, default="raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--analysis-config", default="configs/analysis.yaml")
    args = parser.parse_args(argv)
    result = build_view(args.variant, args.output_dir, args.analysis_config)
    print(f"{args.variant}: {result['before']['count']} -> {result['after']['count']}; "
          f"excluded={len(result['excluded_ids'])}; relabeled={len(result.get('relabeled_ids', []))}; "
          "val/test unfiltered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
