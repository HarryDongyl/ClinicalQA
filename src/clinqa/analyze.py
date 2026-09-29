"""Phase 1 entry point: statistics + quality checks -> reports/.

Usage: python -m clinqa.analyze --config configs/analysis.yaml
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from clinqa.analysis.checks import Context, bmi_grounding, run_checks
from clinqa.analysis.common import CheckResult
from clinqa.analysis.features import Features, compute_features
from clinqa.analysis.report import render_report
from clinqa.analysis.stats import compute_stats
from clinqa.config import load_yaml, resolve
from clinqa.data_io import load_reference, load_split, verify_manifest
from clinqa.seed import set_seed


def _select_examples(ctx: Context, checks: list[CheckResult]) -> list[tuple[str, str, dict[str, Any]]]:
    """Deterministic example picks: first matching record (split order, then file order)."""
    flagged = {(f.split, f.id) for c in checks for f in c.flags if c.severity in ("error", "warn")}

    def first(pred, allow_flagged: bool = False):
        for s, r, f in ctx.records():
            if (allow_flagged or (s, r["id"]) not in flagged) and pred(r, f):
                return s, r
        return None

    def bmi_units(r: dict[str, Any], f: Features, weight: set[str], height: set[str]) -> bool:
        if not any(tc["tool"] == "calculate_bmi" for tc in r.get("tool_calls", [])):
            return False
        g = bmi_grounding(r, f, ctx)
        return all(g[k] and g[k]["source"] == "note" for k in ("weight", "height")) and \
            g["weight"]["unit"] in weight and g["height"]["unit"] in height

    picks = [
        ("Example 1 - extractive (lab table)", first(lambda r, f: r["answer_type"] == "extractive" and r["table"]["type"] == "labs")),
        ("Example 2 - numeric_reasoning", first(lambda r, f: r["answer_type"] == "numeric_reasoning" and r["table"]["type"] == "labs")),
        ("Example 3 - tool_call: calculate_bmi, metric units in note", first(lambda r, f: bmi_units(r, f, {"kg"}, {"cm"}))),
        ("Example 4 - uncertain: height missing", first(lambda r, f: r["answer_type"] == "uncertain" and f.uncertain_category == "height")),
        ("Example 5 (additional) - tool_call: calculate_bmi, imperial units in note (lb / in)", first(lambda r, f: bmi_units(r, f, {"lb"}, {"in"}))),
        ("Example 6 (additional) - tool_call: unit_convert", first(lambda r, f: any(tc["tool"] == "unit_convert" for tc in r.get("tool_calls", [])))),
        ("Example 7 (data-quality illustration) - tool_call whose BMI arguments are not in the input (Q5)",
         first(lambda r, f: r["answer_type"] == "tool_call" and any(fl.id == r["id"] for c in checks if c.check_id == "Q5" for fl in c.flags), allow_flagged=True)),
    ]
    return [(title, p[0], p[1]) for title, p in picks if p is not None]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dataset statistics and quality checks.")
    parser.add_argument("--config", default="configs/analysis.yaml")
    args = parser.parse_args(argv)
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    manifest = verify_manifest(cfg["data_config"])

    data = {s: load_split(s, cfg["data_config"]) for s in cfg["splits"]}
    feats = {s: [compute_features(s, r) for r in data[s]] for s in cfg["splits"]}
    ctx = Context(data=data, feats=feats, reference=load_reference(cfg["data_config"]), cfg=cfg)

    stats = compute_stats(data, feats, cfg["report"]["top_templates"])
    checks = run_checks(ctx)
    examples = _select_examples(ctx, checks)

    out_dir = resolve(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_analysis.md").write_text(render_report(stats, checks, examples, manifest, cfg), encoding="utf-8")
    summary = {
        "stats": stats,
        "checks": [
            {
                "id": c.check_id, "title": c.title, "status": c.status, "severity_if_flagged": c.severity,
                "flags_by_split": c.counts_by_split(cfg["splits"] + ["reference"]), "metrics": c.metrics, "notes": c.notes,
            }
            for c in checks
        ],
        "input_sha256": {k: v["sha256"] for k, v in manifest["files"].items()},
    }
    (out_dir / "data_stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out_dir / "quality_flags.jsonl").open("w", encoding="utf-8") as fh:
        for c in checks:
            for fl in c.flags:
                fh.write(json.dumps({"severity": c.severity, **fl.to_dict()}, ensure_ascii=False) + "\n")

    for c in checks:
        cnt = c.counts_by_split(cfg["splits"] + ["reference"])
        print(f"{c.check_id:4s} {c.status:5s} {sum(cnt.values()):5d} flags  {c.title}")
    print(f"wrote {out_dir.relative_to(resolve('.'))}/data_analysis.md, data_stats.json, quality_flags.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
