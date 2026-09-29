"""Validation-only scoring and blinded review packet. Never opens a test artifact.

Run: .venv/bin/python scripts/rescore_validation_v2.py --out reports/scorer_v2_validation
Output directories must be new. Original scores are never replaced.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clinqa.scoring_v2 import VERSION, compile_contract, score_v2

DEFAULT_LABELS = ["base", "raw_lr1e4_step000125", "raw_lr1e4_step000250", "raw_lr5e5_step000125",
                  "raw_lr5e5_step000250", "q5filtered_lr5e5_step000121", "q5filtered_lr5e5_step000242"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    out = {}
    for task in sorted({r["answer_type"] for r in rows}):
        rs = [r for r in rows if r["answer_type"] == task]
        counts = Counter(r["status"] for r in rs)
        n, p, f, u = len(rs), counts["pass"], counts["fail"], counts["review"]
        out[task] = {"n": n, "pass": p, "fail": f, "review": u, "resolved_coverage": (p+f)/n,
                     "auto_pass_fraction": p/n, "max_fraction_if_reviews_pass": (p+u)/n,
                     "full_contract_accuracy": p/n if not u else None,
                     "core_status": dict(Counter(r["core_status"] for r in rs)),
                     "legacy_correct": sum(r["legacy"]["correct"] for r in rs)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="reports/scorer_v2_validation")
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    args = ap.parse_args()
    if any("test" in label.lower() or Path(label).name != label for label in args.labels):
        raise ValueError("Only validation labels are permitted; no paths or test labels.")
    dest = Path(args.out)
    if not dest.is_absolute():
        dest = ROOT / dest
    if dest.exists():
        raise FileExistsError("Choose a new output directory; previous review artifacts are immutable.")
    data_path = ROOT / "data/val.jsonl"
    records = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    assert len(records) == 250 and all(r["id"].startswith("val_") for r in records)
    contracts = {r["id"]: compile_contract(r) for r in records}
    input_hashes = {"data/val.jsonl": sha(data_path), "scripts/rescore_validation_v2.py": sha(Path(__file__))}
    for p in (ROOT / "src/clinqa").rglob("*.py"):
        input_hashes[str(p.relative_to(ROOT))] = sha(p)
    results, review, key, changes = {}, [], {}, []
    for label in args.labels:
        path = ROOT / "outputs" / label / "val/trajectories.jsonl"
        info_path = path.parent / "run.json"
        info = json.loads(info_path.read_text())
        if info["split"] != "val" or info["split_sha256"] != sha(data_path):
            raise ValueError("Split provenance mismatch")
        ts = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        trajectories = {t["id"]: t for t in ts}
        assert len(ts) == len(trajectories) == len(records) and trajectories.keys() == contracts.keys()
        input_hashes[str(path.relative_to(ROOT))], input_hashes[str(info_path.relative_to(ROOT))] = sha(path), sha(info_path)
        results[label] = []
        for r in records:
            t, c = trajectories[r["id"]], contracts[r["id"]]
            scored = score_v2(r, t, c)
            results[label].append(scored)
            # All predictions, including passes, enter the blinded adjudication packet.
            code = hashlib.sha256((label + ":" + r["id"]).encode()).hexdigest()[:16]
            key[code] = {"label": label, "id": r["id"]}
            review.append({"review_id": code, "input": {k: r[k] for k in ("note", "table", "question")},
                           "contract": {k: v for k, v in asdict(c).items() if k != "id"},
                           "answer": t.get("final_answer"), "calls": [c for tr in t["turns"] for c in tr.get("calls", [])],
                           "adjudication": {"status": None, "required_facts": [], "contradictions": [], "reason": None}})
            if scored["correct"] != scored["legacy"]["correct"]:
                changes.append({"label": label, **scored})
    # Only write after every input and score has been validated.
    dest.mkdir(parents=True)
    def dump(name, obj):
        (dest / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    def lines(name, rows):
        (dest / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    dump("contracts.json", {i: asdict(c) for i, c in contracts.items()})
    for label, rows in results.items():
        lines(label + ".scored.jsonl", rows)
    summary = {"scorer_version": VERSION, "split": "val", "test_accessed": False,
               "scope": "Bounded input-derived task contracts; review is unresolved, not zero and not correct",
               "runs": {label: summarize(rows) for label, rows in results.items()},
               "behavior": {label: {"tool_e2e_legacy": sum(r["behavior"]["tool_e2e"] is True for r in rows),
                                    "schema_invalid_examples": sum(not r["behavior"]["schema_valid"] for r in rows),
                                    "incomplete_examples": sum(not r["behavior"]["finished_answer"] for r in rows),
                                    "over_calls": sum(r["behavior"]["over_call"] is True for r in rows),
                                    "missing_tool_input_status": dict(Counter(r["status"] for r in rows if r["policy"] == "missing_tool_inputs"))}
                            for label, rows in results.items()}}
    dump("summary.json", summary)
    dump("input_manifest.json", input_hashes)
    lines("changed_scores.jsonl", changes)
    random.Random(42).shuffle(review)
    lines("blinded_review.jsonl", review)
    dump("review_key.json", key)
    report = ["# Validation-only scorer v2 report", "", "No test records or predictions were opened. Legacy scores remain unchanged.",
              "", "Pass/fail/review refer to bounded task contracts. Review is not counted as wrong; full contract accuracy remains unavailable when unresolved examples exist. Intervals describe unresolved coverage conditional on automatic labels, not statistical confidence intervals or validated clinical accuracy. Tool-call rows measure input-grounded policy success (including appropriate no-call behavior), not the original tool E2E metric; unchanged behavior counters are stored separately in summary.json.", "",
              "| Run | Task | Legacy correct | Pass | Fail | Review | Possible pass fraction |", "|---|---|---:|---:|---:|---:|---|"]
    for label, tasks in summary["runs"].items():
        for task, m in tasks.items():
            report.append(f"| {label} | {task} | {m['legacy_correct']}/{m['n']} | {m['pass']} | {m['fail']} | {m['review']} | {m['auto_pass_fraction']:.1%}–{m['max_fraction_if_reviews_pass']:.1%} |")
    report += ["", "Complete the blinded adjudication packet before selecting a model by full semantic accuracy. Keep the identity key separate from reviewers. The packet includes all statuses, so false positives can be audited alongside false negatives.", ""]
    (dest / "REPORT.md").write_text("\n".join(report))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
