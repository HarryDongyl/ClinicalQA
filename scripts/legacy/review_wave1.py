"""Reproduce the wave-one artifact audit without generating or replacing predictions.

Run from Clinical: .venv/bin/python scripts/legacy/review_wave1.py [--verify-hf]
Only reports/wave1_review is written. Hugging Face access is read-only and optional.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from clinqa.evaluate import aggregate
from clinqa.metrics import score_example, args_ok
from clinqa.analysis.features import classify_uncertain
import numpy as np

OUT = ROOT / "reports/wave1_review"
INPUTS = {}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path, lines=False):
    path = ROOT / path
    INPUTS[str(path.relative_to(ROOT))] = digest(path)
    text = path.read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()] if lines else json.loads(text)


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def paired(left, right, repetitions=5000):
    """Paired, answer-type-stratified bootstrap of the legacy four-task macro."""
    assert left.keys() == right.keys()
    rng = np.random.default_rng(42)
    distributions, estimates = [], []
    for task in sorted({r["answer_type"] for r in left.values()}):
        ids = [i for i, r in left.items() if r["answer_type"] == task and not r["q5"]]
        delta = np.array([int(right[i]["correct"]) - int(left[i]["correct"]) for i in ids])
        estimates.append(float(delta.mean()))
        distributions.append(delta[rng.integers(0, len(delta), (repetitions, len(delta)))].mean(axis=1))
    return {"subset": "q5_grounded", "metric": "legacy_macro_task_success", "right_minus_left": float(np.mean(estimates)),
            "paired_bootstrap_ci95": np.quantile(np.mean(distributions, axis=0), [.025, .975]).tolist(),
            "seed": 42, "repetitions": repetitions,
            "limitation": "Conditional on these scored examples; no training-seed uncertainty, multiplicity adjustment, or scorer-bias correction."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-hf", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    INPUTS[str(Path(__file__).resolve().relative_to(ROOT))] = digest(Path(__file__).resolve())
    data = {s: {r["id"]: r for r in read(f"data/{s}.jsonl", True)} for s in ("val", "test")}
    runs, scores_by_run, trajectories_by_run = {}, {}, {}
    for path in sorted((ROOT / "outputs").glob("*/*/metrics.json")):
        split, label = path.parent.name, path.parent.parent.name
        if split not in data:
            continue
        rel = path.parent.relative_to(ROOT)
        metrics, info = read(rel / "metrics.json"), read(rel / "run.json")
        scores = read(rel / "scored.jsonl", True)
        trajectories = read(rel / "trajectories.jsonl", True)
        sm, tm = {r["id"]: r for r in scores}, {r["id"]: r for r in trajectories}
        assert len(sm) == len(scores) == len(data[split]) == len(tm) == len(trajectories)
        assert sm.keys() == tm.keys() == data[split].keys()
        assert digest(ROOT / f"data/{split}.jsonl") == info["split_sha256"]
        prompt_path = ROOT / info["protocol"]["format_config"]["system_prompt"]
        INPUTS[str(prompt_path.relative_to(ROOT))] = digest(prompt_path)
        assert digest(prompt_path) == info["protocol"]["prompt_sha256"]
        assert hashlib.sha256(prompt_path.read_text().strip().encode()).hexdigest() == info["system_prompt_sha256"]
        for i, row in sm.items():
            expected = asdict(score_example(data[split][i], tm[i]))
            assert all(row[k] == v for k, v in expected.items()), (label, i)
        for subset, selected in (("full", scores), ("q5_grounded", [r for r in scores if not r["q5"]]),
                                 ("q5_only", [r for r in scores if r["q5"]])):
            assert aggregate(selected) == metrics["subsets"][subset], (label, subset)
        source_checks = {p: digest(ROOT / p) == sha for p, sha in info["protocol"]["source_sha256"].items()}
        assert all(source_checks.values()), (label, source_checks)
        for p in source_checks:
            INPUTS[p] = digest(ROOT / p)
        call_turns = [t for r in trajectories for t in r["turns"] if t.get("calls")]
        subgroups = {}
        for row in scores:
            record = data[split][row["id"]]
            group = row["answer_type"]
            if row["q5"]:
                group = "q5"
            elif group == "uncertain":
                group = "uncertain_" + classify_uncertain(record["question"], record["answer"])
            elif group == "tool_call":
                gold = record["tool_calls"][0]
                detail = args_ok(record, gold, {"name": gold["tool"], "arguments": gold["arguments"]})[1]
                group = "unit_convert" if gold["tool"] == "unit_convert" else ("bmi_imperial" if "imperial" in detail else "bmi_metric")
            subgroups.setdefault(group, []).append(row)
        runs[label] = {"split": split, "n": len(scores), "metrics": metrics["subsets"],
                       "source_hashes_match": all(source_checks.values()), "run": info,
                       "errors": dict(Counter(r["error"] for r in scores)),
                       "q5_errors": dict(Counter(r["error"] for r in scores if r["q5"])),
                       "subgroups": {g: {"n": len(v), "correct": sum(r["correct"] for r in v),
                                         "errors": dict(Counter(r["error"] for r in v))} for g, v in subgroups.items()},
                       "call_turns": len(call_turns), "call_turns_with_text": sum(bool(t.get("content", "").strip()) for t in call_turns),
                       "mean_generated_tokens": float(np.mean([r["n_new_tokens"] for r in trajectories])),
                       "p95_generated_tokens": float(np.quantile([r["n_new_tokens"] for r in trajectories], .95))}
        scores_by_run[label], trajectories_by_run[label] = sm, tm
    assert len(runs) == 9, f"Expected the frozen first-wave inventory, found {len(runs)}. Update the review explicitly for new waves."
    training, adapters = {}, []
    for name in ("raw_lr1e4", "raw_lr5e5", "q5filtered_lr5e5"):
        m = read(f"outputs/{name}/manifest.json")
        log = read(f"outputs/{name}/train_log.jsonl", True)
        training[name] = {"manifest": m, "validation_loss": [r for r in log if "val_loss/all" in r],
                          "selection": read(f"outputs/{name}/selection.json")}
        checkpoints = [(f"checkpoint-{c['step']}", c["adapter_sha256"]) for c in m["checkpoints"] if c["epoch_end"]]
        checkpoints.append(("final", m["final_adapter_sha256"]))
        for sub, expected in checkpoints:
            p = ROOT / "checkpoints" / name / sub / "adapter_model.safetensors"
            actual = digest(p)
            assert actual == expected, str(p)
            INPUTS[str(p.relative_to(ROOT))] = actual
            adapters.append({"run": name, "subfolder": sub, "sha256": actual, "matches_manifest": True})
    comparisons = {}
    for a, b in [("base", "raw_lr1e4_step000125"), ("base_test", "raw_lr1e4_test"),
                 ("raw_lr5e5_step000125", "raw_lr1e4_step000125"),
                 ("raw_lr5e5_step000125", "q5filtered_lr5e5_step000121"),
                 ("raw_lr1e4_step000125", "raw_lr1e4_step000250"),
                 ("q5filtered_lr5e5_step000121", "q5filtered_lr5e5_step000242")]:
        comparisons[f"{a} -> {b}"] = paired(scores_by_run[a], scores_by_run[b])
    case_ids = ["val_001", "val_005", "val_010", "val_018", "val_023", "val_028", "val_041", "val_047", "val_062", "val_065", "val_069", "val_077", "val_080", "val_085", "val_118", "val_124", "val_151", "val_157", "val_181", "val_185", "val_194", "val_200", "val_202", "val_204", "val_224", "val_230", "val_231", "val_235", "val_012"]
    cases = []
    for i in case_ids:
        cases.append({"id": i, "record": data["val"][i], "outputs": {
            label: {"score": scores_by_run[label][i], "answer": trajectories_by_run[label][i]["final_answer"],
                    "calls": [t["calls"] for t in trajectories_by_run[label][i]["turns"] if t.get("calls")]}
            for label in ("base", "raw_lr1e4_step000125", "q5filtered_lr5e5_step000121")}})
    save("summary.json", {"status": "Retrospective artifact audit; no new model inference", "runs": runs,
                          "training": training, "local_adapters": adapters, "paired_comparisons": comparisons})
    save("case_evidence.json", cases)
    if args.verify_hf:
        from huggingface_hub import HfApi, hf_hub_download
        api, remote = HfApi(), []
        for name in training:
            repo = "Harrydongyl/clinqa-" + name
            info = api.model_info(repo, files_metadata=True)
            siblings = {s.rfilename: s for s in info.siblings}
            for local in [a for a in adapters if a["run"] == name]:
                sub = local["subfolder"]
                sha = siblings[f"{sub}/adapter_model.safetensors"].lfs.sha256
                config_local = ROOT / "checkpoints" / name / sub / "adapter_config.json"
                with tempfile.TemporaryDirectory(prefix="clinqa-hf-review-") as cache:
                    config_remote = Path(hf_hub_download(repo, f"{sub}/adapter_config.json", revision=info.sha, cache_dir=cache))
                    config_match = digest(config_local) == digest(config_remote)
                assert sha == local["sha256"] and config_match
                remote.append({"repo": repo, "revision": info.sha, "private": info.private,
                               "subfolder": sub, "adapter_sha256": sha, "weights_match": True,
                               "adapter_config_match": config_match})
        save("huggingface_verification.json", {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "verification": "Pinned remote LFS SHA256 versus local bytes, and downloaded adapter config byte equality", "adapters": remote})
    save("input_manifest.json", INPUTS)
    print(json.dumps({"runs_verified": len(runs), "predictions_rescored": sum(r["n"] for r in runs.values()),
                      "local_adapters_verified": len(adapters), "report_directory": str(OUT),
                      "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
