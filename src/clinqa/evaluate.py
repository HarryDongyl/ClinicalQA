"""Evaluation: generate trajectories (GPU), score them (CPU), compare runs, select checkpoints.

    python -m clinqa.evaluate generate --run base --split val
    python -m clinqa.evaluate epochs --run raw_lr1e4
    python -m clinqa.evaluate freeze --runs raw_lr1e4 raw_lr5e5 q5filtered_lr5e5
    python -m clinqa.evaluate final --final-config configs/final_eval.yaml

Per label and split, under outputs/<label>/<split>/:
  trajectories.jsonl   rollout log, no gold (D-029)
  run.json             model/adapter/prompt/template hashes, versions, hardware, runtime
  scored.jsonl         per-example scores joined by id
  metrics.json         headline metrics with numerators/denominators and Wilson 95% CIs, per subset
  error_analysis.md    failure categories with example ids

Subsets: full; q5_grounded (excludes rows whose gold BMI arguments are not in the input, heuristic
Q5); q5_only. Test is generated only through `final` (D-030) and refuses to overwrite.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from clinqa.analysis.checks import Context, q5_arg_grounding
from clinqa.analysis.features import compute_features
from clinqa.config import PROJECT_ROOT, load_yaml, resolve
from clinqa.data_io import load_split
from clinqa.metrics import score_example
from clinqa.run_info import adapter_sha256, git_state, hardware, package_versions, sha256_file, sha256_text
from clinqa.schemas import tool_schemas_sha256

TYPES = ("extractive", "numeric_reasoning", "tool_call", "uncertain")


# ---------------------------------------------------------------- helpers


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def q5_ids(split: str, analysis_config: str) -> set[str]:
    cfg = load_yaml(analysis_config)
    records = load_split(split, cfg["data_config"])
    ctx = Context(data={split: records}, feats={split: [compute_features(split, r) for r in records]},
                  reference=[], cfg=cfg)
    return {f.id for f in q5_arg_grounding(ctx).flags}


def run_dir(cfg: dict[str, Any], label: str, split: str) -> Path:
    return resolve(cfg["output_dir"]) / label / split


def protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    fmt = load_yaml(cfg["format_config"])
    return {"model": cfg["model"], "budget": cfg["budget"], "batch_size": cfg["batch_size"],
            "format_config": fmt, "prompt_sha256": sha256_file(resolve(fmt["system_prompt"])),
            "analysis_config_sha256": sha256_file(resolve(cfg["analysis_config"])),
            "source_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256_file(p)
                              for p in sorted((PROJECT_ROOT / "src/clinqa").rglob("*.py"))}}


def verify_validation(cfg: dict[str, Any], label: str, adapter: str | None) -> None:
    """Refuse partial, stale or differently configured validation artifacts."""
    out = run_dir(cfg, label, "val")
    info = json.loads((out / "run.json").read_text())
    expected = load_split("val", load_yaml(cfg["format_config"])["data_config"])
    expected_ids = {r["id"] for r in expected}
    for filename in ("trajectories.jsonl", "scored.jsonl"):
        rows = _read_jsonl(out / filename)
        if len(rows) != len(expected) or {r["id"] for r in rows} != expected_ids:
            raise ValueError(f"{label}: incomplete validation IDs")
    if info.get("limit") is not None or info.get("n") != len(expected):
        raise ValueError(f"{label}: partial validation cannot select a checkpoint")
    if info.get("protocol") != protocol(cfg) or info.get("adapter_sha256") != adapter_sha256(adapter):
        raise ValueError(f"{label}: validation protocol or adapter changed; use a new experiment label")
    if info.get("split_sha256") != sha256_file(resolve("data/val.jsonl")):
        raise ValueError(f"{label}: validation split changed")


# ---------------------------------------------------------------- generate


def generate(cfg: dict[str, Any], run: str, split: str, adapter: str | None, label: str,
             limit: int | None = None, extra_info: dict[str, Any] | None = None) -> Path:
    from clinqa.infer import Budget, HFGenerator, rollout
    from clinqa.modeling import load_for_inference, load_tokenizer
    from clinqa.formatting import template_sha256
    from clinqa.seed import set_seed

    fmt = load_yaml(cfg["format_config"])
    if adapter:
        adapter = str(resolve(adapter))
    system = resolve(fmt["system_prompt"]).read_text(encoding="utf-8").strip()
    records = load_split(split, fmt["data_config"])[:limit]
    # The runner never sees gold: strip it before rollout.
    inputs = [{k: r[k] for k in ("id", "note", "table", "question")} for r in records]
    set_seed(42)
    tok = load_tokenizer(cfg["model"], padding_side="left")
    t0 = time.perf_counter()
    model = load_for_inference(cfg["model"], adapter)
    load_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    trajectories = rollout(inputs, HFGenerator(model, tok), tok, system, budget=Budget(**cfg["budget"]),
                           batch_size=cfg["batch_size"])
    gen_s = time.perf_counter() - t1
    out = run_dir(cfg, label, split)
    _write_jsonl(out / "trajectories.jsonl", trajectories)
    import torch

    train_manifest = resolve(f"outputs/{run}/manifest.json")
    info = {"label": label, "run": run, "split": split, "adapter": adapter, "adapter_sha256": adapter_sha256(adapter),
            "protocol": protocol(cfg),
            "train_manifest": str(train_manifest) if adapter and train_manifest.exists() else None,
            "model": cfg["model"], "model_load": getattr(model, "clinqa_load_info", None), "seed": 42,
            "split_sha256": sha256_file(resolve(f"data/{split}.jsonl")), "tool_schemas_sha256": tool_schemas_sha256(),
            **(extra_info or {}),
            "budget": cfg["budget"], "batch_size": cfg["batch_size"], "n": len(trajectories), "limit": limit,
            "system_prompt_sha256": sha256_text(system), "chat_template_sha256": template_sha256(tok),
            "git": git_state(), "packages": package_versions(), "hardware": hardware(),
            "runtime_s": {"model_load": round(load_s, 1), "generate": round(gen_s, 1)},
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None}
    (out / "run.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(trajectories)} trajectories to {out} in {gen_s:.0f}s")
    return out


# ---------------------------------------------------------------- score


def _rate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [r[key] for r in rows if r[key] is not None]
    k = sum(bool(v) for v in vals)
    return {"k": k, "n": len(vals), "rate": round(k / len(vals), 4) if vals else None, "ci95": wilson(k, len(vals))}


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    by = defaultdict(list)
    for s in scores:
        by[s["answer_type"]].append(s)
    tool = by["tool_call"]
    called_tool = [s for s in tool if s["called"]]
    m = {
        "n": len(scores),
        "extractive_accuracy": _rate(by["extractive"], "correct"),
        "numeric_accuracy": _rate(by["numeric_reasoning"], "correct"),
        "tool_selection_accuracy": _rate(tool, "tool_selected"),
        "tool_argument_accuracy": _rate(tool, "tool_args_correct"),
        "tool_argument_accuracy_given_call": _rate(called_tool, "tool_args_correct"),
        "tool_execution_success": _rate(tool, "tool_executed"),
        "tool_result_in_answer": _rate(tool, "result_in_answer"),
        "tool_e2e": _rate(tool, "tool_e2e"),
        "uncertainty_detection_rate": _rate(by["uncertain"], "correct"),
        "over_call_rate": _rate([s for s in scores if s["answer_type"] != "tool_call"], "over_call"),
        "over_refusal_rate": _rate([s for s in scores if s["answer_type"] != "uncertain"], "over_refusal"),
        "unsupported_argument_rate": _rate(scores, "unsupported_args"),  # over examples with >= 1 call
        "errors": {t: dict(Counter(s["error"] for s in by[t]).most_common()) for t in TYPES},
    }
    parts = [m["extractive_accuracy"]["rate"], m["numeric_accuracy"]["rate"], m["tool_e2e"]["rate"],
             m["uncertainty_detection_rate"]["rate"]]
    m["macro_task_success"] = round(sum(parts) / 4, 4) if all(p is not None for p in parts) else None
    return m


def score(cfg: dict[str, Any], label: str, split: str) -> dict[str, Any]:
    out = run_dir(cfg, label, split)
    trajectories = {t["id"]: t for t in _read_jsonl(out / "trajectories.jsonl")}
    records = [r for r in load_split(split, load_yaml(cfg["format_config"])["data_config"]) if r["id"] in trajectories]
    q5 = q5_ids(split, cfg["analysis_config"])
    scores = []
    for r in records:
        s = score_example(r, trajectories[r["id"]]).as_dict()
        s["q5"] = r["id"] in q5
        scores.append(s)
    _write_jsonl(out / "scored.jsonl", scores)
    metrics = {"label": label, "split": split, "q5_ids": sorted(q5 & set(trajectories)),
               "stop_reasons": dict(Counter(t["stop_reason"] for t in trajectories.values())),
               "subsets": {"full": aggregate(scores), "q5_grounded": aggregate([s for s in scores if not s["q5"]]),
                           "q5_only": aggregate([s for s in scores if s["q5"]])}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (out / "error_analysis.md").write_text(error_analysis(label, split, scores, trajectories), encoding="utf-8")
    return metrics


def _snippet(t: dict[str, Any]) -> str:
    calls = [c for turn in t["turns"] for c in turn.get("calls", [])]
    call = f"call={json.dumps(calls[0], ensure_ascii=False)} " if calls else ""
    ans = (t.get("final_answer") or t["turns"][-1]["raw"] if t["turns"] else "")[:160].replace("\n", " ").replace("|", "/")
    return f"{call}stop={t['stop_reason']} answer=\"{ans}\""


def error_analysis(label: str, split: str, scores: list[dict[str, Any]], traj: dict[str, dict[str, Any]]) -> str:
    L = [f"# Error analysis — {label} / {split}", "",
         "Generated by `clinqa.evaluate score`. One primary error category per example; `correct` rows omitted "
         "from the examples. Heuristic scorer: see reports/REPORT.md for limitations.", ""]
    for t in TYPES:
        rows = [s for s in scores if s["answer_type"] == t]
        if not rows:
            continue
        L += [f"## {t} ({sum(s['correct'] for s in rows)}/{len(rows)} correct)", "",
              "| error | count | example ids |", "|---|---|---|"]
        cats = Counter(s["error"] for s in rows)
        for err, n in cats.most_common():
            ids = [s["id"] for s in rows if s["error"] == err][:3]
            L.append(f"| {err} | {n} | {', '.join(ids)} |")
        L.append("")
        for err, _ in cats.most_common():
            if err == "correct":
                continue
            L += [f"**{err}**", ""]
            for s in [s for s in rows if s["error"] == err][:3]:
                L.append(f"- `{s['id']}`{' (Q5)' if s.get('q5') else ''}: {_snippet(traj[s['id']])}")
            L.append("")
    over = [s for s in scores if s["over_call"]]
    ref = [s for s in scores if s["over_refusal"]]
    L += ["## Cross-type behaviour", "",
          f"- over-calls on non-tool examples: {len(over)} ({', '.join(s['id'] for s in over[:5])})",
          f"- over-refusals on non-uncertain examples: {len(ref)} ({', '.join(s['id'] for s in ref[:5])})", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- compare / select


def compare(cfg: dict[str, Any], a: str, b: str, split: str, subset: str = "full") -> dict[str, Any]:
    sa = {s["id"]: s for s in _read_jsonl(run_dir(cfg, a, split) / "scored.jsonl")}
    sb = {s["id"]: s for s in _read_jsonl(run_dir(cfg, b, split) / "scored.jsonl")}
    ids = sorted(i for i in sa.keys() & sb.keys() if subset == "full" or (subset == "q5_grounded") != sa[i]["q5"])
    rng = random.Random(cfg["bootstrap"]["seed"])
    out: dict[str, Any] = {"a": a, "b": b, "split": split, "subset": subset, "n": len(ids), "by_type": {}}
    lines = [f"# Paired comparison — {a} vs {b} / {split} / {subset}", "",
             f"Paired by id; bootstrap {cfg['bootstrap']['n']} resamples (seed {cfg['bootstrap']['seed']}).", "",
             "| answer type | n | " + a + " | " + b + " | diff (b-a) | 95% CI | fixed | broken | both wrong |",
             "|---|---|---|---|---|---|---|---|---|"]
    for t in TYPES:
        tid = [i for i in ids if sa[i]["answer_type"] == t]
        if not tid:
            continue
        x = [int(sa[i]["correct"]) for i in tid]
        y = [int(sb[i]["correct"]) for i in tid]
        diffs = []
        for _ in range(cfg["bootstrap"]["n"]):
            idx = [rng.randrange(len(tid)) for _ in tid]
            diffs.append(sum(y[j] - x[j] for j in idx) / len(tid))
        diffs.sort()
        ci = [round(diffs[int(0.025 * len(diffs))], 4), round(diffs[int(0.975 * len(diffs)) - 1], 4)]
        fixed = [i for i in tid if not sa[i]["correct"] and sb[i]["correct"]]
        broken = [i for i in tid if sa[i]["correct"] and not sb[i]["correct"]]
        wrong = [i for i in tid if not sa[i]["correct"] and not sb[i]["correct"]]
        d = (sum(y) - sum(x)) / len(tid)
        out["by_type"][t] = {"n": len(tid), "a": sum(x), "b": sum(y), "diff": round(d, 4), "ci95": ci,
                             "fixed": fixed, "broken": broken, "both_wrong": wrong}
        lines.append(f"| {t} | {len(tid)} | {sum(x)} | {sum(y)} | {d:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                     f"{len(fixed)} | {len(broken)} | {len(wrong)} |")
    lines += ["", "Broken examples (correct in a, wrong in b):", ""]
    for t, v in out["by_type"].items():
        for i in v["broken"][:5]:
            lines.append(f"- {t} `{i}`: {sa[i]['error']} -> {sb[i]['error']}")
    path = resolve(cfg["output_dir"]) / f"compare_{a}_vs_{b}_{split}_{subset}"
    path.with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def select(cfg: dict[str, Any], labels: list[str], split: str = "val") -> dict[str, Any]:
    """Predeclared checkpoint rule; labels must be given in training order (earliest first)."""
    if split != "val" or not labels or len(set(labels)) != len(labels):
        raise ValueError("selection requires unique candidate labels on validation only")
    sub = cfg["selection"]["subset"]
    rows = []
    for order, label in enumerate(labels):
        m = json.loads((run_dir(cfg, label, split) / "metrics.json").read_text())["subsets"][sub]
        rows.append({"label": label, "order": order, "macro": m["macro_task_success"],
                     "unsupported_calls": m["unsupported_argument_rate"]["k"], "over_calls": m["over_call_rate"]["k"]})
    if any(r["macro"] is None for r in rows):
        raise ValueError("all four task types are required for checkpoint selection")
    best = max(r["macro"] for r in rows)
    near = [r for r in rows if r["macro"] >= best - cfg["selection"]["tie_margin"]]
    chosen = min(near, key=lambda r: (r["unsupported_calls"], r["over_calls"], r["order"]))
    return {"rule": cfg["selection"], "candidates": rows, "selected": chosen["label"]}


def freeze(cfg: dict[str, Any], runs: list[str], destination: str) -> dict[str, Any]:
    """Freeze one candidate from all epoch checkpoints, plus the base control."""
    verify_validation(cfg, "base", None)
    candidates = {}
    for run in runs:
        if run == "base" or run not in cfg["runs"]:
            raise ValueError(f"unknown train run: {run}")
        selection = json.loads((resolve(cfg["output_dir"]) / run / "selection.json").read_text())
        for label, path in selection["checkpoints"].items():
            verify_validation(cfg, label, path)
            state = json.loads((resolve(path) / "trainer_state.json").read_text())
            candidates[label] = {"run": run, "adapter": str(resolve(path)), "epoch": state["epoch"]}
    labels = sorted(candidates, key=lambda label: (candidates[label]["epoch"], runs.index(candidates[label]["run"])))
    decision = select(cfg, labels)
    chosen = candidates[decision["selected"]]
    target = resolve(destination)
    if target.exists() and load_yaml(target).get("runs"):
        raise ValueError("final config is already frozen; refusing to replace test selection")
    final = {"selection": decision, "candidate_runs": runs, "protocol": protocol(cfg),
             "runs": [{"label": "base_test", "run": "base", "adapter": None},
                      {"label": chosen["run"] + "_test", "run": chosen["run"],
                       "adapter": str(resolve(chosen["adapter"]).relative_to(PROJECT_ROOT)),
                       "adapter_sha256": adapter_sha256(chosen["adapter"])}]}
    import yaml
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(final, sort_keys=False), encoding="utf-8")
    (resolve(cfg["output_dir"]) / "final_selection.json").write_text(json.dumps(final, indent=2) + "\n")
    return final


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/eval_core.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--run", required=True)
    g.add_argument("--split", default="val", choices=["train", "val"])
    g.add_argument("--adapter", default=None, help="override the run's adapter (e.g. an epoch checkpoint)")
    g.add_argument("--label", default=None)
    g.add_argument("--limit", type=int, default=None)
    s = sub.add_parser("score")
    s.add_argument("--label", required=True)
    s.add_argument("--split", default="val", choices=["train", "val", "test"])
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--split", default="val", choices=["val", "test"])
    c.add_argument("--subset", default="full", choices=["full", "q5_grounded"])
    sl = sub.add_parser("select")
    sl.add_argument("--labels", nargs="+", required=True)
    sl.add_argument("--out", required=True)
    e = sub.add_parser("epochs")
    e.add_argument("--run", required=True)
    e.add_argument("--train-output", default=None, help="defaults to outputs/<run>")
    fr = sub.add_parser("freeze")
    fr.add_argument("--runs", nargs="+", required=True)
    fr.add_argument("--out", default="configs/final_eval.yaml")
    f = sub.add_parser("final")
    f.add_argument("--final-config", default="configs/final_eval.yaml")
    f.add_argument("--rerun-reason", default=None, help="required to overwrite an existing test run (disclosed)")
    a = p.parse_args(argv)
    cfg = load_yaml(a.config)

    if a.cmd == "generate":
        run_cfg = cfg["runs"][a.run]
        adapter = a.adapter if a.adapter is not None else run_cfg["adapter"]
        label = a.label or a.run
        if run_dir(cfg, label, a.split).exists():
            if a.split == "val" and a.limit is None:
                verify_validation(cfg, label, adapter)
                print(f"reusing complete validated output: {label}")
                return 0
            raise ValueError("output label already exists; choose a new label")
        generate(cfg, a.run, a.split, adapter, label, a.limit)
        m = score(cfg, label, a.split)
        print(json.dumps({k: m["subsets"]["full"][k] for k in ("macro_task_success",)}, indent=2))
    elif a.cmd == "score":
        m = score(cfg, a.label, a.split)
        full = m["subsets"]["full"]
        print(json.dumps({k: (v["rate"] if isinstance(v, dict) and "rate" in v else v) for k, v in full.items()
                          if k != "errors"}, indent=2))
    elif a.cmd == "compare":
        r = compare(cfg, a.a, a.b, a.split, a.subset)
        print(json.dumps({t: {k: v[k] for k in ("a", "b", "diff", "ci95")} for t, v in r["by_type"].items()}, indent=2))
    elif a.cmd == "select":
        r = select(cfg, a.labels)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(r, indent=2))
    elif a.cmd == "epochs":
        manifest = json.loads((resolve(a.train_output or f"outputs/{a.run}") / "manifest.json").read_text())
        labels = []
        for ck in manifest["checkpoints"]:
            if not ck.get("epoch_end", abs(ck["epoch"] - round(ck["epoch"])) < 1e-6):
                continue
            label = f"{a.run}_step{ck['step']:06d}"
            if (run_dir(cfg, label, "val") / "metrics.json").exists():
                verify_validation(cfg, label, ck["path"])
            else:
                generate(cfg, a.run, "val", ck["path"], label)
                score(cfg, label, "val")
            labels.append(label)
        r = select(cfg, labels)
        r["checkpoints"] = {f"{a.run}_step{c['step']:06d}": c["path"] for c in manifest["checkpoints"]
                            if f"{a.run}_step{c['step']:06d}" in labels}
        out = resolve(cfg["output_dir"]) / a.run / "selection.json"
        out.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(r, indent=2))
    elif a.cmd == "freeze":
        print(json.dumps(freeze(cfg, a.runs, a.out), indent=2))
    elif a.cmd == "final":
        final = load_yaml(a.final_config)
        if not final.get("runs") or final.get("protocol") != protocol(cfg):
            raise ValueError("missing/stale final freeze; run make freeze before test")
        git = git_state()
        if git["dirty"] or not git["commit"]:
            print("refusing: final-eval requires a clean, committed tree so the frozen config is recorded (D-030)")
            return 1
        for entry in final["runs"]:
            if entry["adapter"] and entry.get("adapter_sha256") != adapter_sha256(resolve(entry["adapter"])):
                raise ValueError("frozen adapter hash changed")
            sel_path = resolve(cfg["output_dir"]) / entry["run"] / "selection.json"
            if entry["adapter"] and sel_path.exists():
                sel = json.loads(sel_path.read_text())
                # Cross-run freeze ranks all epochs together; its winner need not be
                # the within-run winner because the near-tie band is non-transitive.
                chosen_label = final["selection"]["selected"]
                chosen = sel["checkpoints"].get(chosen_label)
                if chosen is None:
                    raise ValueError("frozen checkpoint is absent from validation candidates")
                if Path(chosen).resolve() != resolve(entry["adapter"]).resolve():
                    print(f"refusing: {entry['label']} adapter {entry['adapter']} is not the selected checkpoint "
                          f"{chosen} ({sel_path})")
                    return 1
            elif entry["adapter"]:
                print(f"refusing: no {sel_path}; run `make epochs RUN={entry['run']}` first")
                return 1
        frozen = {"final_config": a.final_config, "final_config_sha256": sha256_file(resolve(a.final_config)),
                  "final_git_commit": git["commit"]}
        for entry in final["runs"]:
            out = run_dir(cfg, entry["label"], "test")
            if (out / "trajectories.jsonl").exists() and not a.rerun_reason:
                print(f"refusing to overwrite {out}: test runs once (D-030); pass --rerun-reason to disclose a rerun")
                return 1
        for entry in final["runs"]:
            generate(cfg, entry["run"], "test", entry["adapter"], entry["label"], extra_info=frozen)
            if a.rerun_reason:
                (run_dir(cfg, entry["label"], "test") / "RERUN.txt").write_text(a.rerun_reason + "\n")
            score(cfg, entry["label"], "test")
        labels = [e["label"] for e in final["runs"]]
        for other in labels[1:]:
            for subset in ("full", "q5_grounded"):
                compare(cfg, labels[0], other, "test", subset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
