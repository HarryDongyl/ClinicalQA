"""Pre-training smoke test (make smoke, PLAN section 6). Never a reported result.

Trains the smoke config on a tiny stratified batch, then checks:
  1. every logged loss is finite and the loss falls to < 50% of the first logged value (overfit)
  2. every example has supervised tokens
  3. an adapter saved to disk and reloaded gives the same loss as the in-memory model
  4. the reloaded adapter is active (disabling it raises the loss)
  5. a real greedy rollout through infer.rollout completes with well-formed trajectories
Writes <output_dir>/smoke_report.json; exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from typing import Any

from clinqa.config import load_run_config, resolve
from clinqa.data_io import load_split
from clinqa.infer import Budget, HFGenerator, rollout
from clinqa.metrics import score_example
from clinqa.modeling import device_kind, load_for_inference
from clinqa.train import per_type_loss, train


def run(cfg: dict[str, Any], n_rollout: int = 4, resume: str | None = None) -> dict[str, Any]:
    res = train(cfg, resume=resume)
    out_dir = resolve(cfg["output_dir"])
    logs = [json.loads(line) for line in (out_dir / "train_log.jsonl").read_text().splitlines()]
    losses = [row["loss"] for row in logs if "loss" in row]
    checks: dict[str, Any] = {}
    checks["finite_loss"] = bool(losses) and all(math.isfinite(x) for x in losses)
    checks["overfit"] = bool(losses) and losses[-1] < 0.5 * losses[0]
    checks["all_examples_supervised"] = all(any(lab != -100 for lab in e["labels"]) for e in res["examples"])

    trained = per_type_loss(res["model"], res["examples"], res["collator"])["all"]
    del res["model"]
    gc.collect()
    if device_kind() == "cuda":
        import torch
        torch.cuda.empty_cache()
    reloaded = load_for_inference(cfg["model"], adapter=str(resolve(cfg["checkpoint_dir"]) / "final"))
    after = per_type_loss(reloaded, res["examples"], res["collator"])["all"]
    with reloaded.disable_adapter():
        base = per_type_loss(reloaded, res["examples"], res["collator"])["all"]
    # On CUDA the reloaded model is bf16/NF4 while the trained one has fp32 norms after kbit preparation.
    checks["reload_matches"] = (abs(after - trained) <= 0.05 + 0.1 * trained if device_kind() == "cuda"
                                else abs(after - trained) <= 1e-3 * max(1.0, trained))
    checks["adapter_active"] = base > after + 0.1

    records = [r for r in load_split("train") if r["id"] in {e["id"] for e in res["examples"]}]
    if not cfg.get("example_ids"):
        records = records[:n_rollout]
    inputs = [{k: r[k] for k in ("id", "note", "table", "question")} for r in records]
    trajectories = rollout(inputs, HFGenerator(reloaded, res["tokenizer"]), res["tokenizer"], res["system"],
                           budget=Budget(), batch_size=1)
    checks["rollout_complete"] = bool(trajectories) and all(t["stop_reason"] == "answer" for t in trajectories)
    scores = [score_example(r, t).as_dict() for r, t in zip(records, trajectories)]
    if cfg.get("require_cuda"):
        tool_scores = [s for s in scores if s["answer_type"] == "tool_call"]
        checks["cuda_nf4"] = res["manifest"]["quantized_4bit"] and device_kind() == "cuda"
        checks["four_types_covered"] = len({r["answer_type"] for r in records}) == 4
        checks["both_tools_covered"] = {c["tool"] for r in records for c in r.get("tool_calls", [])} == {
            "calculate_bmi", "unit_convert"}
        checks["tool_arguments_and_execution"] = bool(tool_scores) and all(
            s["tool_selected"] and s["tool_args_correct"] and s["tool_executed"] and s["result_in_answer"]
            for s in tool_scores)
    with (out_dir / "trajectories.jsonl").open("w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    report = {"checks": checks, "passed": all(checks.values()),
              "loss": {"first_logged": losses[0] if losses else None, "last_logged": losses[-1] if losses else None,
                       "trained_eval": trained, "reloaded_eval": after, "base_eval": base},
              "rollout": [{"id": s["id"], "error": s["error"], "stop": t["stop_reason"],
                           "final_answer": (t["final_answer"] or "")[:200]} for s, t in zip(scores, trajectories)],
              "manifest": {k: res["manifest"][k] for k in ("runtime_s", "tokens_per_s", "peak_vram_gb",
                                                           "trainable_params", "precision", "quantized_4bit")}}
    (out_dir / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tiny overfit + adapter reload + rollout smoke test.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args(argv)
    report = run(load_run_config(args.config), resume=args.resume)
    print(json.dumps({"checks": report["checks"], "loss": report["loss"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
