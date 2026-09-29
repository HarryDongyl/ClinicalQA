"""Data/format checks on CPU; optional RunPod CUDA environment gate."""
from __future__ import annotations

import argparse
import json
import shutil

from clinqa.config import PROJECT_ROOT, load_run_config, load_yaml, resolve
from clinqa.training_data import audit_train_view, grounding_flags
from clinqa.run_info import hardware, package_versions


def inspect_data() -> dict:
    reports = {}
    for name in ("raw_lr1e4", "raw_lr5e5", "q5filtered_lr5e5"):
        cfg = load_run_config(f"configs/train/{name}.yaml")
        records, report = audit_train_view(cfg, load_yaml(cfg["format_config"]))
        reports[name] = report
    report_path = resolve("reports/training_data_gate.json")
    report_path.write_text(json.dumps(reports, indent=2) + "\n")
    lines = ["# Training supervision policy", "", "Generated from train only; canonical splits and targets are unchanged.",
             "", "Q5 filtering is a heuristic quarantine, not clinical adjudication. No missing gold measurements are inserted into inputs.",
             "", "| Run | Train rows | Excluded | Remaining Q5 |", "|---|---:|---:|---:|"]
    for name, report in reports.items():
        lines.append(f"| {name} | {report['train_count']} | {len(report['excluded_ids'])} | {len(report['remaining_q5_ids'])} |")
    lines += ["", "Raw controls explicitly retain unsupported targets. The q5filtered run is the mitigated candidate.",
              "Compare raw_lr1e4 with raw_lr5e5 for LR; compare raw_lr5e5 with q5filtered_lr5e5 for filtering.",
              "Filtering changes sample count, type distribution and optimizer-step count; it is a data-policy ablation, not a pure LR comparison.",
              "", "## Quarantined train candidates", ""]
    cfg = load_run_config("configs/train/raw_lr1e4.yaml")
    raw, _ = audit_train_view(cfg, load_yaml(cfg["format_config"]))
    for flag in grounding_flags(raw):
        lines.append(f"- `{flag.id}`: {flag.detail}")
    resolve("reports/TRAINING_DATA_POLICY.md").write_text("\n".join(lines) + "\n")
    return reports


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args(argv)
    reports = inspect_data()
    print(json.dumps({k: {f: v[f] for f in ("train_count", "policy")} for k, v in reports.items()}, indent=2))
    if not args.gpu:
        return 0
    import torch
    import bitsandbytes  # noqa: F401
    from clinqa.modeling import compute_dtype
    from clinqa.formatting import load_tokenizer, prompt_messages, build_conversation, render, encode, template_sha256

    info = hardware()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("RunPod recipe requires exactly one visible CUDA GPU")
    if info["gpu_memory_gb"] < 20:
        raise ValueError("this 24GB recipe requires >=20 GiB visible VRAM; do not silently change the experiment")
    if shutil.disk_usage(PROJECT_ROOT).free < 25 * 2**30:
        raise ValueError("less than 25 GiB free on workspace; increase persistent storage")
    cfg = load_run_config("configs/train/q5filtered_lr5e5.yaml")
    fmt = load_yaml(cfg["format_config"])
    tok = load_tokenizer(fmt)
    system = resolve(fmt["system_prompt"]).read_text().strip()
    records, _ = audit_train_view(cfg, fmt)
    longest = 0
    for record in records:
        conversation = build_conversation(record, system)
        encoded = encode(tok, conversation, max_length=fmt["max_length"])
        prompt = render(tok, prompt_messages(record, system), add_generation_prompt=True)
        prefix_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        if encoded.input_ids[:len(prefix_ids)] != prefix_ids or encoded.n_supervised == 0:
            raise ValueError(f"{record['id']}: train/inference token prefix or labels disagree")
        longest = max(longest, len(encoded.input_ids))
    dtype = compute_dtype(cfg["model"])
    # Actually exercise a CUDA operation; model-specific NF4 forward/backward is checked by gpu-smoke.
    x = torch.ones((16, 16), device="cuda", dtype=dtype)
    assert torch.isfinite(x @ x).all()
    info.update(packages=package_versions(), compute_dtype=str(dtype), max_train_tokens=longest,
                chat_template_sha256=template_sha256(tok), training_gpu_smoke_still_required=True)
    resolve("outputs").mkdir(exist_ok=True)
    resolve("outputs/runpod_preflight.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
