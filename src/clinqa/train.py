"""SFT training: plain transformers Trainer + PEFT LoRA/QLoRA over our own labels (D-021, D-022).

    uv run python -m clinqa.train --config configs/train_raw.yaml

Writes:
  <checkpoint_dir>/checkpoint-<step>/   adapter per epoch (gitignored; pushed to HF Hub by hand, D-018)
  <output_dir>/manifest.json            resolved config, hashes, versions, hardware, runtime, checkpoints
  <output_dir>/train_log.jsonl          loss / lr / grad norm / step time / peak VRAM every logging step,
                                        and per-answer-type teacher-forced val loss at each epoch end
  <output_dir>/tb/                      the same scalars as TensorBoard events
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from clinqa.config import load_run_config, load_yaml, resolve
from clinqa.data_io import load_split, read_jsonl
from clinqa.formatting import IGNORE_INDEX, build_conversation, encode_segments, template_kwargs, template_sha256
from clinqa.modeling import compute_dtype, device_kind, load_base_model, load_tokenizer
from clinqa.run_info import adapter_sha256, data_hashes, git_state, hardware, package_versions, sha256_text
from clinqa.schemas import tool_schemas_sha256
from clinqa.seed import set_seed
from clinqa.training_data import audit_train_view
from clinqa.checkpoints import discover, resume_path

ANSWER_TYPES = ("extractive", "numeric_reasoning", "tool_call", "uncertain")


def stratified_head(records: list[dict[str, Any]], n: int | None) -> list[dict[str, Any]]:
    """First n records keeping answer types balanced (file order within type); None keeps everything."""
    if n is None:
        return records
    per = max(1, n // len(ANSWER_TYPES))
    out, seen = [], defaultdict(int)
    for r in records:
        if seen[r["answer_type"]] < per:
            out.append(r)
            seen[r["answer_type"]] += 1
    return out


def encode_records(records: list[dict[str, Any]], tok: Any, system: str, max_length: int,
                   segmented: bool = False) -> list[dict[str, Any]]:
    """One item per record. An item holds one sequence, or one per turn for segmented templates (D-077)."""
    out = []
    for r in records:
        segs = encode_segments(tok, build_conversation(r, system), max_length=max_length, segmented=segmented)
        out.append({"id": r["id"], "answer_type": r["answer_type"],
                    "segments": [{"input_ids": e.input_ids, "labels": e.labels} for e in segs]})
    return out


def n_input_tokens(examples: list[dict[str, Any]]) -> int:
    return sum(len(seg["input_ids"]) for e in examples for seg in e["segments"])


class Collator:
    """Right-pad input_ids/labels; padding is masked from attention and loss.

    Every segment of every item in the micro-batch becomes one row. The Trainer normalises the
    loss by the supervised-token count of the whole accumulated batch, so a record split into
    segments contributes exactly its assistant tokens, as in one sequence; optimizer steps are
    still counted per record.
    """

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        batch = [seg for b in batch for seg in b["segments"]]
        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [self.pad_id] * (n - len(b["input_ids"])) for b in batch]
        labels = [b["labels"] + [IGNORE_INDEX] * (n - len(b["labels"])) for b in batch]
        mask = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(mask)}


def per_type_loss(model: Any, examples: list[dict[str, Any]], collator: Collator) -> dict[str, float]:
    """Token-weighted teacher-forced loss on assistant tokens, by answer type (diagnostic only)."""
    import torch

    from contextlib import nullcontext

    sums, counts = defaultdict(float), defaultdict(int)
    was_training = model.training
    model.eval()
    # Match training numerics on GPU (the Trainer runs forward passes under bf16 autocast).
    dtype_name = getattr(model, "clinqa_load_info", {}).get("dtype", "")
    amp_dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    amp = torch.autocast("cuda", dtype=amp_dtype) if torch.cuda.is_available() else nullcontext()
    with torch.no_grad(), amp:
        for ex in examples:
            for seg in ex["segments"]:
                batch = {k: v.to(model.device) for k, v in collator([{"segments": [seg]}]).items()}
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                shift_logits, shift_labels = logits[0, :-1].float(), batch["labels"][0, 1:]
                keep = shift_labels != IGNORE_INDEX
                loss = torch.nn.functional.cross_entropy(shift_logits[keep], shift_labels[keep], reduction="sum")
                for key in (ex["answer_type"], "all"):
                    sums[key] += loss.item()
                    counts[key] += int(keep.sum())
    if was_training:
        model.train()
    return {k: sums[k] / counts[k] for k in sorted(sums)}


def make_callback(log_path: Path, tb_dir: Path, val_examples: list[dict[str, Any]] | None, collator: Collator,
                  checkpoints: list[dict[str, Any]], tokenizer: Any) -> Any:
    from transformers import TrainerCallback

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(tb_dir))
    except ImportError:  # tensorboard is optional at runtime
        writer = None

    class RunLogger(TrainerCallback):
        def __init__(self) -> None:
            self.t_last = time.perf_counter()
            self.step_last = 0

        def _write(self, row: dict[str, Any]) -> None:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if writer:
                for k, v in row.items():
                    if isinstance(v, (int, float)) and k not in ("step", "epoch"):
                        writer.add_scalar(k, v, row["step"])
                writer.flush()

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kw: Any) -> None:
            import torch

            now = time.perf_counter()
            steps = max(1, state.global_step - self.step_last)
            row = {"step": state.global_step, "epoch": round(state.epoch or 0, 4), **(logs or {}),
                   "sec_per_step": round((now - self.t_last) / steps, 3)}
            if torch.cuda.is_available():
                row["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            self.t_last, self.step_last = now, state.global_step
            self._write(row)

        def on_epoch_end(self, args: Any, state: Any, control: Any, model: Any = None, **kw: Any) -> None:
            control.should_save = True  # keep epoch checkpoints in addition to recovery checkpoints
            if val_examples and model is not None:
                losses = per_type_loss(model, val_examples, collator)
                self._write({"step": state.global_step, "epoch": round(state.epoch or 0, 4),
                             **{f"val_loss/{k}": v for k, v in losses.items()}})

        def on_save(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
            path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            tokenizer.save_pretrained(str(path))  # adapter + tokenizer/template travel together
            checkpoints.append({"step": state.global_step, "epoch": round(state.epoch or 0, 4), "path": str(path),
                                "adapter_sha256": adapter_sha256(path)})

    return RunLogger()


def train(cfg: dict[str, Any], resume: str | None = None) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments

    t_start = time.perf_counter()
    if cfg.get("require_cuda") and not torch.cuda.is_available():
        raise ValueError("this run requires a CUDA GPU; use configs/smoke.yaml for CPU smoke")
    set_seed(cfg["seed"])
    fmt = load_yaml(cfg["format_config"])
    records, data_policy = audit_train_view(cfg, fmt)
    if cfg.get("example_ids"):
        by_id = {r["id"]: r for r in records}
        records = [by_id[i] for i in cfg["example_ids"]]
    else:
        records = stratified_head(records, cfg.get("max_examples"))
    system = resolve(fmt["system_prompt"]).read_text(encoding="utf-8").strip()
    for key in ("tokenizer", "tokenizer_revision"):
        if fmt.get(key) != cfg["model"].get(key):
            raise ValueError(f"format config {key} {fmt.get(key)!r} differs from the model's {cfg['model'].get(key)!r}")
    if dict(fmt.get("chat_template_kwargs") or {}) != dict(cfg["model"].get("chat_template_kwargs") or {}):
        raise ValueError("chat_template_kwargs differ between the format config and the model config")
    segmented = bool(fmt.get("segmented_turns"))
    tok = load_tokenizer(cfg["model"])
    train_examples = encode_records(records, tok, system, fmt["max_length"], segmented)
    val_examples = (encode_records(load_split("val", fmt["data_config"]), tok, system, fmt["max_length"], segmented)
                    if cfg.get("val_loss") else None)

    out_dir, ckpt_dir = resolve(cfg["output_dir"]), resolve(cfg["checkpoint_dir"])
    checkpoint = resume_path(ckpt_dir, resume)
    contract_path = out_dir / "run_contract.json"
    from clinqa.config import PROJECT_ROOT
    from clinqa.run_info import sha256_file

    contract = {"config": cfg, "data_policy": data_policy, "packages": package_versions(),
                "system_prompt_sha256": sha256_text(system), "chat_template_sha256": template_sha256(tok),
                "chat_template_kwargs": template_kwargs(tok), "segmented_turns": segmented,
                "tool_schemas_sha256": tool_schemas_sha256(),
                "source_hashes": {str(p.relative_to(PROJECT_ROOT)): sha256_file(p)
                                  for p in sorted((PROJECT_ROOT / "src/clinqa").rglob("*.py"))}}
    if checkpoint:
        if not contract_path.exists() or json.loads(contract_path.read_text()) != contract:
            raise ValueError("resume contract mismatch: config, data, code and packages must match the original run")
        if (out_dir / "manifest.json").exists():
            raise ValueError("run already completed; use a new run_id for a new experiment")
    elif (out_dir.exists() and any(out_dir.iterdir())) or (ckpt_dir.exists() and any(ckpt_dir.iterdir())):
        raise ValueError("run artifacts already exist; use --resume latest or a new run_id; refusing overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    (out_dir / "data_policy.json").write_text(json.dumps(data_policy, indent=2) + "\n")
    log_path = out_dir / "train_log.jsonl"
    if not checkpoint:
        log_path.write_text("", encoding="utf-8")

    t_load = time.perf_counter()
    model = load_base_model(cfg["model"], for_training=True)
    tr = cfg["training"]
    quantized = getattr(model, "is_loaded_in_4bit", False)
    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=tr["gradient_checkpointing"])
    elif tr["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                             target_modules=lora["target_modules"], bias="none",
                                             task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    load_s = time.perf_counter() - t_load

    steps_per_epoch = math.ceil(len(train_examples) / (tr["micro_batch"] * tr["grad_accum"]))
    total_steps = steps_per_epoch * tr["epochs"]
    dtype = compute_dtype(cfg["model"])
    bf16 = dtype == torch.bfloat16
    fp16 = dtype == torch.float16
    args = TrainingArguments(
        output_dir=str(ckpt_dir), seed=cfg["seed"], data_seed=cfg["seed"],
        num_train_epochs=tr["epochs"], per_device_train_batch_size=tr["micro_batch"],
        gradient_accumulation_steps=tr["grad_accum"], learning_rate=tr["learning_rate"],
        lr_scheduler_type=tr["scheduler"], warmup_steps=math.ceil(tr["warmup_ratio"] * total_steps),
        max_grad_norm=tr["max_grad_norm"], weight_decay=tr["weight_decay"], bf16=bf16, fp16=fp16,
        gradient_checkpointing=False,  # enabled on the model above (kbit-aware)
        logging_steps=tr["logging_steps"], logging_first_step=True,
        save_strategy="steps", save_steps=tr.get("save_steps", 50), save_only_model=False,
        eval_strategy="no", report_to="none", remove_unused_columns=False, dataloader_num_workers=0,
    )
    collator = Collator(tok.pad_token_id)
    checkpoints: list[dict[str, Any]] = []
    trainer = Trainer(model=model, args=args, train_dataset=train_examples, data_collator=collator,
                      callbacks=[make_callback(log_path, out_dir / "tb", val_examples, collator, checkpoints, tok)])
    t_train = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    train_s = time.perf_counter() - t_train
    trainer.save_model(str(ckpt_dir / "final"))
    tok.save_pretrained(str(ckpt_dir / "final"))

    n_tokens = n_input_tokens(train_examples) * tr["epochs"]
    manifest = {
        "run_id": cfg["run_id"], "config": cfg, "git": git_state(), "packages": package_versions(),
        "hardware": hardware(), "precision": "bf16" if bf16 else ("fp16" if fp16 else "fp32"), "quantized_4bit": quantized,
        "data_policy": data_policy, "resume_from_checkpoint": str(checkpoint) if checkpoint else None,
        "data": data_hashes(cfg["train_view"]), "n_train_examples": len(train_examples),
        "train_ids_sha256": sha256_text("\n".join(e["id"] for e in train_examples)),
        "system_prompt_sha256": sha256_text(system), "chat_template_sha256": template_sha256(tok),
        "chat_template_kwargs": template_kwargs(tok), "segmented_turns": segmented,
        "n_sequences": sum(len(e["segments"]) for e in train_examples),
        "supervised_tokens": sum(sum(x != IGNORE_INDEX for x in seg["labels"])
                                 for e in train_examples for seg in e["segments"]),
        "tool_schemas_sha256": tool_schemas_sha256(), "model_load": getattr(model, "clinqa_load_info", None),
        "final_adapter_sha256": adapter_sha256(ckpt_dir / "final"),
        "trainable_params": trainable, "total_params": total,
        "steps_per_epoch": steps_per_epoch, "optimizer_steps": result.global_step,
        "final_train_loss": result.training_loss, "checkpoints": discover(ckpt_dir),
        "runtime_s": {"model_load": round(load_s, 1), "train": round(train_s, 1),
                      "total": round(time.perf_counter() - t_start, 1)},
        "tokens_per_s": round(n_tokens / train_s, 1) if checkpoint is None else None,
        "runtime_scope": "current process only; resumed time excludes earlier processes",
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"manifest": manifest, "model": model, "tokenizer": tok, "examples": train_examples,
            "collator": collator, "system": system}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA SFT on formatted conversations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="latest or a complete checkpoint path; same run/config only")
    args = parser.parse_args(argv)
    m = train(load_run_config(args.config), resume=args.resume)["manifest"]
    print(json.dumps({k: m[k] for k in ("run_id", "optimizer_steps", "final_train_loss", "runtime_s",
                                        "tokens_per_s", "peak_vram_gb", "checkpoints")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
