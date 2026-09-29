"""Exercise real Trainer/PEFT recovery on a tiny random CPU model (no downloads)."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")


def test_real_trainer_resume_preserves_optimizer_state_and_refuses_config_drift(tmp_path, monkeypatch):
    from transformers import GPT2Config, GPT2LMHeadModel
    from clinqa import train as module

    class Tokenizer:
        pad_token_id = 0
        chat_template = "fixture-template"

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer_config.json").write_text('{}')

    records = [{"id": f"train_{i}"} for i in range(4)]
    examples = [{"id": r["id"], "answer_type": "extractive", "input_ids": [1, 2, 3, 4],
                 "labels": [-100, -100, 3, 4]} for r in records]
    prompt = tmp_path / "system.txt"
    prompt.write_text("fixture")
    monkeypatch.setattr(module, "load_yaml", lambda _: {"system_prompt": str(prompt), "max_length": 8})
    monkeypatch.setattr(module, "audit_train_view", lambda c, f: (records, {"train_count": 4}))
    monkeypatch.setattr(module, "load_tokenizer", lambda _: Tokenizer())
    monkeypatch.setattr(module, "encode_records", lambda *a, **k: examples)
    monkeypatch.setattr(module, "data_hashes", lambda _: {"fixture": "unchanged"})
    monkeypatch.setattr(module, "compute_dtype", lambda _: torch.float32)

    def load(*a, **k):
        model = GPT2LMHeadModel(GPT2Config(vocab_size=16, n_positions=8, n_embd=8, n_layer=1, n_head=1,
                                         resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
        model.config.use_cache = False
        return model

    monkeypatch.setattr(module, "load_base_model", load)
    cfg = {"run_id": "fixture", "seed": 42, "format_config": "fixture", "model": {}, "train_view": "fixture",
           "output_dir": str(tmp_path / "outputs"), "checkpoint_dir": str(tmp_path / "checkpoints"),
           "val_loss": False, "lora": {"r": 2, "alpha": 4, "dropout": 0, "target_modules": ["c_attn"]},
           "training": {"epochs": 1, "micro_batch": 1, "grad_accum": 1, "learning_rate": 1e-3,
                        "scheduler": "constant", "warmup_ratio": 0, "weight_decay": 0,
                        "max_grad_norm": 1, "gradient_checkpointing": False, "logging_steps": 1, "save_steps": 2}}
    original_callback = module.make_callback

    def interrupt_after_save(*args, **kwargs):
        callback = original_callback(*args, **kwargs)
        save = callback.on_save

        def interrupt(*a, **k):
            save(*a, **k)
            raise RuntimeError("simulated process interruption")
        callback.on_save = interrupt
        return callback

    monkeypatch.setattr(module, "make_callback", interrupt_after_save)
    with pytest.raises(RuntimeError, match="simulated"):
        module.train(cfg)
    ck = Path(cfg["checkpoint_dir"]) / "checkpoint-2"
    assert all((ck / f).exists() for f in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"))
    monkeypatch.setattr(module, "make_callback", original_callback)
    altered = dict(cfg, seed=43)
    with pytest.raises(ValueError, match="contract mismatch"):
        module.train(altered, resume="latest")
    result = module.train(cfg, resume="latest")["manifest"]
    assert result["optimizer_steps"] == 4
    assert [c["step"] for c in result["checkpoints"]] == [2, 4]
    assert result["checkpoints"][-1]["epoch_end"]
    assert result["tokens_per_s"] is None  # do not overstate resumed throughput
    with pytest.raises(ValueError, match="already completed"):
        module.train(cfg, resume="latest")
