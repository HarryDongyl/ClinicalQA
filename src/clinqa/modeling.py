"""Model/tokenizer loading shared by training and inference (D-020, D-022, D-023).

The tokenizer and chat template come from the configured template model: Qwen3-4B-Instruct-2507
for the 4B runs and the 0.6B smoke model (Qwen3 sizes share one vocabulary), and Qwen3-8B's own
template for the wave-three 8B arms (D-077). `chat_template_kwargs` from the model config (e.g.
enable_thinking: false) are attached to the tokenizer and applied by every render. On CUDA the
base model is 4-bit NF4 (QLoRA); on CPU/MPS it is loaded unquantized for smoke tests only.
"""

from __future__ import annotations

from typing import Any


def device_kind() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_tokenizer(model_cfg: dict[str, Any], padding_side: str = "right") -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_cfg["tokenizer"], revision=model_cfg["tokenizer_revision"])
    tok.padding_side = padding_side
    tok.clinqa_template_kwargs = dict(model_cfg.get("chat_template_kwargs") or {})
    if tok.pad_token is None:
        tok.pad_token = "<|endoftext|>"
    return tok


def load_base_model(model_cfg: dict[str, Any], for_training: bool) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    kind = device_kind()
    dtype = compute_dtype(model_cfg)
    kwargs: dict[str, Any] = {"revision": model_cfg["revision"], "attn_implementation": model_cfg.get("attn", "sdpa")}
    if kind == "cuda" and model_cfg.get("quantization") == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype)
        kwargs["dtype"] = dtype
        kwargs["device_map"] = {"": 0}
    elif kind == "cuda":
        kwargs["dtype"] = dtype
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], **kwargs)
    model.config.use_cache = not for_training
    # What was actually applied (nf4 is requested but only honoured on CUDA).
    model.clinqa_load_info = {"device": kind, "quantization": "nf4" if "quantization_config" in kwargs else "none",
                              "dtype": str(kwargs["dtype"]).replace("torch.", "")}
    return model


def compute_dtype(model_cfg: dict[str, Any]) -> Any:
    import torch

    if device_kind() != "cuda":
        return torch.float32
    requested = model_cfg.get("compute_dtype", "auto")
    if requested not in {"auto", "bf16", "fp16"}:
        raise ValueError(f"unsupported compute_dtype: {requested}")
    supported = torch.cuda.is_bf16_supported()
    if requested == "bf16" and not supported:
        raise ValueError("BF16 requested but not supported by this GPU")
    return torch.bfloat16 if supported and requested != "fp16" else torch.float16


def load_for_inference(model_cfg: dict[str, Any], adapter: str | None) -> Any:
    model = load_base_model(model_cfg, for_training=False)
    if adapter:
        from peft import PeftModel

        info = model.clinqa_load_info
        model = PeftModel.from_pretrained(model, adapter)
        model.clinqa_load_info = info
    model.eval()
    return model
