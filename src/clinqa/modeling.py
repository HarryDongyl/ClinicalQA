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
    tok.clinqa_call_format = model_cfg.get("tool_call_format", "json")  # Qwen3.5 emits XML calls (D-088)
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
    model, info = AutoModelForCausalLM.from_pretrained(model_cfg["name"], output_loading_info=True, **kwargs)
    # A checkpoint whose keys do not map onto the model (e.g. a multimodal Qwen3.5 checkpoint loaded text-only)
    # would otherwise leave layers randomly initialised with only a warning (D-088).
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise ValueError(f"{model_cfg['name']}: weights not loaded for {len(info.get('missing_keys', []))} "
                         f"parameters, e.g. {sorted(info.get('missing_keys', []))[:5]}; "
                         f"mismatched {info.get('mismatched_keys', [])[:5]}")
    model.config.use_cache = not for_training
    # What was actually applied (nf4 is requested but only honoured on CUDA).
    model.clinqa_load_info = {"device": kind, "quantization": "nf4" if "quantization_config" in kwargs else "none",
                              "dtype": str(kwargs["dtype"]).replace("torch.", ""),
                              "ignored_checkpoint_keys": len(info.get("unexpected_keys", [])),
                              "linear_attention_kernel": _linear_attention_kernel(model),
                              "linear_attention_ops": _linear_attention_ops(model)}
    return model


def _linear_attention_kernel(model: Any) -> str | None:
    """Module providing Gated DeltaNet kernels (fla or the transformers torch fallback); None without linear attention."""
    import sys

    module = sys.modules.get(type(model).__module__)
    fn = getattr(module, "torch_chunk_gated_delta_rule", None)  # rebound to the fla kernel when fla imports
    return getattr(fn, "__module__", None) if fn is not None else None


def _linear_attention_ops(model: Any) -> dict[str, str | None] | None:
    """Module of each op a Gated DeltaNet layer actually holds (D-106).

    `_linear_attention_kernel` reads a module-level fallback name and therefore always reports the Transformers module.
    Each layer binds its ops at construction (fla / causal-conv1d when importable, torch fallbacks otherwise), so the
    first layer that has them shows what really runs. None without linear attention.
    """
    names = ("chunk_gated_delta_rule", "recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update")
    for module in model.modules():
        if any(hasattr(module, n) for n in names):
            return {n: getattr(getattr(module, n, None), "__module__", None) for n in names}
    return None


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
