"""Mask audit + formatted examples (make audit-masks).

Writes:
  reports/formatted_examples.md  exact model input (rendered prompt) and expected output for six
                                 examples: extractive, numeric, uncertain, metric BMI, conversion,
                                 imperial BMI (assignment deliverable 3, PLAN section 4)
  reports/mask_audit.md          the same conversations with supervised spans marked, plus
                                 whole-split mask statistics
Fails (exit 1) if any conversation has an all-masked assistant turn or a span whose decoded
tokens differ from the assistant text.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from clinqa.analysis.features import compute_features
from clinqa.config import load_yaml, resolve
from clinqa.formatting import (
    END_OF_TURN,
    IGNORE_INDEX,
    build_conversation,
    encode,
    load_tokenizer,
    prompt_messages,
    render,
    template_sha256,
)
from clinqa.data_io import read_jsonl

SUP_OPEN, SUP_CLOSE = "⟦", "⟧"


def _bmi_units(r: dict[str, Any]) -> set[str]:
    return {m.unit for m in compute_features("train", r).note_measurements}


def select_examples(records: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Deterministic: first train record (by file order) matching each category."""
    def tool(r: dict[str, Any]) -> str | None:
        return r["tool_calls"][0]["tool"] if r.get("tool_calls") else None

    rules = [
        ("Extractive", lambda r: r["answer_type"] == "extractive"),
        ("Numeric reasoning", lambda r: r["answer_type"] == "numeric_reasoning"),
        ("Uncertain", lambda r: r["answer_type"] == "uncertain"),
        ("Tool call: BMI from metric note values",
         lambda r: tool(r) == "calculate_bmi" and _bmi_units(r) >= {"kg", "cm"} and not _bmi_units(r) & {"lb", "in"}),
        ("Tool call: unit conversion", lambda r: tool(r) == "unit_convert"),
        ("Tool call: BMI from imperial note values (implicit lb/in -> kg/cm)",
         lambda r: tool(r) == "calculate_bmi" and _bmi_units(r) >= {"lb", "in"}),
    ]
    return [(title, next(r for r in records if pred(r))) for title, pred in rules]


def _marked(enc: Any) -> str:
    out, pos = [], 0
    for s, e in enc.char_spans:
        out += [enc.text[pos:s], SUP_OPEN, enc.text[s:e], SUP_CLOSE]
        pos = e
    out.append(enc.text[pos:])
    return "".join(out)


def _expected_spans(messages: list[dict[str, Any]], tok: Any) -> list[str]:
    """Assistant text as the template renders it, reconstructed independently of the span finder."""
    out = []
    for m in messages:
        if m["role"] != "assistant":
            continue
        if m.get("tool_calls"):
            fn = m["tool_calls"][0]["function"]
            out.append('<tool_call>\n{"name": "' + fn["name"] + '", "arguments": '
                       + json.dumps(fn["arguments"], ensure_ascii=False) + "}\n</tool_call>" + END_OF_TURN)
        else:
            out.append(m["content"] + END_OF_TURN)
    return out


def audit_record(tok: Any, messages: list[dict[str, Any]]) -> tuple[Any, list[str]]:
    enc = encode(tok, messages)
    problems = []
    got = [enc.text[s:e] for s, e in enc.char_spans]
    if got != _expected_spans(messages, tok):
        problems.append("span text differs from assistant messages")
    sup_ids = [i for i, lab in zip(enc.input_ids, enc.labels) if lab != IGNORE_INDEX]
    if tok.decode(sup_ids) != "".join(got):
        problems.append("supervised tokens do not decode to the span text")
    offsets = tok(enc.text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    for s, e in enc.char_spans:
        if not any(lab != IGNORE_INDEX for lab, (a, b) in zip(enc.labels, offsets) if a >= s and b <= e):
            problems.append("all-masked assistant turn")
    return enc, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/format_core.yaml")
    parser.add_argument("--view", default="raw")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args(argv)
    cfg = load_yaml(args.config)
    system = resolve(cfg["system_prompt"]).read_text(encoding="utf-8").strip()
    tok = load_tokenizer(cfg)
    records = read_jsonl(resolve(cfg["train_views"][args.view]))

    # Whole-split statistics.
    problems: dict[str, list[str]] = {}
    sup_counts, total = [], []
    for r in records:
        enc, p = audit_record(tok, build_conversation(r, system))
        if p:
            problems[r["id"]] = p
        sup_counts.append(enc.n_supervised)
        total.append(len(enc.input_ids))

    examples = select_examples(records)
    header = [f"Tokenizer `{cfg['tokenizer']}@{cfg['tokenizer_revision'][:12]}`, "
              f"chat template sha256 `{template_sha256(tok)[:16]}`, system prompt `{cfg['system_prompt']}`. "
              "Generated by `make audit-masks`; do not edit by hand.", ""]

    fx = ["# Formatted SFT examples", "", *header,
          "Each example shows the exact text the model receives (rendered with the native Qwen chat template, "
          "tool schemas included, ending in the generation prompt) and the exact supervised continuation. "
          "For tool examples the continuation contains the tool response turn, which the runtime supplies "
          "from the real executor and which is masked from the loss.", ""]
    ma = ["# Assistant-only loss mask audit", "", *header,
          f"Supervised spans are wrapped in `{SUP_OPEN} {SUP_CLOSE}`; everything outside is label -100.", "",
          "## Whole-split check", "",
          f"- view `{args.view}`: {len(records)} conversations, {len(problems)} with problems",
          f"- total tokens: min {min(total)}, max {max(total)}",
          f"- supervised tokens per conversation: min {min(sup_counts)}, max {max(sup_counts)}, "
          f"mean {sum(sup_counts) / len(sup_counts):.1f}",
          f"- supervised share of all tokens: {sum(sup_counts) / sum(total):.1%}", ""]
    if problems:
        ma += ["Problems:", "", "```json", json.dumps(problems, indent=2), "```", ""]

    for title, r in examples:
        msgs = build_conversation(r, system)
        prompt = render(tok, prompt_messages(r, system), add_generation_prompt=True)
        full = render(tok, msgs)
        enc, p = audit_record(tok, msgs)
        fx += [f"## {title} — `{r['id']}` ({r['answer_type']})", "",
               "**Model input** (rendered prompt):", "", "````text", prompt, "````", "",
               "**Expected output** (continuation; assistant turns supervised):", "",
               "````text", full[len(prompt):], "````", ""]
        ma += [f"## {title} — `{r['id']}`", "",
               f"{len(enc.input_ids)} tokens, {enc.n_supervised} supervised, "
               f"{len(enc.char_spans)} assistant span(s), problems: {p or 'none'}", "",
               "````text", _marked(enc)[len(prompt) - len("<|im_start|>assistant\n"):], "````", ""]
    ma += ["(Only the tail from the first assistant header is shown; the prompt above it is fully masked.)", ""]

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "formatted_examples.md").write_text("\n".join(fx), encoding="utf-8")
    (output_dir / "mask_audit.md").write_text("\n".join(ma), encoding="utf-8")
    print(f"audited {len(records)} conversations; problems: {len(problems)}; "
          f"wrote audits to {output_dir}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
