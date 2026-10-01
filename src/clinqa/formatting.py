"""SFT formatting: records -> native Qwen chat conversations + assistant-only loss labels.

Conversation contract (D-024):

    system    system prompt (the chat template appends the tool schemas)
    user      note / table / question sections
    assistant final answer                          (extractive, numeric, uncertain)
      or
    assistant tool call, no text                    (tool_call)
    tool      {"result": x} from the real executor
    assistant final answer

answer_type, gold tool calls and quality metadata never enter the conversation; they
go to a sidecar file joined by id at scoring time.

Loss labels (D-021): each assistant span is located by prefix-diff rendering (render
the conversation up to the turn with a generation prompt, then including the turn)
and must align exactly with token boundaries. The generation prompt
`<|im_start|>assistant\\n` is masked; the assistant content and its `<|im_end|>` are
supervised; everything else is -100.

Templates that render an earlier assistant turn differently once a later turn exists
(Qwen3-8B with enable_thinking=False adds an empty think block only to the turn being
generated) cannot supervise every turn inside one sequence. With `segmented_turns: true`
in the format config, `encode_segments` supervises such a turn in its own sequence,
rendered exactly as at inference (D-077). Without it the old single-sequence contract
is enforced, so the 4B rendering and labels are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clinqa.config import load_yaml, resolve
from clinqa.data_io import load_split, read_jsonl
from clinqa.schemas import TOOL_SCHEMAS, tool_response_content, tool_schemas_sha256
from clinqa.tools import execute_tool

IGNORE_INDEX = -100
END_OF_TURN = "<|im_end|>"


class FormattingError(ValueError):
    pass


# ---------------------------------------------------------------- text rendering


def _cell(text: str) -> str:
    return text.replace("|", "\\|").strip()


def render_table(table: dict[str, Any]) -> str:
    """Markdown table with every row; an empty unit is shown explicitly (P-002)."""
    headers = table["headers"]
    unit_col = headers.index("Unit") if "Unit" in headers else None
    lines = ["| " + " | ".join(_cell(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in table["rows"]:
        cells = [_cell(c) for c in row]
        if unit_col is not None and unit_col < len(cells) and cells[unit_col] == "":
            cells[unit_col] = "(no unit)"
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def user_content(record: dict[str, Any]) -> str:
    table = record["table"]
    return (f"## Encounter note\n{record['note'].strip()}\n\n"
            f"## Table ({table['type']})\n{render_table(table)}\n\n"
            f"## Question\n{record['question'].strip()}")


def prompt_messages(record: dict[str, Any], system_prompt: str,
                    demos: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Exactly what the model sees at inference time before its first turn.

    `demos` are fixed few-shot turns inserted between the system prompt and the question
    (inference-only baselines, D-078); training never passes them.
    """
    return ([{"role": "system", "content": system_prompt}] + list(demos or [])
            + [{"role": "user", "content": user_content(record)}])


def assistant_call_message(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]}


def tool_message(result: float | str) -> dict[str, Any]:
    return {"role": "tool", "content": tool_response_content(result)}


def target_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Supervised continuation. Tool results come from the executor, checked against gold."""
    if record["answer_type"] != "tool_call":
        return [{"role": "assistant", "content": record["answer"].strip()}]
    calls = record.get("tool_calls") or []
    if len(calls) != 1:
        raise FormattingError(f"{record['id']}: expected exactly one gold tool call, got {len(calls)}")
    call = calls[0]
    args = dict(call["arguments"])
    if call["tool"] == "unit_convert":
        args.setdefault("substance", None)  # D-024: substance always emitted
    result = execute_tool(call["tool"], args)
    if result != call["result"]:
        raise FormattingError(f"{record['id']}: executor result {result!r} != gold {call['result']!r}")
    return [assistant_call_message(call["tool"], args), tool_message(result),
            {"role": "assistant", "content": record["answer"].strip()}]


def build_conversation(record: dict[str, Any], system_prompt: str) -> list[dict[str, Any]]:
    return prompt_messages(record, system_prompt) + target_messages(record)


def sidecar(record: dict[str, Any]) -> dict[str, Any]:
    """Gold/metadata kept out of the model input."""
    return {"id": record["id"], "answer_type": record["answer_type"], "answer": record["answer"],
            "tool_calls": record.get("tool_calls")}


# ---------------------------------------------------------------- tokenization


def render(tokenizer: Any, messages: list[dict[str, Any]], add_generation_prompt: bool = False) -> str:
    # Template switches (e.g. enable_thinking) are attached to the tokenizer by load_tokenizer,
    # so training, formatting and inference cannot render with different settings.
    return tokenizer.apply_chat_template(messages, tools=TOOL_SCHEMAS, tokenize=False,
                                         add_generation_prompt=add_generation_prompt,
                                         **template_kwargs(tokenizer))


def template_kwargs(tokenizer: Any) -> dict[str, Any]:
    return dict(getattr(tokenizer, "clinqa_template_kwargs", None) or {})


@dataclass
class Encoded:
    input_ids: list[int]
    labels: list[int]
    char_spans: list[tuple[int, int]]
    text: str

    @property
    def n_supervised(self) -> int:
        return sum(1 for x in self.labels if x != IGNORE_INDEX)


def _turn_span(tokenizer: Any, messages: list[dict[str, Any]], i: int) -> tuple[str, tuple[int, int]]:
    """(render up to and including assistant turn i, its character span after the generation prompt)."""
    prefix = render(tokenizer, messages[:i], add_generation_prompt=True)
    upto = render(tokenizer, messages[: i + 1])
    if not upto.startswith(prefix):
        raise FormattingError(f"turn {i}: generation prompt is not a prefix of the rendered turn")
    end = upto.rfind(END_OF_TURN)
    if end < len(prefix):
        raise FormattingError(f"turn {i}: assistant turn does not end with {END_OF_TURN}")
    span = (len(prefix), end + len(END_OF_TURN))
    if span[1] - span[0] <= len(END_OF_TURN):
        raise FormattingError(f"turn {i}: empty assistant turn")
    return upto, span


def assistant_char_spans(tokenizer: Any, messages: list[dict[str, Any]], full: str) -> list[tuple[int, int]]:
    spans = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        upto, span = _turn_span(tokenizer, messages, i)
        if not full.startswith(upto):
            raise FormattingError(f"turn {i}: incremental render is not a prefix of the full conversation")
        spans.append(span)
    return spans


def encode(tokenizer: Any, messages: list[dict[str, Any]], max_length: int | None = None) -> Encoded:
    """Tokenize a full conversation and label only assistant spans. Never truncates."""
    text = render(tokenizer, messages)
    spans = assistant_char_spans(tokenizer, messages, text)
    return _label(tokenizer, text, spans, max_length)


def encode_segments(tokenizer: Any, messages: list[dict[str, Any]], max_length: int | None = None,
                    segmented: bool = False) -> list[Encoded]:
    """Encode a conversation as one sequence, or as one sequence per non-prefix-stable turn.

    An assistant turn whose incremental render is a prefix of the full conversation is
    supervised inside the full sequence. A turn that renders differently once later turns
    exist (and so differently from what the model saw when generating it) gets its own
    sequence ending at that turn, rendered exactly as at inference. Every assistant turn is
    supervised exactly once. With segmented=False this is `encode` (one sequence or an error).
    """
    if not segmented:
        return [encode(tokenizer, messages, max_length)]
    text = render(tokenizer, messages)
    in_full, separate = [], []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        upto, span = _turn_span(tokenizer, messages, i)
        (in_full if text.startswith(upto) else separate).append((upto, span))
    out = [_label(tokenizer, upto, [span], max_length) for upto, span in separate]
    if in_full:
        out.append(_label(tokenizer, text, [span for _, span in in_full], max_length))
    return out


def _label(tokenizer: Any, text: str, spans: list[tuple[int, int]], max_length: int | None) -> Encoded:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    labels = [IGNORE_INDEX] * len(ids)
    for s, e in spans:
        starts = {a for a, _ in offsets}
        ends = {b for _, b in offsets}
        if s not in starts or e not in ends:
            raise FormattingError(f"assistant span {s}-{e} does not align with token boundaries")
        for j, (a, b) in enumerate(offsets):
            if a >= s and b <= e and b > a:
                labels[j] = ids[j]
    if max_length is not None and len(ids) > max_length:
        raise FormattingError(f"conversation has {len(ids)} tokens > max_length {max_length}")
    return Encoded(input_ids=ids, labels=labels, char_spans=spans, text=text)


def load_tokenizer(cfg: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"], revision=cfg["tokenizer_revision"])
    tok.clinqa_template_kwargs = dict(cfg.get("chat_template_kwargs") or {})
    return tok


def template_sha256(tokenizer: Any) -> str:
    return hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- CLI


def _percentiles(values: list[int]) -> dict[str, int]:
    v = sorted(values)
    pick = lambda q: v[min(len(v) - 1, int(round(q * (len(v) - 1))))]  # noqa: E731
    return {"n": len(v), "p50": pick(0.5), "p95": pick(0.95), "p99": pick(0.99), "max": v[-1]}


def format_split(records: list[dict[str, Any]], system_prompt: str, tokenizer: Any, max_length: int,
                 full: bool = True, segmented: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, lengths, overlength = [], defaultdict(list), []
    for r in records:
        msgs = build_conversation(r, system_prompt) if full else prompt_messages(r, system_prompt)
        if full:
            n = max(len(e.input_ids) for e in encode_segments(tokenizer, msgs, segmented=segmented))
        else:
            n = len(tokenizer(render(tokenizer, msgs, add_generation_prompt=True), add_special_tokens=False)["input_ids"])
        lengths[r["answer_type"]].append(n)
        lengths["all"].append(n)
        if n > max_length:
            overlength.append(r["id"])
        rows.append({"id": r["id"], "messages": msgs})
    stats = {"measured": "full_conversation" if full else "prompt_only",
             "by_answer_type": {k: _percentiles(v) for k, v in sorted(lengths.items())},
             "answer_type_counts": dict(sorted(Counter(r["answer_type"] for r in records).items())),
             "overlength_ids": overlength}
    return rows, stats


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Format records into native Qwen SFT conversations.")
    parser.add_argument("--config", default="configs/format_core.yaml")
    args = parser.parse_args(argv)
    cfg = load_yaml(args.config)
    system_prompt = resolve(cfg["system_prompt"]).read_text(encoding="utf-8").strip()
    tokenizer = load_tokenizer(cfg)
    out = resolve(cfg["output_dir"])
    report: dict[str, Any] = {
        "tokenizer": cfg["tokenizer"], "tokenizer_revision": cfg["tokenizer_revision"],
        "chat_template_sha256": template_sha256(tokenizer), "tool_schemas_sha256": tool_schemas_sha256(),
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "max_length": cfg["max_length"], "splits": {},
    }
    if template_kwargs(tokenizer) or cfg.get("segmented_turns"):  # absent for the 4B contract, as before
        report.update(chat_template_kwargs=template_kwargs(tokenizer), segmented_turns=bool(cfg.get("segmented_turns")))
    for variant, path in cfg["train_views"].items():
        records = read_jsonl(resolve(path))
        rows, stats = format_split(records, system_prompt, tokenizer, cfg["max_length"],
                                   segmented=bool(cfg.get("segmented_turns")))
        _write_jsonl(out / variant / "train.jsonl", rows)
        _write_jsonl(out / variant / "train.sidecar.jsonl", [sidecar(r) for r in records])
        report["splits"][f"{variant}/train"] = stats
    for split in cfg["eval_splits"]:
        records = load_split(split, cfg["data_config"])
        # Test is measured prompt-only: its targets are never rendered into training artefacts.
        rows, stats = format_split(records, system_prompt, tokenizer, cfg["max_length"], full=split != "test",
                                   segmented=bool(cfg.get("segmented_turns")))
        _write_jsonl(out / "eval" / f"{split}.jsonl", rows)
        _write_jsonl(out / "eval" / f"{split}.sidecar.jsonl", [sidecar(r) for r in records])
        report["splits"][split] = stats
    length_path = resolve(cfg["length_report"])
    length_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, s in report["splits"].items():
        a = s["by_answer_type"]["all"]
        print(f"{name:22s} n={a['n']:4d} p50={a['p50']} p95={a['p95']} max={a['max']} overlength={len(s['overlength_ids'])}")
    # Training inputs must fit; test prompts are informational (inference has no prompt cap).
    if any(s["overlength_ids"] for name, s in report["splits"].items() if name != "test"):
        print("error: overlength conversations; see", length_path)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
