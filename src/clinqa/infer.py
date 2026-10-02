"""Rollout runner: generate -> parse -> validate -> execute -> append tool response -> answer.

Both tools are offered on every example; gold is never read here (no answer_type, no
gold arguments/results). Budgets (D-023): at most `max_calls` executed calls,
`max_assistant_turns` assistant turns, `max_new_tokens` per turn and
`max_total_new_tokens` per example. Malformed output is recorded, never repaired
(D-026). A schema-invalid call is neither executed nor retried (stop_reason
schema_error); executor errors on valid calls are returned to the model as {"error": ...}.

Each trajectory (one JSON line per id, D-029) holds the rendered prompt, every raw
turn, parse status, calls, executor outputs, stop reason, token counts, per-token
log-probs and the first-token decision log-prob of opening a <tool_call>.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from clinqa.formatting import prompt_messages, render, tool_message
from clinqa.schemas import parse_assistant_output
from clinqa.tools import execute_tool


@dataclass
class GenOutput:
    text: str  # decoded continuation, end-of-turn tokens stripped
    n_tokens: int
    finished: bool  # stopped on an end-of-turn token (False: hit max_new_tokens)
    token_logprobs: list[float] = field(default_factory=list)
    call_logprob: float | None = None  # log p(first token == <tool_call>)


class Generator(Protocol):
    def generate(self, prompts: list[str], max_new_tokens: int) -> list[GenOutput]: ...


@dataclass
class Budget:
    max_calls: int = 1
    max_assistant_turns: int = 2
    max_new_tokens: int = 256
    max_total_new_tokens: int = 512


class _State:
    def __init__(self, record: dict[str, Any], system_prompt: str, tokenizer: Any,
                 demos: list[dict[str, Any]] | None = None) -> None:
        self.id = record["id"]
        self.messages = prompt_messages(record, system_prompt, demos)
        self.prompt = render(tokenizer, self.messages, add_generation_prompt=True)
        self.turns: list[dict[str, Any]] = []
        self.calls_made = 0
        self.tokens = 0
        self.final_answer: str | None = None
        self.stop_reason: str | None = None

    @property
    def done(self) -> bool:
        return self.stop_reason is not None


def _step(state: _State, out: GenOutput, budget: Budget, latency_s: float, call_format: str = "json") -> None:
    parsed = parse_assistant_output(out.text, call_format)
    turn: dict[str, Any] = {
        "raw": out.text, "status": parsed.status, "content": parsed.content, "errors": parsed.errors,
        "calls": [{"name": c.name, "arguments": c.arguments} for c in parsed.calls], "results": [],
        "n_tokens": out.n_tokens, "finished": out.finished, "latency_s": round(latency_s, 4),
        "call_logprob": out.call_logprob, "token_logprobs": [round(x, 4) for x in out.token_logprobs],
    }
    state.turns.append(turn)
    state.tokens += out.n_tokens
    if not out.finished:
        state.stop_reason = "max_tokens"
    elif parsed.status in ("invalid_json", "unterminated"):
        state.stop_reason = "parse_error"
    elif parsed.status == "schema_error":
        state.stop_reason = "schema_error"  # not executed, no retry turn (D-026)
    elif parsed.status == "no_call":
        state.final_answer = parsed.content
        state.stop_reason = "answer"
    elif state.calls_made + len(parsed.calls) > budget.max_calls:
        state.stop_reason = "budget"
    else:
        # Re-render the call through the chat template, exactly as in training.
        state.messages.append({"role": "assistant", "content": parsed.content, "tool_calls": [
            {"type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in parsed.calls]})
        for c in parsed.calls:
            result = execute_tool(c.name, c.arguments)
            turn["results"].append(result)
            state.messages.append(tool_message(result))
        state.calls_made += len(parsed.calls)
    if not state.done and (len(state.turns) >= budget.max_assistant_turns or state.tokens >= budget.max_total_new_tokens):
        state.stop_reason = "budget"


def rollout(records: list[dict[str, Any]], generator: Generator, tokenizer: Any, system_prompt: str,
            budget: Budget | None = None, batch_size: int = 16,
            demos: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """`demos`: fixed few-shot messages placed before every question (inference-only, D-078)."""
    budget = budget or Budget()
    states = [_State(r, system_prompt, tokenizer, demos) for r in records]
    call_format = getattr(tokenizer, "clinqa_call_format", "json")
    while True:
        active = [s for s in states if not s.done]
        if not active:
            break
        prompts = {id(s): render(tokenizer, s.messages, add_generation_prompt=True) for s in active}
        # Sort by prompt length so batches pad little; generation is greedy, so order does not change outputs
        # beyond numerical batch effects (recorded in the run manifest).
        active.sort(key=lambda s: len(prompts[id(s)]))
        for i in range(0, len(active), batch_size):
            batch = active[i:i + batch_size]
            limit = min(budget.max_new_tokens, min(budget.max_total_new_tokens - s.tokens for s in batch))
            t0 = time.perf_counter()
            outs = generator.generate([prompts[id(s)] for s in batch], max_new_tokens=limit)
            per_example = (time.perf_counter() - t0) / len(batch)
            for s, out in zip(batch, outs):
                _step(s, out, budget, per_example, call_format)
    return [{"id": s.id, "prompt": s.prompt, "prompt_sha256": hashlib.sha256(s.prompt.encode()).hexdigest(),
             "prompt_tokens": len(tokenizer(s.prompt, add_special_tokens=False)["input_ids"]),
             "turns": s.turns, "final_answer": s.final_answer, "stop_reason": s.stop_reason,
             "n_calls": s.calls_made, "n_new_tokens": s.tokens} for s in states]


class _GreedyLogprobRecorder:
    """Logits processor recording, per step, log p(greedy token) and at step 0 log p(<tool_call>).

    Avoids output_scores (B x T x V floats). Valid only for greedy decoding, where the chosen
    token is the argmax of the (unprocessed) logits.
    """

    def __init__(self, call_id: int) -> None:
        self.call_id = call_id
        self.steps: list[Any] = []
        self.call_lp: Any = None

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        s = scores.float()
        lse = torch.logsumexp(s, dim=-1)
        self.steps.append((s.max(dim=-1).values - lse).cpu())
        if self.call_lp is None:
            self.call_lp = (s[:, self.call_id] - lse).cpu()
        return scores


class HFGenerator:
    """Greedy batched generation with transformers (D-023)."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        import torch

        self.torch = torch
        self.model = model
        self.tok = tokenizer
        self.tok.padding_side = "left"
        self.stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
        self.call_id = tokenizer.convert_tokens_to_ids("<tool_call>")

    def generate(self, prompts: list[str], max_new_tokens: int) -> list[GenOutput]:
        from transformers import LogitsProcessorList

        torch = self.torch
        enc = self.tok(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to(self.model.device)
        rec = _GreedyLogprobRecorder(self.call_id)
        with torch.no_grad():
            seqs = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                       top_p=None, top_k=None, eos_token_id=self.stop_ids,
                                       pad_token_id=self.tok.pad_token_id,
                                       logits_processor=LogitsProcessorList([rec]))
        new = seqs[:, enc["input_ids"].shape[1]:]
        step_lp = torch.stack(rec.steps, dim=1).tolist() if rec.steps else [[] for _ in prompts]
        results = []
        for b in range(new.shape[0]):
            ids = new[b].tolist()
            n = next((j + 1 for j, t in enumerate(ids) if t in self.stop_ids), len(ids))
            finished = bool(ids) and ids[n - 1] in self.stop_ids
            body = ids[: n - 1] if finished else ids[:n]
            text = self.tok.decode(body, skip_special_tokens=False)
            results.append(GenOutput(text=text.strip(), n_tokens=n, finished=finished,
                                     token_logprobs=step_lp[b][:n], call_logprob=rec.call_lp[b].item()))
        return results
