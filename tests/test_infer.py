import json

import pytest

from clinqa.config import load_yaml, resolve
from clinqa.data_io import load_split
from clinqa.formatting import build_conversation, load_tokenizer, render
from clinqa.infer import Budget, GenOutput, rollout

pytest.importorskip("transformers")

CFG = load_yaml("configs/format_core.yaml")
SYSTEM = resolve(CFG["system_prompt"]).read_text(encoding="utf-8").strip()
BMI_CALL = '<tool_call>\n{"name": "calculate_bmi", "arguments": {"weight_kg": 106.4, "height_cm": 189.2}}\n</tool_call>'


@pytest.fixture(scope="module")
def tok():
    try:
        return load_tokenizer(CFG)
    except OSError as e:
        pytest.skip(f"tokenizer unavailable: {e}")


@pytest.fixture(scope="module")
def train():
    return {r["id"]: r for r in load_split("train")}


class Scripted:
    """Returns scripted outputs per turn; records every prompt it was given."""

    def __init__(self, script):
        self.script = script  # list of GenOutput or str, one per turn
        self.prompts = []

    def generate(self, prompts, max_new_tokens):
        outs = []
        for p in prompts:
            self.prompts.append(p)
            item = self.script[min(p.count("<|im_start|>assistant") - 1, len(self.script) - 1)]
            outs.append(item if isinstance(item, GenOutput) else GenOutput(text=item, n_tokens=10, finished=True))
        return outs


def _strip_gold(r):
    return {k: v for k, v in r.items() if k not in ("answer", "answer_type", "tool_calls")}


def test_plain_answer(tok, train):
    t = rollout([_strip_gold(train["train_000"])], Scripted(["HR is 117 bpm."]), tok, SYSTEM)[0]
    assert (t["stop_reason"], t["final_answer"], t["n_calls"], len(t["turns"])) == ("answer", "HR is 117 bpm.", 0, 1)


def test_tool_call_executes_real_tool_and_matches_training_render(tok, train):
    r = train["train_006"]
    gen = Scripted([BMI_CALL, r["answer"]])
    t = rollout([_strip_gold(r)], gen, tok, SYSTEM)[0]
    assert t["stop_reason"] == "answer" and t["n_calls"] == 1
    assert t["turns"][0]["results"] == [29.7]
    assert '<tool_response>\n{"result": 29.7}\n</tool_response>' in gen.prompts[1]
    # second-turn prompt is exactly the training conversation up to the final answer
    train_text = render(tok, build_conversation(r, SYSTEM))
    assert train_text.startswith(gen.prompts[1])


def test_invalid_json_is_not_repaired(tok, train):
    bad = '<tool_call>\n{"name": "calculate_bmi", "arguments": {"weight_kg": 106.4,}}\n</tool_call>'
    t = rollout([train["train_006"]], Scripted([bad]), tok, SYSTEM)[0]
    assert t["stop_reason"] == "parse_error" and t["n_calls"] == 0 and t["final_answer"] is None


def test_schema_error_not_executed_or_retried(tok, train):
    bad = '<tool_call>\n{"name": "calculate_bmi", "arguments": {"weight_kg": "106.4", "height_cm": 189.2}}\n</tool_call>'
    gen = Scripted([bad, "Could not compute."])
    t = rollout([train["train_006"]], gen, tok, SYSTEM)[0]
    assert t["turns"][0]["status"] == "schema_error" and t["turns"][0]["results"] == []
    assert t["stop_reason"] == "schema_error" and len(gen.prompts) == 1 and t["final_answer"] is None


def test_executor_error_is_returned_to_model(tok, train):
    call = ('<tool_call>\n{"name": "unit_convert", "arguments": {"value": 1.0, "from_unit": "mg/dL", '
            '"to_unit": "furlong", "substance": null}}\n</tool_call>')
    gen = Scripted([call, "Unsupported conversion."])
    t = rollout([train["train_015"]], gen, tok, SYSTEM)[0]
    assert t["turns"][0]["results"][0].startswith("Error: unsupported conversion")
    assert json.loads(gen.prompts[1].split("<tool_response>\n")[1].split("\n</tool_response>")[0])["error"]
    assert t["stop_reason"] == "answer"


def test_two_calls_in_one_turn_exceed_budget(tok, train):
    t = rollout([train["train_006"]], Scripted([BMI_CALL + "\n" + BMI_CALL]), tok, SYSTEM)[0]
    assert t["stop_reason"] == "budget" and t["n_calls"] == 0


def test_repeated_call_exceeds_budget(tok, train):
    t = rollout([train["train_006"]], Scripted([BMI_CALL, BMI_CALL]), tok, SYSTEM)[0]
    assert t["stop_reason"] == "budget" and t["n_calls"] == 1 and t["final_answer"] is None


def test_max_tokens(tok, train):
    t = rollout([train["train_000"]], Scripted([GenOutput("HR is", 256, False)]), tok, SYSTEM)[0]
    assert t["stop_reason"] == "max_tokens" and t["final_answer"] is None


def test_turn_budget_and_logging_fields(tok, train):
    t = rollout([train["train_006"]], Scripted([BMI_CALL, "BMI 29.7"]), tok, SYSTEM,
                budget=Budget(max_assistant_turns=1))[0]
    assert t["stop_reason"] == "budget"
    assert {"raw", "status", "calls", "results", "n_tokens", "latency_s", "call_logprob", "token_logprobs"} <= set(
        t["turns"][0])
    assert t["prompt"].endswith("<|im_start|>assistant\n") and len(t["prompt_sha256"]) == 64


def test_batch_preserves_order(tok, train):
    ids = ["train_000", "train_001", "train_002"]
    out = rollout([train[i] for i in ids], Scripted(["x"]), tok, SYSTEM, batch_size=2)
    assert [t["id"] for t in out] == ids
