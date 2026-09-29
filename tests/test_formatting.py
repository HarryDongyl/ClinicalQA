import copy
import json

import pytest

from clinqa.config import load_yaml, resolve
from clinqa.data_io import load_split
from clinqa.formatting import (
    IGNORE_INDEX,
    FormattingError,
    build_conversation,
    encode,
    prompt_messages,
    render,
    render_table,
    target_messages,
    user_content,
)

pytest.importorskip("transformers")

CFG = load_yaml("configs/format_core.yaml")
SYSTEM = resolve(CFG["system_prompt"]).read_text(encoding="utf-8").strip()


@pytest.fixture(scope="module")
def tok():
    from clinqa.formatting import load_tokenizer

    try:
        return load_tokenizer(CFG)
    except OSError as e:  # offline without a cached tokenizer
        pytest.skip(f"tokenizer unavailable: {e}")


@pytest.fixture(scope="module")
def train():
    return {r["id"]: r for r in load_split("train")}


def _first(train, pred):
    return next(r for r in train.values() if pred(r))


def test_render_table_shows_empty_unit_and_escapes_pipes():
    table = {"type": "labs", "headers": ["Test", "Value", "Unit", "Reference Range"],
             "rows": [["INR", "1.1", "", "0.8-1.2"], ["A|B", "2", "mg/dL", "1-3"]]}
    text = render_table(table)
    assert "| INR | 1.1 | (no unit) | 0.8-1.2 |" in text
    assert "A\\|B" in text
    assert text.splitlines()[1] == "|---|---|---|---|"


def test_prompt_has_no_label_leakage(train):
    for r in list(train.values())[:200]:
        text = json.dumps(prompt_messages(r, SYSTEM))
        assert r["answer_type"] not in text
        assert r["id"] not in text
        assert r["answer"] not in text


def test_user_content_sections(train):
    r = train["train_000"]
    c = user_content(r)
    assert c.startswith("## Encounter note\n") and "\n\n## Table (vitals)\n| Vital |" in c
    assert c.endswith("## Question\n" + r["question"].strip())


def test_tool_target_uses_executor_and_emits_substance(train):
    r = _first(train, lambda r: r["answer_type"] == "tool_call" and r["tool_calls"][0]["tool"] == "unit_convert")
    call, tool, final = target_messages(r)
    assert call["content"] == "" and "substance" in call["tool_calls"][0]["function"]["arguments"]
    assert json.loads(tool["content"]) == {"result": r["tool_calls"][0]["result"]}
    assert final["content"] == r["answer"].strip()


def test_executor_mismatch_is_an_error(train):
    r = copy.deepcopy(_first(train, lambda r: r["answer_type"] == "tool_call"))
    r["tool_calls"][0]["result"] = -1.0
    with pytest.raises(FormattingError):
        target_messages(r)


def _supervised_text(tok, enc):
    return tok.decode([i for i, l in zip(enc.input_ids, enc.labels) if l != IGNORE_INDEX])


def test_mask_plain_answer(tok, train):
    r = _first(train, lambda r: r["answer_type"] == "extractive")
    enc = encode(tok, build_conversation(r, SYSTEM))
    assert _supervised_text(tok, enc) == r["answer"].strip() + "<|im_end|>"
    # generation header is masked
    first = enc.labels.index(next(l for l in enc.labels if l != IGNORE_INDEX))
    assert tok.decode(enc.input_ids[first - 3:first]).endswith("<|im_start|>assistant\n")


def test_mask_tool_conversation(tok, train):
    r = _first(train, lambda r: r["answer_type"] == "tool_call" and r["tool_calls"][0]["tool"] == "calculate_bmi")
    enc = encode(tok, build_conversation(r, SYSTEM))
    sup = _supervised_text(tok, enc)
    args = r["tool_calls"][0]["arguments"]
    expected_call = ('<tool_call>\n{"name": "calculate_bmi", "arguments": '
                     + json.dumps(args) + '}\n</tool_call><|im_end|>')
    assert sup == expected_call + r["answer"].strip() + "<|im_end|>"
    assert "tool_response" not in sup and "result" not in sup


def test_inference_prompt_is_prefix_of_training_text(tok, train):
    for at in ("extractive", "numeric_reasoning", "tool_call", "uncertain"):
        r = _first(train, lambda r: r["answer_type"] == at)
        full = render(tok, build_conversation(r, SYSTEM))
        prompt = render(tok, prompt_messages(r, SYSTEM), add_generation_prompt=True)
        assert full.startswith(prompt)


def test_overlength_raises(tok, train):
    with pytest.raises(FormattingError):
        encode(tok, build_conversation(train["train_000"], SYSTEM), max_length=16)


def test_every_train_record_formats(tok, train):
    counts = {}
    for r in train.values():
        enc = encode(tok, build_conversation(r, SYSTEM), max_length=CFG["max_length"])
        assert enc.n_supervised > 0
        counts[r["answer_type"]] = counts.get(r["answer_type"], 0) + 1
    assert counts == {"extractive": 800, "numeric_reasoning": 400, "tool_call": 500, "uncertain": 300}
