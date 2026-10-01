"""Wave-three infrastructure: segmented 8B encoding, few-shot prompts, relabel view, P1 probes, analysis helpers."""

import importlib.util
import json
import sys

import pytest

from clinqa.config import PROJECT_ROOT, load_yaml, resolve
from clinqa.data_io import load_split
from clinqa.formatting import (IGNORE_INDEX, FormattingError, build_conversation, encode, encode_segments,
                               load_tokenizer, prompt_messages, render)

pytest.importorskip("transformers")

CFG = load_yaml("configs/format_core.yaml")
SYSTEM = resolve(CFG["system_prompt"]).read_text(encoding="utf-8").strip()
IDS = ["train_000", "train_003", "train_004", "train_006", "train_015", "train_046"]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tok():
    try:
        return load_tokenizer(CFG)
    except OSError as e:
        pytest.skip(f"tokenizer unavailable: {e}")


@pytest.fixture(scope="module")
def tok8b():
    try:
        return load_tokenizer(load_yaml("configs/format_w3_8b.yaml"))
    except OSError as e:
        pytest.skip(f"Qwen3-8B tokenizer unavailable: {e}")


@pytest.fixture(scope="module")
def train():
    return {r["id"]: r for r in load_split("train")}


# ---------------------------------------------------------------- encoding


def test_4b_segments_are_the_single_sequence_encoding(tok, train):
    for i in IDS:
        msgs = build_conversation(train[i], SYSTEM)
        ref = encode(tok, msgs)
        for segmented in (False, True):
            (seg,) = encode_segments(tok, msgs, segmented=segmented)
            assert (seg.input_ids, seg.labels) == (ref.input_ids, ref.labels)


def test_4b_template_has_no_kwargs(tok):
    assert tok.clinqa_template_kwargs == {}


def test_8b_single_sequence_contract_refuses_tool_rows(tok8b, train):
    with pytest.raises(FormattingError):
        encode(tok8b, build_conversation(train["train_006"], SYSTEM))


def test_8b_segments_supervise_each_turn_once_with_the_inference_prefix(tok8b, train):
    for i in IDS:
        msgs = build_conversation(train[i], SYSTEM)
        segs = encode_segments(tok8b, msgs, segmented=True)
        turns = [j for j, m in enumerate(msgs) if m["role"] == "assistant"]
        assert len(segs) == len(turns)  # tool rows: the call turn needs its own sequence
        prefixes = [render(tok8b, msgs[:j], add_generation_prompt=True) for j in turns]
        supervised = []
        for s in segs:
            for a, _ in s.char_spans:
                assert s.text[:a] in prefixes
            supervised.append(tok8b.decode([t for t in s.labels if t != IGNORE_INDEX]))
        assert all("<think>" not in t and t.endswith("<|im_end|>") for t in supervised)
        assert supervised[-1].startswith(msgs[-1]["content"])
        if train[i]["answer_type"] == "tool_call":
            assert supervised[0].startswith("<tool_call>")
    assert prefixes[-1].endswith("<think>\n\n</think>\n\n")


def test_collator_flattens_segments_and_keeps_single_sequence_items():
    from clinqa.train import Collator, n_input_tokens

    one = {"id": "a", "segments": [{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}]}
    two = {"id": "b", "segments": [{"input_ids": [4, 5], "labels": [-100, 5]},
                                   {"input_ids": [6, 7, 8, 9], "labels": [-100, -100, 8, 9]}]}
    batch = Collator(0)([one, two])
    assert batch["input_ids"].tolist() == [[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9]]
    assert batch["labels"].tolist() == [[-100, 2, 3, -100], [-100, 5, -100, -100], [-100, -100, 8, 9]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]]
    assert n_input_tokens([one, two]) == 9


# ---------------------------------------------------------------- few-shot


def test_demos_sit_between_system_and_question(tok, train):
    from clinqa.infer import GenOutput, rollout

    demos = [{"role": "user", "content": "demo question"}, {"role": "assistant", "content": "demo answer"}]
    msgs = prompt_messages(train["train_000"], SYSTEM, demos)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"] and msgs[1:3] == demos

    class Gen:
        def generate(self, prompts, max_new_tokens):
            return [GenOutput(text="ok", n_tokens=2, finished=True) for _ in prompts]

    record = {k: train["train_000"][k] for k in ("id", "note", "table", "question")}
    t = rollout([record], Gen(), tok, SYSTEM, demos=demos)[0]
    assert "demo answer" in t["prompt"] and t["prompt"].index("demo answer") < t["prompt"].index("## Question")
    assert t["prompt_tokens"] == len(tok(t["prompt"], add_special_tokens=False)["input_ids"])


def test_unapproved_demos_are_refused(tmp_path):
    from clinqa.evaluate import load_demos

    path = tmp_path / "demos.json"
    path.write_text(json.dumps({"status": "draft", "messages": [], "messages_sha256": ""}))
    with pytest.raises(ValueError, match="not approved"):
        load_demos({"fewshot": str(path)})
    assert load_demos({}) is None


def test_fewshot_draft_matches_its_rule():
    spec = json.loads((PROJECT_ROOT / "configs/w3/fewshot_v3.json").read_text())
    fit = json.loads((PROJECT_ROOT / "configs/w3/trainfit_ids.json").read_text())
    assert len(spec["ids"]) == 4 and not set(spec["ids"]) & set(fit["ids"])
    assert [m["role"] for m in spec["messages"]].count("user") == 4 and fit["n"] == 200


# ---------------------------------------------------------------- relabel view


def test_relabel_view_is_audited_against_the_review(tmp_path):
    from clinqa.data_views import build_view
    from clinqa.training_data import audit_train_view, grounding_flags

    ids = {f.id for f in grounding_flags(load_split("train"))}
    flagged = [r["id"] for r in load_split("train") if r["id"] in ids]  # canonical order
    review = tmp_path / "review.jsonl"
    rows = [{"id": i, "accepted": n % 2 == 0, "reviewer": "test", "review_type": "ai_grounding_review",
             "answer": "Weight and height are not documented."}
            for n, i in enumerate(flagged)]
    review.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = build_view("q5_relabeled", tmp_path / "views", review_path=str(review))
    accepted = [r["id"] for r in rows if r["accepted"]]
    assert summary["after"]["count"] == 2000 - len(flagged) + len(accepted)
    assert summary["relabeled_ids"] == accepted
    assert summary["review_complete"] is True
    assert summary["manual_review_complete"] is False
    assert summary["clinical_adjudication"] is False
    assert summary["review_types"] == ["ai_grounding_review"]
    fmt = {**CFG, "train_views": {"q5_relabeled": str(tmp_path / "views/q5_relabeled/train.jsonl")}}
    cfg = {"train_view": "q5_relabeled"}
    records, report = audit_train_view(cfg, fmt)
    by_id = {r["id"]: r for r in records}
    original = {r["id"]: r for r in load_split("train")}
    for i in accepted:
        assert by_id[i]["answer_type"] == "uncertain" and "tool_calls" not in by_id[i]
        assert all(by_id[i][k] == original[i][k] for k in ("note", "table", "question"))
    assert report["remaining_q5_ids"] == [] and report["relabeled_ids"] == accepted
    assert report["clinical_adjudication"] is False
    assert report["review_types"] == ["ai_grounding_review"]
    with pytest.raises(ValueError, match="unexpected training count"):
        audit_train_view({**cfg, "expected_train_examples": 2000}, fmt)
    # The review cannot change after the view was built.
    review.write_text(review.read_text().replace('"accepted": false', '"accepted": true', 1))
    with pytest.raises(ValueError, match="relabel review changed"):
        audit_train_view(cfg, fmt)


def test_relabel_review_must_be_complete(tmp_path):
    from clinqa.data_views import load_relabel_review

    review = tmp_path / "review.jsonl"
    review.write_text(json.dumps({"id": "train_009", "accepted": None, "reviewer": None, "answer": "x"}) + "\n")
    with pytest.raises(ValueError, match="not reviewed"):
        load_relabel_review({"train_009"}, str(review))
    with pytest.raises(ValueError, match="exactly"):
        load_relabel_review({"train_009", "train_023"}, str(review))


# ---------------------------------------------------------------- P1 and analysis


def test_p1_edit_removes_every_measurement():
    prep = _script("w3_prep")
    val = {r["id"]: r for r in load_split("val")}
    probe, info = prep.edit_record(val["val_002"])
    assert probe and "93.7" not in probe["note"] and "172.6" not in probe["note"]
    assert prep.probe_problems(probe, val["val_002"]) == []
    excluded = {**val["val_002"], "question": "Using a weight of 80 kg and height of 170 cm, what is the BMI?"}
    assert prep.edit_record(excluded)[0] is None


def test_p1_draft_records_pass_the_automatic_checks():
    prep = _script("w3_prep")
    spec = json.loads((PROJECT_ROOT / "configs/w3/p1_probes.json").read_text())
    val = {r["id"]: r for r in load_split("val")}
    assert spec["n_sources"] == 40 and spec["n_eligible"] + len(spec["excluded"]) == 40
    for p in spec["records"]:
        assert prep.probe_problems(p, val[p["source_id"]]) == []
        assert set(p) == {"id", "source_id", "note", "table", "question"}


def test_probe_classification_and_auroc():
    an = _script("w3_analyze")
    turn = {"status": "ok", "calls": [{"name": "calculate_bmi", "arguments": {"weight_kg": 80, "height_cm": 170}}]}
    t = {"id": "p", "stop_reason": "answer", "turns": [turn], "final_answer": "BMI is 27.7 kg/m²."}
    c = an.classify_probe(t)
    assert c["call_fabrication"] and c["fabrication"] and c["text_claims"]
    ok = {"id": "q", "stop_reason": "answer", "turns": [{"status": "no_call", "calls": []}],
          "final_answer": "Weight and height are not documented, so BMI cannot be calculated."}
    c = an.classify_probe(ok)
    assert not c["fabrication"] and c["states_missing"]
    assert an.auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert an.auroc([0.5, 0.5], [False, True]) == 0.5
    assert an.auroc([0.1], [True]) is None


def test_wave3_configs_resolve():
    from clinqa.config import load_run_config

    a8b = load_run_config("configs/train/w3_8b_filtered_lr1e4.yaml")
    fmt = load_yaml(a8b["format_config"])
    assert a8b["model"]["tokenizer"] == fmt["tokenizer"] and a8b["model"]["quantization"] == "nf4"
    assert a8b["model"]["chat_template_kwargs"] == fmt["chat_template_kwargs"] == {"enable_thinking": False}
    assert a8b["training"]["learning_rate"] == 1.0e-4 and a8b["train_view"] == "q5_filtered"
    for s in (42, 43, 44):
        f = load_run_config(f"configs/train/w3_relabel_lr1e4_s{s}.yaml")
        assert (f["seed"], f["train_view"], f["training"]["learning_rate"]) == (s, "q5_relabeled", 1.0e-4)
        assert "expected_train_examples" not in f
    for name in ("v1", "v3", "v3_fs4", "8b"):
        assert load_yaml(f"configs/eval_w3_{name}.yaml")["batch_size"] == 2


def _probe_record():
    return {"note": "She reports an unintentional weight loss of 8 lbs.", "question": "What is her BMI?",
            "table": {"type": "labs", "headers": ["Test", "Value", "Unit"], "rows": [["LDL", "112.1", "mg/dL"]]}}


def test_p1_classifier_counts_wrong_tool_fabrication():
    an = _script("w3_analyze")
    t = {"id": "c", "stop_reason": "answer", "final_answer": "Weight and height are not documented.",
         "turns": [{"status": "ok", "calls": [{"name": "unit_convert", "arguments": {
             "value": 80, "from_unit": "kg", "to_unit": "lb", "substance": None}}]}]}
    c = an.classify_probe(t, _probe_record())
    assert c["call_fabrication"] and c["fabrication"] and not c["intended"]
    # Converting the documented weight change is grounded: an unintended call, not a fabricated measurement.
    t["turns"][0]["calls"][0]["arguments"].update(value=8, from_unit="lb", to_unit="kg")
    c = an.classify_probe(t, _probe_record())
    assert c["called"] and not c["call_fabrication"] and not c["intended"]


def test_p1_classifier_checks_every_visible_turn():
    an = _script("w3_analyze")
    t = {"id": "e", "stop_reason": "answer", "final_answer": "Weight and height are not documented.",
         "turns": [{"status": "ok", "content": "The patient weighs 80 kg.", "raw": "The patient weighs 80 kg.",
                    "calls": []},
                   {"status": "no_call", "content": "Weight and height are not documented.", "calls": []}]}
    c = an.classify_probe(t, _probe_record())
    assert c["text_claims"] == ["80 kg"] and c["fabrication"]


def test_p1_classifier_routes_examples_and_grounded_values_to_review():
    an = _script("w3_analyze")
    for text in ("For example, a patient weighing 80 kg at 170 cm would have a BMI of 27.7 kg/m².",
                 "Height is not documented; obesity is defined as BMI ≥30 kg/m²."):
        t = {"id": "h", "stop_reason": "answer", "final_answer": text,
             "turns": [{"status": "no_call", "content": text, "calls": []}]}
        c = an.classify_probe(t, _probe_record())
        assert not c["fabrication"] and c["review_mentions"]
    grounded = "She lost 8 lbs, but current weight and height are not documented."
    t = {"id": "g", "stop_reason": "answer", "final_answer": grounded,
         "turns": [{"status": "no_call", "content": grounded, "calls": []}]}
    c = an.classify_probe(t, _probe_record())
    assert not c["fabrication"] and not c["review_mentions"] and c["intended"]
    asserted = "Her BMI of 31.2 indicates obesity."
    t = {"id": "a", "stop_reason": "answer", "final_answer": asserted,
         "turns": [{"status": "no_call", "content": asserted, "calls": []}]}
    assert an.classify_probe(t, _probe_record())["fabrication"]


def test_gates_pin_the_p1_classifier():
    an = _script("w3_analyze")
    assert load_yaml("configs/w3/gates.yaml")["p1_classifier"] == an.P1_CLASSIFIER
