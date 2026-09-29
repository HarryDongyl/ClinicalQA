import json

import pytest

from clinqa.config import load_yaml
from clinqa.data_io import load_split
from clinqa.evaluate import compare, q5_ids, score, select, wilson
from clinqa.metrics import gold_trajectory


def test_select_rejects_test_split_and_empty_candidates(cfg):
    with pytest.raises(ValueError, match="validation only"):
        select(cfg, ["anything"], split="test")
    with pytest.raises(ValueError, match="unique"):
        select(cfg, [])


def test_freeze_requires_full_matching_validation_and_locks_adapter(cfg, tmp_path, monkeypatch):
    from clinqa import evaluate as ev
    from clinqa.run_info import adapter_sha256, sha256_file
    from clinqa.config import resolve

    monkeypatch.setattr(ev, "PROJECT_ROOT", tmp_path)
    val = load_split("val")
    checkpoint = tmp_path / "checkpoints/raw_lr5e5/checkpoint-125"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter fixture")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"epoch": 1.0, "global_step": 125}))
    label = "raw_lr5e5_step000125"
    for name, adapter in [("base", None), (label, str(checkpoint))]:
        _write(cfg, name, "val", [gold_trajectory(r) for r in val])
        score(cfg, name, "val")
        directory = ev.run_dir(cfg, name, "val")
        (directory / "run.json").write_text(json.dumps({
            "n": 250, "limit": None, "protocol": ev.protocol(cfg), "adapter_sha256": adapter_sha256(adapter),
            "split_sha256": sha256_file(resolve("data/val.jsonl"))}))
    selection_dir = tmp_path / "raw_lr5e5"
    selection_dir.mkdir()
    (selection_dir / "selection.json").write_text(json.dumps({"selected": label, "checkpoints": {label: str(checkpoint)}}))
    final = ev.freeze(cfg, ["raw_lr5e5"], str(tmp_path / "final.yaml"))
    assert final["runs"][1]["adapter"] == "checkpoints/raw_lr5e5/checkpoint-125"
    assert final["runs"][1]["adapter_sha256"] == adapter_sha256(checkpoint)
    with pytest.raises(ValueError, match="already frozen"):
        ev.freeze(cfg, ["raw_lr5e5"], str(tmp_path / "final.yaml"))
    info_path = ev.run_dir(cfg, label, "val") / "run.json"
    info = json.loads(info_path.read_text())
    info["n"] = 20
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="partial validation"):
        ev.freeze(cfg, ["raw_lr5e5"], str(tmp_path / "other.yaml"))


@pytest.fixture()
def cfg(tmp_path):
    c = load_yaml("configs/eval_core.yaml")
    c["output_dir"] = str(tmp_path)
    return c


def _write(cfg, label, split, trajectories):
    from pathlib import Path

    d = Path(cfg["output_dir"]) / label / split
    d.mkdir(parents=True)
    (d / "trajectories.jsonl").write_text("".join(json.dumps(t) + "\n" for t in trajectories))


def _refuse(r):
    return {"id": r["id"], "turns": [{"status": "no_call", "calls": [], "results": []}],
            "final_answer": "This is not documented in the note.", "stop_reason": "answer"}


def test_wilson():
    assert wilson(0, 0) is None
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and round(lo, 2) == 0.40


def test_val_q5_ids_are_the_known_seven():
    assert len(q5_ids("val", "configs/analysis.yaml")) == 7


def test_score_gold_and_refusal_runs(cfg):
    val = load_split("val")
    _write(cfg, "gold", "val", [gold_trajectory(r) for r in val])
    _write(cfg, "refuse", "val", [_refuse(r) for r in val])
    g = score(cfg, "gold", "val")["subsets"]
    assert g["full"]["n"] == 250 and g["q5_grounded"]["n"] == 243 and g["q5_only"]["n"] == 7
    assert g["full"]["tool_e2e"]["rate"] == 1.0 and g["full"]["tool_selection_accuracy"]["k"] == 62
    assert g["full"]["over_call_rate"]["k"] == 0
    rf = score(cfg, "refuse", "val")["subsets"]["full"]
    assert rf["extractive_accuracy"]["k"] == 0 and rf["tool_selection_accuracy"]["k"] == 0
    assert rf["over_refusal_rate"]["rate"] == 1.0
    cmp = compare(cfg, "refuse", "gold", "val")
    assert cmp["by_type"]["extractive"]["diff"] == 1.0 and len(cmp["by_type"]["extractive"]["fixed"]) == 100
    sel = select(cfg, ["refuse", "gold"])
    assert sel["selected"] == "gold"
