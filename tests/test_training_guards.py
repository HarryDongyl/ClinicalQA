import copy
import json
from pathlib import Path

import pytest

from clinqa.config import load_run_config, load_yaml
from clinqa.data_io import sha256_file
from clinqa.data_views import build_view
from clinqa.training_data import audit_train_view
from clinqa.checkpoints import discover, resume_path


@pytest.fixture(scope="module")
def views(tmp_path_factory):
    root = tmp_path_factory.mktemp("guard_views")
    for view in ("raw", "q5_filtered"):
        build_view(view, root)
    fmt = load_yaml("configs/format_core.yaml")
    fmt["train_views"] = {v: str(root / v / "train.jsonl") for v in ("raw", "q5_filtered")}
    return fmt


def test_filtered_excludes_only_candidates_and_raw_requires_acknowledgment(views):
    filtered = load_run_config("configs/train/q5filtered_lr5e5.yaml")
    records, report = audit_train_view(filtered, views)
    assert len(records) == 1922 and len(report["excluded_ids"]) == 78
    assert report["remaining_q5_ids"] == [] and report["clinical_adjudication"] is False
    raw = load_run_config("configs/train/raw_lr5e5.yaml")
    original, report = audit_train_view(raw, views)
    assert len(original) == 2000 and len(report["remaining_q5_ids"]) == 78
    raw["allow_ungrounded_targets"] = False
    with pytest.raises(ValueError, match="Q5 candidates remain"):
        audit_train_view(raw, views)


def test_tampered_label_is_rejected_even_if_view_hash_is_updated(views, tmp_path):
    import shutil
    original = Path(views["train_views"]["q5_filtered"]).parent
    target = tmp_path / "q5_filtered"
    shutil.copytree(original, target)
    path = target / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["answer"] = "Changed target"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["train_sha256"] = sha256_file(path)
    (target / "manifest.json").write_text(json.dumps(manifest))
    fmt = copy.deepcopy(views)
    fmt["train_views"]["q5_filtered"] = str(path)
    with pytest.raises(ValueError, match="unmodified canonical train selection"):
        audit_train_view(load_run_config("configs/train/q5filtered_lr5e5.yaml"), fmt)


def test_resume_skips_incomplete_checkpoint_and_rejects_adapter_only(tmp_path):
    for step in (50, 100):
        ck = tmp_path / f"checkpoint-{step}"
        ck.mkdir()
        (ck / "adapter_model.safetensors").write_bytes(b"fixture")
        (ck / "trainer_state.json").write_text(json.dumps({"global_step": step, "epoch": step / 100}))
    with pytest.raises(ValueError, match="resumable"):
        resume_path(tmp_path, "latest")
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (tmp_path / "checkpoint-50" / name).write_bytes(b"fixture")
    assert resume_path(tmp_path, "latest").name == "checkpoint-50"
    assert [c["epoch_end"] for c in discover(tmp_path)] == [False, True]


def test_run_names_and_controlled_comparisons():
    raw = load_run_config("configs/train/raw_lr1e4.yaml")
    lower = load_run_config("configs/train/raw_lr5e5.yaml")
    filtered = load_run_config("configs/train/q5filtered_lr5e5.yaml")
    assert raw["training"]["learning_rate"] == 1e-4
    assert lower["training"]["learning_rate"] == filtered["training"]["learning_rate"] == 5e-5
    assert lower["train_view"] == raw["train_view"] == "raw"
    assert filtered["train_view"] == "q5_filtered"
    for cfg in (raw, lower, filtered):
        assert cfg["checkpoint_dir"] == f"checkpoints/{cfg['run_id']}"
        assert cfg["output_dir"] == f"outputs/{cfg['run_id']}"
    assert load_run_config("configs/train_raw.yaml") == raw
    assert load_run_config("configs/train_grounded.yaml") == filtered
