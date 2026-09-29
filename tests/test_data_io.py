import json
from pathlib import Path

import pytest

from clinqa.config import load_yaml, resolve
from clinqa.data_io import DataPreparationError, load_split, prepare_data, read_jsonl, sha256_file, verify_manifest
from clinqa.validation import validate_records

EXPECTED = {"train": 2000, "val": 250, "test": 400}
REQUIRED_FIELDS = {"id", "note", "table", "question", "answer", "answer_type"}


@pytest.fixture()
def tmp_config(tmp_path):
    cfg = load_yaml("configs/data.yaml")
    cfg["output_dir"] = str(tmp_path / "data")
    cfg["manifest"] = str(tmp_path / "data" / "MANIFEST.json")
    return cfg


def test_prepare_copies_byte_identical_and_is_idempotent(tmp_config, tmp_path):
    first = prepare_data(tmp_config)
    assert set(first["actions"].values()) == {"copied"}
    for name, info in first["manifest"]["files"].items():
        copied = tmp_path / "data" / Path(info["target"]).name
        assert sha256_file(copied) == sha256_file(resolve(info["source"])) == info["sha256"]
        if name in EXPECTED:
            assert info["n_records"] == EXPECTED[name]
    manifest_bytes = (tmp_path / "data" / "MANIFEST.json").read_bytes()

    second = prepare_data(tmp_config)
    assert set(second["actions"].values()) == {"unchanged"}
    assert (tmp_path / "data" / "MANIFEST.json").read_bytes() == manifest_bytes


def test_prepare_refuses_to_overwrite_modified_target(tmp_config, tmp_path):
    prepare_data(tmp_config)
    target = tmp_path / "data" / "val.jsonl"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(DataPreparationError):
        prepare_data(tmp_config)
    assert prepare_data(tmp_config, force=True)["actions"]["val"] == "copied"


def test_prepare_rejects_wrong_record_count(tmp_config):
    tmp_config["splits"]["val"]["expected_count"] = 251
    with pytest.raises(DataPreparationError, match="expected 251"):
        prepare_data(tmp_config)
    assert not Path(tmp_config["output_dir"]).exists()


def test_read_jsonl_reports_line_of_invalid_json(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"a": 1}\n{not json}\n', encoding="utf-8")
    with pytest.raises(DataPreparationError, match=":2:"):
        read_jsonl(p)


@pytest.mark.parametrize("split", sorted(EXPECTED))
def test_splits_load_with_expected_schema(split):
    rows = load_split(split)
    assert len(rows) == EXPECTED[split]
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert REQUIRED_FIELDS <= set(r)
        assert ("tool_calls" in r) == (r["answer_type"] == "tool_call")


def test_manifest_matches_data_files():
    manifest = json.loads(resolve("data/MANIFEST.json").read_text(encoding="utf-8"))
    for info in manifest["files"].values():
        assert sha256_file(resolve(info["target"])) == info["sha256"]


@pytest.mark.parametrize("line", [
    '{"id": "a", "id": "b"}', '{"x": NaN}', '{"x": Infinity}',
    '{"x": -Infinity}', '{"x": 1e999}', '{"nested": {"x": 1, "x": 2}}',
])
def test_json_reader_rejects_ambiguous_or_nonfinite_input(tmp_path, line):
    path = tmp_path / "invalid.jsonl"
    path.write_text(line + "\n")
    with pytest.raises(DataPreparationError, match=":1:"):
        read_jsonl(path)


@pytest.mark.parametrize("change", [
    lambda r: r.update(table=[]),
    lambda r: r["table"].update(rows=[["Weight", "70"]]),
    lambda r: r["tool_calls"][0]["arguments"].update(weight_kg=True),
    lambda r: r["tool_calls"][0]["arguments"].update(weight_kg="70"),
    lambda r: r["tool_calls"][0]["arguments"].update(height_cm=float("inf")),
    lambda r: r.update(answer_type="uncertain"),
    lambda r: r["tool_calls"][0].update(tool="unknown"),
])
def test_structural_validation_precedes_feature_extraction(change):
    r = {"id": "train_0", "note": "Weight 70 kg, height 175 cm.", "question": "BMI?",
         "answer": "22.9", "answer_type": "tool_call",
         "table": {"type": "vitals", "headers": ["Vital", "Value", "Unit"],
                   "rows": [["INR", "1", ""]]},
         "tool_calls": [{"tool": "calculate_bmi", "arguments": {"weight_kg": 70, "height_cm": 175}, "result": 22.9}]}
    validate_records([r], "train")
    change(r)
    with pytest.raises(ValueError, match="train_0"):
        validate_records([r], "train")


def test_manifest_checks_current_bytes(tmp_config, tmp_path):
    import yaml

    prepare_data(tmp_config)
    config_path = tmp_path / "data.yaml"
    config_path.write_text(yaml.safe_dump(tmp_config))
    verify_manifest(str(config_path))
    target = Path(tmp_config["output_dir"]) / "train.jsonl"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(DataPreparationError, match="checksum mismatch"):
        verify_manifest(str(config_path))
