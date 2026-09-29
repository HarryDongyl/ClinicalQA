import json

import pytest

from clinqa.data_io import sha256_file, split_path
from clinqa.data_views import build_view


def test_views_preserve_raw_and_evaluation_and_exclude_only_train_q5(tmp_path):
    before = {s: sha256_file(split_path(s)) for s in ("train", "val", "test")}
    raw = build_view("raw", tmp_path)
    filtered = build_view("q5_filtered", tmp_path)
    assert raw["after"]["count"] == 2000
    assert raw["excluded_ids"] == []
    assert raw["train_sha256"] == before["train"]
    assert filtered["after"]["count"] == 1922
    assert filtered["after"]["answer_types"] == {
        "extractive": 800, "numeric_reasoning": 400, "tool_call": 422, "uncertain": 300}
    assert len(filtered["excluded_ids"]) == 78
    assert "train_1890" not in filtered["excluded_ids"]
    assert all(i.startswith("train_") for i in filtered["excluded_ids"])
    assert {s: sha256_file(split_path(s)) for s in before} == before
    assert not (tmp_path / "q5_filtered/val.jsonl").exists()
    assert not (tmp_path / "q5_filtered/test.jsonl").exists()
    first = (tmp_path / "q5_filtered/manifest.json").read_bytes()
    build_view("q5_filtered", tmp_path)
    assert first == (tmp_path / "q5_filtered/manifest.json").read_bytes()
    audit = [json.loads(l) for l in (tmp_path / "q5_filtered/q5_review.jsonl").read_text().splitlines()]
    assert len(audit) == 78
    assert all(r["excluded"] and r["review_status"].startswith("heuristic") for r in audit)


def test_unknown_view_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        build_view("filter_test", tmp_path)
