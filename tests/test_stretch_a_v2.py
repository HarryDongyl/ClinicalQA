"""Stretch A revision (D-100): frozen v2 training rows, their construction and the q5_relabeled_egfr2 view."""

import importlib.util
import json
import sys
from collections import Counter

import pytest

from clinqa.config import PROJECT_ROOT
from clinqa.data_io import sha256_file
from clinqa.data_views import EGFR_VIEWS, egfr_view_additions
from clinqa.parsing import extract_age_sex

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("stretch_a_data_v2", PROJECT_ROOT / "scripts" / "stretch_a_data_v2.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)

DATA = PROJECT_ROOT / "data" / "stretch_a_v2"
ROWS = [json.loads(x) for x in (DATA / "train_additions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
BY = {k: [r for r in ROWS if r["stretch_a"]["kind"] == k] for k in ("positive", "negative_age", "negative_sex")}


def test_frozen_file_matches_manifest_and_a_fresh_build():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert sha256_file(DATA / "train_additions.jsonl") == manifest["sha256"]["train_additions.jsonl"]
    fresh = v2.build()
    assert [r for k in ("positive", "negative_age", "negative_sex") for r in fresh[k]] == ROWS  # deterministic, seed 42


def test_counts_stage_balance_and_pairing():
    assert {k: len(v) for k, v in BY.items()} == {"positive": 120, "negative_age": 60, "negative_sex": 20}
    stages = Counter(r["stretch_a"]["stage"] for r in BY["positive"])
    assert stages == {"G1": 29, "G2": 29, "G3a": 13, "G3b": 15, "G4": 23, "G5": 11}
    pos_src = {r["source_id"] for r in BY["positive"]}
    paired = [r for r in BY["negative_age"] if r["stretch_a"]["paired"]]
    assert len(paired) == 40 and all(r["source_id"] in pos_src for r in paired)
    assert len({r["stretch_a"]["source_stage"] for r in paired}) == 6  # every category has paired negatives
    unpaired = [r for k in ("negative_age", "negative_sex") for r in BY[k] if not r["stretch_a"].get("paired")]
    assert not {r["source_id"] for r in unpaired} & pos_src
    assert len({r["id"] for r in ROWS}) == len(ROWS) and all(r["id"].startswith("sa2_train_") for r in ROWS)


def test_questions_are_balanced_within_each_kind():
    for kind, rows in BY.items():
        counts = Counter(r["question"] for r in rows)
        assert set(counts) == set(v2.QUESTIONS) and max(counts.values()) - min(counts.values()) <= 1, kind


def test_positives_recompute_and_state_one_matching_band_and_category():
    for r in BY["positive"]:
        args, res = r["tool_calls"][0]["arguments"], r["tool_calls"][0]["result"]
        assert v2.independent_egfr(args["creatinine_mg_dl"], args["age"], args["sex"]) == res
        stage = v2.v1.g_stage(res)
        assert f"{res} mL/min/1.73m², {v2.BAND[stage]} (KDIGO GFR category {stage}," in r["answer"]
        assert sum(f"category {g}" in r["answer"] for g in v2.STAGES) == 1


def test_negatives_remove_the_cue_and_say_what_is_documented():
    for r in BY["negative_age"]:
        assert extract_age_sex(r["note"])[0] is None and not r["stretch_a"]["residue"]
        assert r["answer_type"] == "uncertain" and "age is not recorded" in r["answer"] and "sex (" in r["answer"]
    for r in BY["negative_sex"]:
        assert extract_age_sex(r["note"])[1] is None and not r["stretch_a"]["residue"]
        assert "sex is not recorded" in r["answer"] and "age (" in r["answer"]


def test_evaluation_sets_are_disjoint_from_training_sources():
    val = {json.loads(x)["source_id"] for x in (PROJECT_ROOT / "data" / "stretch_a" / "val_egfr.jsonl")
           .read_text(encoding="utf-8").splitlines() if x.strip()}
    assert not val & {r["source_id"] for r in ROWS} and all(s.startswith("train_") for s in {r["source_id"] for r in ROWS})


def test_view_loader_checks_prefix_and_hash(tmp_path, monkeypatch):
    rows = egfr_view_additions("q5_relabeled_egfr2")
    assert len(rows) == 200 and all(set(r) <= {"id", "note", "table", "question", "answer", "answer_type", "tool_calls"}
                                    for r in rows)
    assert EGFR_VIEWS["q5_relabeled_egfr"][2] == "sa_train_" and EGFR_VIEWS["q5_relabeled_egfr2"][2] == "sa2_train_"
    from clinqa import data_views
    bad = tmp_path / "train_additions.jsonl"
    bad.write_text((DATA / "train_additions.jsonl").read_text(encoding="utf-8").replace("sa2_train_000", "sa2_train_x"),
                   encoding="utf-8")
    with pytest.raises(ValueError, match="frozen"):
        data_views.stretch_a_additions(str(bad), str(DATA / "manifest.json"), "sa2_train_")


def test_prompt_v1e2_only_adds_the_kdigo_table():
    v1e = (PROJECT_ROOT / "configs" / "prompts" / "system_v1e.txt").read_text(encoding="utf-8")
    v1e2 = (PROJECT_ROOT / "configs" / "prompts" / "system_v1e2.txt").read_text(encoding="utf-8")
    line = " KDIGO GFR categories (mL/min/1.73m²): G1 ≥90, G2 60–89, G3a 45–59, G3b 30–44, G4 15–29, G5 <15."
    assert v1e2.replace(line, "") == v1e
