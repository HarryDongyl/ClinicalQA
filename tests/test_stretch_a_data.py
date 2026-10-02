"""Stretch A data generator: CKD-EPI 2021 reference values, KDIGO stages, note edits, frozen outputs."""

import importlib.util
import json

import pytest

from clinqa.config import PROJECT_ROOT
from clinqa.parsing import extract_age_sex

spec = importlib.util.spec_from_file_location("stretch_a_data", PROJECT_ROOT / "scripts" / "stretch_a_data.py")
sa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa)


@pytest.mark.parametrize("cr,age,sex,egfr", [(1.0, 50, "male", 92), (1.0, 50, "female", 69), (0.7, 40, "female", 112),
                                             (2.0, 70, "male", 35)])
def test_ckd_epi_2021_reference_values(cr, age, sex, egfr):
    assert sa.ckd_epi_2021(cr, age, sex) == egfr


@pytest.mark.parametrize("egfr,stage", [(90, "G1"), (89, "G2"), (60, "G2"), (59, "G3a"), (44, "G3b"), (29, "G4"),
                                        (14, "G5")])
def test_kdigo_stage(egfr, stage):
    assert sa.g_stage(egfr) == stage


def test_age_and_sex_removal():
    note = "HPI: 54-year-old male presenting. He reports that his symptoms started. She told her doctor. Mr. L denies."
    no_age = sa.remove_age(note)
    assert extract_age_sex(no_age)[0] is None and not sa.AGE_RESIDUE.search(no_age)
    no_sex = sa.remove_sex(note)
    assert extract_age_sex(no_sex)[1] is None and not sa.SEX_RESIDUE.search(no_sex)
    assert "54-year-old" in no_sex


def test_durations_are_not_age_residue():
    assert not sa.AGE_RESIDUE.search("Ischemic stroke 2 years ago; quit smoking 10 years ago.")


def test_generated_data_is_consistent():
    d = PROJECT_ROOT / "data" / "stretch_a"
    if not d.exists():
        pytest.skip("generate with scripts/stretch_a_data.py")
    for name in ("train_additions.jsonl", "val_egfr.jsonl"):
        for line in (d / name).read_text().splitlines():
            r = json.loads(line)
            if r["answer_type"] == "tool_call":
                a = r["tool_calls"][0]["arguments"]
                assert r["tool_calls"][0]["result"] == sa.ckd_epi_2021(a["creatinine_mg_dl"], a["age"], a["sex"])
                assert str(r["tool_calls"][0]["result"]) in r["answer"]
                assert not any(row[0] == "eGFR" for row in r["table"]["rows"])
