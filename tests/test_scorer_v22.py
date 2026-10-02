"""Scorer v2.2 prototype: numeric contradiction guards on top of the frozen v2.1 (scripts/scorer_v22.py)."""

import importlib.util
import json

import pytest

from clinqa import scorer_v2
from clinqa.config import PROJECT_ROOT

_spec = importlib.util.spec_from_file_location("scorer_v22", PROJECT_ROOT / "scripts" / "scorer_v22.py")
s = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s)

IRON = {"type": "labs", "headers": ["Test", "Value", "Unit", "Reference Range"],
        "rows": [["Serum Iron", "40.0", "μg/dL", "60-170"], ["TIBC", "473.2", "μg/dL", "250-370"],
                 ["Ferritin", "6.8", "ng/mL", "12-300"], ["Creatinine", "1.2", "mg/dL", "0.7-1.3"]]}
REC = {"id": "t", "note": "Follow-up visit.", "table": IRON, "answer_type": "numeric_reasoning",
       "question": "By how much does the patient's TIBC exceed the upper limit of the normal reference range?",
       "answer": "The TIBC of 473.2 μg/dL exceeds the upper limit (370 μg/dL) by 103.2 μg/dL."}
GOOD = "The TIBC of 473.2 μg/dL exceeds the upper limit of normal (370 μg/dL) by 103.2 μg/dL."


def _score(text):
    return s.score(REC, {"final_answer": text, "turns": []})


def test_correct_answer_passes_and_matches_v21():
    out = _score(GOOD)
    assert out["correct"] and out["v21_correct"] and out["guards"] == [] and out["scorer"] == "2.2"


@pytest.mark.parametrize("extra,guard", [
    ("The ferritin of 6.8 ng/mL is below the lower limit (12 ng/mL) by 4.2 ng/mL.", "amount(Ferritin"),
    ("Creatinine is also elevated at 1.2 mg/dL.", "state(Creatinine"),
])
def test_false_extra_claim_fails_only_under_v22(extra, guard):
    out = _score(f"{GOOD} {extra}")
    assert out["v21_correct"] and not out["correct"]
    assert out["error"].startswith(f"guard:{guard}")


@pytest.mark.parametrize("extra", [
    "The ferritin of 6.8 ng/mL is below the lower limit (12 ng/mL) by 5.2 ng/mL.",  # correct amount
    "Ferritin is low at 6.8 ng/mL, and creatinine is within normal limits.",  # correct states
    "Serum iron is below the lower limit by 60 - 40.0 = 20.0 μg/dL.",  # equation operand
    "Ferritin is 1.8 times below the lower limit.",  # fold amount, not a difference
    "Ferritin is 5.2 ng/mL (or 5.2 μg/L) below the lower limit.",  # restated in another unit
    "Dividing 473.2 by 370 gives about 1.28.",  # division
    "Creatinine is 1.2 mg/dL, below the upper limit.",  # bound-relative, not a state claim
])
def test_valid_extras_do_not_fire(extra):
    out = _score(f"{GOOD} {extra}")
    assert out["correct"], out["guards"]


def test_guards_never_turn_a_fail_into_a_pass():
    out = _score("The TIBC exceeds the upper limit by 90 μg/dL.")
    assert not out["v21_correct"] and not out["correct"]


def test_other_answer_types_are_scored_exactly_as_v21():
    rec = {**REC, "answer_type": "extractive"}
    traj = {"final_answer": f"{GOOD} Creatinine is also elevated at 1.2 mg/dL.", "turns": []}
    a, b = scorer_v2.score(rec, traj), s.score(rec, traj)
    assert {k: v for k, v in b.items() if k != "scorer"} == {k: v for k, v in a.items() if k != "scorer"}


@pytest.mark.parametrize("split", ["train", "val"])
def test_gold_self_check_never_fires(split):
    for line in (PROJECT_ROOT / "data" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["answer_type"] == "numeric_reasoning":
            assert s.score(r, s.gold_trajectory(r))["guards"] == [], r["id"]
