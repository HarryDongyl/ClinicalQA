"""Scorer v2 regression tests: one fixture per failure mode found in the wave-1 audit (docs/SCORER_V2.md)."""

import json

import pytest

from clinqa import scorer_v2 as s2
from clinqa.config import PROJECT_ROOT

LIPIDS = {"type": "labs", "headers": ["Test", "Value", "Unit", "Reference Range"],
          "rows": [["Total Cholesterol", "238.7", "mg/dL", "<200"], ["LDL", "219.3", "mg/dL", "<100"],
                   ["HDL", "45.0", "mg/dL", ">40"], ["Triglycerides", "252.4", "mg/dL", "<150"]]}
THYROID = {"type": "labs", "headers": ["Test", "Value", "Unit", "Reference Range"],
           "rows": [["TSH", "2.7", "mIU/L", "0.27-4.20"], ["Free T4", "4.0", "ng/dL", "0.93-1.70"],
                    ["Free T3", "6.1", "pg/mL", "2.0-4.4"]]}
VITALS = {"type": "vitals", "headers": ["Vital", "Value", "Unit"],
          "rows": [["Blood Pressure", "150/92", "mmHg"], ["Heart Rate", "88", "bpm"], ["Temperature", "98.6", "°F"],
                   ["Respiratory Rate", "16", "breaths/min"], ["SpO2", "97", "%"]]}


def rec(question, answer, table=THYROID, at="extractive", note="Follow-up visit.", **kw):
    return {"id": "t", "note": note, "table": table, "question": question, "answer": answer, "answer_type": at, **kw}


def traj(text, turns=None, stop="answer"):
    return {"final_answer": text, "turns": turns or [], "stop_reason": stop}


def ok(record, pred):
    return s2.score(record, traj(pred))["correct"]


Q_T3 = "Is the patient's Free T3 level within the normal reference range?"
G_T3 = "No. The patient's Free T3 is 6.1 pg/mL, which is above the normal reference range of 2.0-4.4 pg/mL."


def test_key_is_built_from_input_not_gold():
    wrong_gold = "The LDL exceeds the limit by 19.3 mg/dL."
    k = s2.build_key(rec("By how much does the patient's LDL exceed the upper limit of normal?", wrong_gold, LIPIDS,
                         "numeric_reasoning"))
    diff = next(c for c in k.checks if c.kind == "diff")
    assert diff.gt == pytest.approx(119.3) and diff.gold_agrees is False


def test_direction_in_a_different_sentence_than_the_value():
    pred = ("No, the Free T3 level is not within the normal reference range. The Free T3 value is 6.1 pg/mL, but the "
            "reference range is 2.0-4.4 pg/mL. This value is above the upper limit of normal.")
    assert ok(rec(Q_T3, G_T3), pred)


def test_not_within_normal_means_abnormal():
    assert ok(rec(Q_T3, G_T3), "The patient's Free T3 of 6.1 pg/mL is not within the normal range.")
    assert not ok(rec(Q_T3, G_T3), "The patient's Free T3 of 6.1 pg/mL is within the normal range.")


def test_below_upper_limit_and_within_normal_do_not_conflict():
    q = "What is the patient's TSH level, and is it within the normal range?"
    pred = "The TSH is 2.7 mIU/L, which is below the upper limit of 4.20 and within the normal reference range."
    assert ok(rec(q, "TSH is 2.7 mIU/L, within normal limits."), pred)


def test_wrong_status_fails():
    q = "What is the patient's Free T4 level, and is it within the normal range?"
    assert not ok(rec(q, "Free T4 is 4.0 ng/dL, elevated."), "Free T4 is 4.0 ng/dL, within the normal range 0.93-1.70.")


def test_unrequested_gold_numbers_are_not_required():
    q = "By how much does the patient's LDL exceed the upper limit of normal?"
    gold = "LDL is 219.3 mg/dL, 119.3 mg/dL above the <100 limit (2.19 times; 119.3% above), with TG at 252.4."
    assert ok(rec(q, gold, LIPIDS, "numeric_reasoning"), "It exceeds the upper limit by 119.3 mg/dL.")


def test_ratio_rounding_tolerated_but_wrong_arithmetic_fails():
    q = "By how many times the upper limit of normal is the patient's LDL elevated?"
    r = rec(q, "About 2.19 times.", LIPIDS, "numeric_reasoning")
    assert ok(r, "The LDL is approximately 2.2 times the upper limit of normal.")
    assert not ok(r, "The LDL is approximately 1.2 times the upper limit of normal.")
    q2 = "By how much does the patient's LDL exceed the upper limit of normal?"
    assert not ok(rec(q2, "119.3 mg/dL", LIPIDS, "numeric_reasoning"), "LDL is 19.3 mg/dL above its threshold.")


def test_most_abnormal_entity_accepts_both_defensible_scales_only():
    q = "Which lipid panel value is most significantly elevated above its upper reference limit?"
    r = rec(q, "LDL is most elevated.", LIPIDS, "numeric_reasoning")
    assert ok(r, "LDL is the most significantly elevated value at 219.3 mg/dL.")
    assert not ok(r, "HDL is the most significantly elevated value.")


def test_verbose_correct_answer_passes():
    pred = ("No, the patient's Free T3 level is 6.1 pg/mL, which is above the normal reference range of 2.0-4.4 pg/mL. "
            "This elevation may reflect thyroiditis, exogenous hormone use or a hyperthyroid state and should be "
            "interpreted together with the TSH of 2.7 mIU/L and the Free T4. Clinical correlation is recommended.")
    assert ok(rec(Q_T3, G_T3), pred)


def test_blanket_all_within_normal_answers_an_empty_set():
    q = "Are any of the patient's lipid values outside their reference ranges?"
    table = {**LIPIDS, "rows": [["Total Cholesterol", "180.0", "mg/dL", "<200"], ["LDL", "90.0", "mg/dL", "<100"],
                                ["HDL", "45.0", "mg/dL", ">40"], ["Triglycerides", "120.0", "mg/dL", "<150"]]}
    r = rec(q, "No, all are within range.", table, "numeric_reasoning")
    assert ok(r, "All four lipid panel values (TC 180.0, LDL 90.0, HDL 45.0, TG 120.0) are within their ranges.")
    assert not ok(r, "The LDL is elevated at 90.0 mg/dL.")


def _bmi_record(note, args, result):
    return rec("What is the patient's BMI?", f"BMI is {result}.", VITALS, "tool_call", note=note,
               tool_calls=[{"tool": "calculate_bmi", "arguments": args, "result": result}])


def _call_traj(args, result, text):
    return traj(text, turns=[{"status": "valid", "calls": [{"name": "calculate_bmi", "arguments": args}],
                              "results": [result]}, {"status": "no_call", "calls": [], "results": []}])


def test_tool_outcome_tolerates_imperial_rounding():
    r = _bmi_record("Weight 241.8 lb, height 64.9 in.", {"weight_kg": 109.7, "height_cm": 164.9}, 40.3)
    t = _call_traj({"weight_kg": 110.0, "height_cm": 165.0}, 40.4, "The BMI is 40.4 kg/m2, class III obesity.")
    assert s2.score(r, t)["correct"]


def test_tool_with_invented_arguments_fails():
    r = _bmi_record("Weight 241.8 lb, height 64.9 in.", {"weight_kg": 109.7, "height_cm": 164.9}, 40.3)
    t = _call_traj({"weight_kg": 78.5, "height_cm": 178.5}, 24.6, "The BMI is 24.6 kg/m2.")
    assert s2.score(r, t)["error"] == "ungrounded_args"


def test_q5_gold_without_measurements_expects_abstention():
    r = _bmi_record("No anthropometrics were recorded today.", {"weight_kg": 78.5, "height_cm": 178.5}, 24.6)
    assert s2.build_key(r).expected == "abstain"
    t = _call_traj({"weight_kg": 78.5, "height_cm": 178.5}, 24.6, "The BMI is 24.6 kg/m2.")
    assert not s2.score(r, t)["correct"]
    assert s2.score(r, traj("BMI cannot be calculated because weight and height are not documented."))["correct"]


def test_uncertain_fabrication_fails():
    r = rec("What is the patient's BMI?", "Height is not documented, so BMI cannot be calculated.", VITALS, "uncertain",
            note="Weight 80 kg.")
    assert s2.score(r, traj("Height is not documented in the note, so BMI cannot be calculated."))["correct"]
    assert not s2.score(r, traj("With a height of 170 cm the BMI is 27.7."))["correct"]


@pytest.mark.parametrize("split", ["train", "val"])
def test_gold_answers_pass_their_own_keys(split):
    """Gold-as-prediction self-check (MedCalc-style keys): >= 97% of answerable records pass."""
    rows = [json.loads(line) for line in (PROJECT_ROOT / "data" / f"{split}.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["answer_type"] in ("extractive", "numeric_reasoning")]
    passed = sum(s2.score(r, traj(r["answer"]))["correct"] for r in rows)
    assert passed / len(rows) >= 0.97
