"""Synthetic metamorphic/adversarial tests; no test-split fixtures or gold-copy oracle."""
from dataclasses import asdict
import pytest

from clinqa.scoring_v2 import compile_contract, score_v2, strict_abstention, state_evidence


def record(question="What is the glucose level, and is it within the normal reference range?", kind="extractive"):
    return {"id": "synthetic", "answer_type": kind, "question": question,
            "note": "Weight 80 kg, height 180 cm.", "answer": "An intentionally irrelevant reference explanation.",
            "table": {"type": "labs", "rows": [["Glucose (fasting)", "152.4", "mg/dL", "70-100"],
                       ["Potassium", "3.0", "mmol/L", "3.5-5.0"]]}, "tool_calls": []}


def trajectory(text):
    return {"final_answer": text, "turns": [{"status": "no_call", "calls": [], "results": []}], "stop_reason": "answer"}


def result(text, r=None):
    return score_v2(r or record(), trajectory(text))


@pytest.mark.parametrize("text", [
    "The glucose is 152.4 mg/dL, above the reference range.",
    "Glucose is 152.4 mg/dL. It is not within the normal range.",
    "Glucose is 152.4 mg/dL. This result falls outside the reference limits.",
    "Glucose: 152.4 mg/dL. This is elevated.",
    "The fasting glucose value is 152.4 mg/dL. No, it is not in the normal range.",
])
def test_paraphrase_and_sentence_invariance(text):
    assert result(text)["status"] == "pass"


@pytest.mark.parametrize("text", [
    "Glucose is 152.4 mg/dL. It is within the normal range.",
    "Glucose is 125.4 mg/dL. It is elevated.",
    "Glucose is 152.4 mg/dL. It is elevated. It is normal.",
])
def test_false_value_or_contradiction_fails(text):
    assert result(text)["status"] == "fail"


def test_polarity_is_set_complement():
    assert {"low", "high"} in state_evidence("not within the normal range")
    assert {"low", "normal"} in state_evidence("not elevated")
    assert {"normal"} not in state_evidence("not within the normal range")


def test_unfamiliar_language_is_unresolved_not_wrong():
    assert result("Glucose is 152.4 mg/dL. This result is beyond the expected band.")["status"] == "review"


def test_analyte_binding_not_global_bag_of_words():
    r = record()
    assert result("Glucose is 152.4 mg/dL. Potassium is low at 3.0 mmol/L.", r)["status"] == "review"
    assert result("Glucose is 152.4 mg/dL and normal. Potassium is low at 3.0 mmol/L.", r)["status"] == "fail"


def test_yes_no_does_not_require_optional_value():
    r = record("Is glucose within the normal reference range?")
    assert result("No.", r)["status"] == "pass"
    assert result("Yes.", r)["status"] == "fail"


def test_gold_explanation_cannot_change_numeric_contract():
    r = record("By how much does glucose exceed the upper reference limit?", "numeric_reasoning")
    before = asdict(compile_contract(r))
    r["answer"] = "Extra explanation mentions 52.4, 52%, 30 and 1.73."
    assert asdict(compile_contract(r)) == before
    assert result("Glucose exceeds the upper limit by 52.4 mg/dL.", r)["status"] == "pass"
    assert result("It is 52.4 mg/dL above the upper limit of 100.", r)["status"] == "pass"
    assert result("It is 52.4 mg/dL below the upper limit.", r)["status"] == "fail"
    assert result("It is 42.4 mg/dL above the upper limit.", r)["status"] == "fail"


def test_wrong_analyte_cannot_supply_correct_derived_number():
    r = record("By how much does glucose exceed the upper reference limit?", "numeric_reasoning")
    assert result("Potassium is 52.4 mg/dL above its upper limit.", r)["status"] != "pass"


def test_optional_percentage_not_mandatory_but_requested_percentage_is():
    r = record("By how much does glucose exceed its upper reference limit?", "numeric_reasoning")
    assert result("Glucose exceeds the limit by 52.4 mg/dL.", r)["status"] == "pass"
    r["question"] += " What is the percentage above the upper limit?"
    assert result("Glucose exceeds the limit by 52.4 mg/dL.", r)["status"] == "review"
    assert result("Glucose exceeds the limit by 52.4 mg/dL, or 52.4% above it.", r)["status"] == "pass"


def test_clinical_inference_negation_is_not_refusal():
    assert not strict_abstention("This does not indicate tachycardia.")
    assert strict_abstention("The height is not documented, so BMI cannot be calculated.")


def test_ambiguous_ranking_needs_review():
    r = record("Which lab is most significantly abnormal and by how much?", "numeric_reasoning")
    c = compile_contract(r)
    assert c.review_reasons and not c.slots
    assert result("Glucose.", r)["correct"] is None


def test_false_extra_claim_fails():
    assert result("Glucose is 152.4 mg/dL and high. All other values are normal.")["status"] == "fail"


def test_missing_tool_inputs_are_not_forced_to_match_bad_gold():
    r = record("Calculate BMI.", "tool_call")
    r["note"] = "Height 180 cm."
    r["tool_calls"] = [{"tool": "calculate_bmi", "arguments": {"weight_kg": 80, "height_cm": 180}, "result": 24.7}]
    assert result("Height is 180 cm, but weight is not documented; BMI cannot be calculated.", r)["status"] == "pass"
    t = trajectory("BMI is 24.7.")
    t["turns"].insert(0, {"status": "valid", "calls": [{"name": "calculate_bmi", "arguments": r["tool_calls"][0]["arguments"]}], "results": [24.7]})
    assert score_v2(r, t)["status"] == "fail"


def test_open_note_fact_not_lexical_f1():
    r = record("What imaging study was ordered?")
    assert result("A completely unrelated fluent statement.", r)["status"] == "review"


def test_strict_reference_boundary():
    r = record("Is glucose within the normal reference range?")
    r["table"]["rows"][0] = ["Glucose (fasting)", "100", "mg/dL", "<100"]
    assert result("No, glucose is not within the reference range.", r)["status"] == "pass"


def test_units_are_not_interchangeable():
    assert result("Glucose is 152.4 mmol/L, above the range.")["status"] == "fail"


def test_reference_definition_is_not_a_patient_assertion():
    assert result("Glucose is 152.4 mg/dL. The normal reference range for glucose is 70-100 mg/dL. It is not within that range.")["status"] != "fail"
    assert result("Glucose is 152.4 mg/dL. This is above the upper limit of normal.")["status"] == "pass"


def test_note_range_is_valid_evidence_when_table_only_contains_vitals():
    r = record()
    r["table"]["rows"] = [["Heart Rate", "85", "bpm"]]
    r["note"] = "Labs: Glucose (fasting) 152.4 mg/dL (ref 70-100)."
    assert result("Glucose is 152.4 mg/dL. It is not within the reference range.", r)["status"] == "pass"


def test_optional_false_equation_is_not_ignored():
    r = record("By how much does glucose exceed the upper reference limit?", "numeric_reasoning")
    assert result("Glucose exceeds the limit by 52.4 mg/dL. Also, 5 - 2 = 4.", r)["status"] == "fail"
    assert result("Glucose exceeds the upper limit by 152.4 - 100 = 52.4 mg/dL.", r)["status"] == "pass"


def test_percent_equations_and_precedence():
    r = record("By how much does glucose exceed the upper reference limit?", "numeric_reasoning")
    assert result("Glucose exceeds the limit by 52.4 mg/dL; 52.4 / 100 = 52.4%.", r)["status"] == "pass"
    assert result("Glucose exceeds the limit by 52.4 mg/dL; 52.4 / 100 × 100 = 52.4%.", r)["status"] == "pass"


def test_wrong_optional_measurement_is_not_ignored():
    assert result("Glucose is 152.4 mg/dL, above the range. Potassium is 8.0 mmol/L.")["status"] == "fail"


def test_different_operations_bind_to_different_question_clauses():
    r = record("By how much does glucose exceed its upper limit, and by what factor is potassium below its lower limit?", "numeric_reasoning")
    slots = compile_contract(r).slots
    assert [(s.kind, s.subject) for s in slots] == [("difference", "Glucose (fasting)"), ("ratio", "Potassium")]


def test_extra_gold_numbers_and_gold_direction_do_not_affect_extractive():
    r = record()
    first = result("Glucose is 152.4 mg/dL and elevated.", r)
    r["answer"] = "This is low and relates to an unasked 25% risk and a dose of 10 mg."
    assert result("Glucose is 152.4 mg/dL and elevated.", r)["status"] == first["status"]


def test_conflicting_inputs_are_not_silently_resolved():
    r = record()
    r["note"] = "Glucose (fasting) 92.4 mg/dL (ref 70-100)."
    assert result("Glucose is 152.4 mg/dL and high.", r)["status"] == "review"


def test_self_correction_gets_review_and_not_a_confident_false_label():
    assert result("Glucose is 152.4 mg/dL, normal. Correction: it is outside the range.")["status"] == "review"


def test_correct_difference_does_not_license_wrong_reference():
    r = record("By how much does glucose exceed the upper reference limit?", "numeric_reasoning")
    assert result("Glucose is 52.4 mg/dL above the upper limit of 120 mg/dL.", r)["status"] == "fail"


def test_percent_point_difference_not_relative_percentage():
    r = record("By how far does HbA1c exceed the upper reference limit?", "numeric_reasoning")
    r["table"]["rows"] = [["HbA1c", "11.6", "%", "<5.7"]]
    assert result("HbA1c is 11.6% - 5.7% = 5.9% above the upper limit.", r)["status"] == "pass"


def test_parenthetical_calculation_is_not_an_asserted_reference_boundary():
    r = record("By how much does glucose exceed the upper reference limit?", "numeric_reasoning")
    assert result("Glucose is 52.4 mg/dL above the upper limit of normal (152.4 - 100 = 52.4).", r)["status"] == "pass"


def test_tool_gold_agreement_does_not_override_grounding_disagreement():
    r = record("Calculate BMI.", "tool_call")
    r["tool_calls"] = [{"tool": "calculate_bmi", "arguments": {"weight_kg": 90, "height_cm": 180}, "result": 27.8}]
    t = trajectory("BMI is 27.8.")
    t["turns"].insert(0, {"status": "valid", "calls": [{"name": "calculate_bmi", "arguments": r["tool_calls"][0]["arguments"]}], "results": [27.8]})
    assert score_v2(r, t)["status"] == "review"
