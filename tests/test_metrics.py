"""Scorer fixtures (PLAN section 7): positive and negative cases for every rule.

Synthetic records keep the fixtures independent of the dataset; the gold-copy test at
the bottom checks the scorer against real train/val gold.
"""

import pytest

from clinqa.data_io import load_split
from clinqa.metrics import gold_trajectory, is_abstention, numbers, number_matches, score_example

NOTE = ("CC: fatigue. HPI: 58-year-old woman. Vitals: HR 117 bpm, BP 150/92. Weight 234.6 lb, height 74.5 in. "
        "Medications: metformin, aspirin daily. Assessment: T2DM.")
LABS = {"type": "labs", "headers": ["Test", "Value", "Unit", "Reference Range"],
        "rows": [["Glucose (fasting)", "150.4", "mg/dL", "70-100"], ["Creatinine", "0.7", "mg/dL", "0.6-1.2"],
                 ["Potassium", "3.1", "mmol/L", "3.5-5.0"]]}


def rec(answer_type, question, answer, tool_calls=None, note=NOTE, table=LABS):
    r = {"id": "fx", "note": note, "table": table, "question": question, "answer": answer, "answer_type": answer_type}
    if tool_calls:
        r["tool_calls"] = tool_calls
    return r


def answer(text, stop="answer"):
    return {"turns": [{"status": "no_call", "calls": [], "results": []}], "final_answer": text, "stop_reason": stop}


def call(name, args, result, final, status="valid", extra=None):
    turns = [{"status": status, "calls": [{"name": name, "arguments": args}], "results": [result]}]
    if extra:
        turns.append({"status": "valid", "calls": [extra], "results": [1.0]})
    turns.append({"status": "no_call", "calls": [], "results": []})
    return {"turns": turns, "final_answer": final, "stop_reason": "answer"}


EXT = rec("extractive", "What is the fasting glucose, and is it within the reference range?",
          "The fasting glucose is 150.4 mg/dL, which is above the reference range of 70-100 mg/dL.")
NUM = rec("numeric_reasoning", "By how much does the fasting glucose exceed the upper reference limit?",
          "The fasting glucose of 150.4 mg/dL exceeds the upper limit of 100 mg/dL by 50.4 mg/dL.")
UNC = rec("uncertain", "What is the current daily dose of aspirin?",
          "Aspirin is listed as a daily medication, but the dose is not documented in the note.")
UNC_ALLERGY = rec("uncertain", "Does the patient have any drug allergies?",
                  "The note contains no allergy documentation, so allergy status cannot be determined.")
BMI = rec("tool_call", "What is the patient's BMI?", "The BMI is 29.7 kg/m², in the overweight range.",
          [{"tool": "calculate_bmi", "arguments": {"weight_kg": 106.4, "height_cm": 189.2}, "result": 29.7}])
CONV = rec("tool_call", "Convert the fasting glucose to mmol/L.", "The fasting glucose is 8.35 mmol/L, elevated.",
           [{"tool": "unit_convert", "arguments": {"value": 150.4, "from_unit": "mg/dL", "to_unit": "mmol/L",
                                                    "substance": "glucose"}, "result": 8.35}])


@pytest.mark.parametrize("record,traj,correct,error", [
    # extractive
    (EXT, answer("Fasting glucose is elevated at 150.4 mg/dL (reference 70-100)."), True, "correct"),
    (EXT, answer("Fasting glucose is 150.4 mg/dL, which is higher than normal."), True, "correct"),
    (EXT, answer("Fasting glucose is 154.0 mg/dL, above the reference range."), False, "number_mismatch"),
    (EXT, answer("Fasting glucose is 150.4 mg/dL, within the normal range."), False, "direction_mismatch"),
    (EXT, answer("Creatinine is 0.7 mg/dL, within the reference range."), False, "number_mismatch"),  # swapped analyte
    (EXT, answer("The glucose value is not documented in the note."), False, "over_refusal"),
    (EXT, answer(""), False, "no_answer_answer"),
    (EXT, answer("Values: 150.4 70 100 0.7 0.6 1.2 3.1 3.5 5.0 117 150 92 above"), False, "number_dump"),
    # numeric
    (NUM, answer("It is 50.4 mg/dL above the upper limit of 100 mg/dL."), True, "correct"),
    (NUM, answer("It exceeds the upper limit by 50 mg/dL."), False, "number_mismatch"),  # lower precision
    (NUM, answer("The glucose of 150.4 mg/dL is 50.4 mg/dL below the upper limit."), False, "direction_mismatch"),
    # uncertain
    (UNC, answer("The aspirin dose is not specified in the note, so the daily dose cannot be determined."), True,
     "correct"),
    (UNC, answer("The patient takes aspirin 81 mg daily."), False, "fabricated_value"),
    (UNC, answer("Aspirin is taken daily."), False, "no_abstention"),
    (UNC, answer("This information is not documented."), False, "missing_field_not_named"),
    (UNC, answer("The dose is not documented (e.g., 81 mg vs 325 mg), so it cannot be confirmed."), True, "correct"),
    (UNC_ALLERGY, answer("Allergies are not documented; this does not mean NKDA."), True, "correct"),
    (UNC_ALLERGY, answer("The patient has no known drug allergies."), False, "fabricated_value"),
    # tool calls
    (BMI, call("calculate_bmi", {"weight_kg": 106.4, "height_cm": 189.2}, 29.7, "BMI is 29.7 kg/m²."), True, "correct"),
    # exact imperial conversion (234.6 lb -> 106.41 kg, 74.5 in -> 189.23 cm) is accepted
    (BMI, call("calculate_bmi", {"weight_kg": 106.41, "height_cm": 189.23}, 29.7, "BMI is 29.7."), True, "correct"),
    (BMI, call("calculate_bmi", {"weight_kg": 234.6, "height_cm": 74.5}, 297.2, "BMI is 297.2."), False,
     "wrong_args_imperial"),
    (BMI, call("unit_convert", {"value": 234.6, "from_unit": "lb", "to_unit": "kg", "substance": None}, 106.41,
               "Weight is 106.41 kg."), False, "wrong_tool"),
    (BMI, answer("The BMI is about 29.7 kg/m²."), False, "no_call"),
    (BMI, answer("Height is not documented, so BMI cannot be calculated."), False, "no_call_abstained"),
    (BMI, call("calculate_bmi", {"weight_kg": 106.4, "height_cm": 189.2}, 29.7, "The BMI is elevated."), False,
     "result_not_in_answer"),
    (BMI, call("calculate_bmi", {"weight_kg": "106.4", "height_cm": 189.2}, 29.7, "BMI 29.7", status="schema_error"),
     False, "schema_error"),
    (BMI, {"turns": [{"status": "invalid_json", "calls": [], "results": []}], "final_answer": None,
           "stop_reason": "parse_error"}, False, "invalid_json"),
    (BMI, call("calculate_bmi", {"weight_kg": 106.4, "height_cm": 189.2}, 29.7, "BMI 29.7.",
               extra={"name": "calculate_bmi", "arguments": {"weight_kg": 1, "height_cm": 1}}), False, "extra_call"),
    (CONV, call("unit_convert", {"value": 150.4, "from_unit": "mg/dl", "to_unit": "mmol/l", "substance": "Glucose"},
                8.35, "Glucose is 8.35 mmol/L."), True, "correct"),
    (CONV, call("unit_convert", {"value": 150.4, "from_unit": "mg/dL", "to_unit": "mmol/L", "substance": "cholesterol"},
                3.9, "3.9 mmol/L."), False, "wrong_args_value"),
])
def test_fixture(record, traj, correct, error):
    s = score_example(record, traj)
    assert (s.correct, s.error) == (correct, error)


EXT_NORMAL = rec("extractive", "Is the creatinine within the reference range?",
                 "The creatinine is 0.7 mg/dL, which is within the normal reference range (less than 1.2).")
UNC_WEIGHT = rec("uncertain", "What is the patient's BMI?",
                 "The height is documented, but the weight is not recorded, so BMI cannot be calculated.",
                 note="CC: fatigue. HPI: 58-year-old woman. Height 165 cm. HR 88 bpm, weight loss of 5 kg.")


@pytest.mark.parametrize("record,traj,correct,error", [
    # adversarial-review regressions
    (EXT_NORMAL, answer("Creatinine is 0.7 mg/dL (in the normal range)."), True, "correct"),
    (EXT_NORMAL, answer("Creatinine is 0.7 mg/dL, which is not within the normal range."), False, "direction_mismatch"),
    (EXT, answer("Fasting glucose is 150.4 mg/dL, which is not elevated."), False, "direction_mismatch"),
    (EXT, answer("Fasting glucose is 150.4 mg/dL, above the reference range. She has a history of hypertension "
                 "and a lower limit of 70."), True, "correct"),
    (NUM, answer("The creatinine of 0.7 mg/dL exceeds the upper limit of 100 mg/dL by 50.4 mg/dL."), False,
     "number_mismatch"),  # derived number kept, analyte swapped
    (UNC_WEIGHT, answer("Weight isn't documented in the note, so BMI can't be computed."), True, "correct"),
    (UNC_WEIGHT, answer("The weight is not documented; her last clinic weight was 78, giving a BMI near 28."), False,
     "fabricated_value"),
    (UNC_WEIGHT, answer("Weight is not recorded (a 5 kg weight loss is noted), so BMI cannot be calculated."), True,
     "correct"),
    (UNC_ALLERGY, answer("The patient has no allergies. Allergy history is otherwise not documented."), False,
     "fabricated_value"),
])
def test_review_regressions(record, traj, correct, error):
    s = score_example(record, traj)
    assert (s.correct, s.error) == (correct, error)


def test_unsupported_arguments_flag():
    good = call("calculate_bmi", {"weight_kg": 106.4, "height_cm": 189.2}, 29.7, "BMI 29.7.")
    invented = call("calculate_bmi", {"weight_kg": 70.0, "height_cm": 189.2}, 19.6, "BMI 19.6.")
    assert score_example(BMI, good).unsupported_args is False
    assert score_example(BMI, invented).unsupported_args is True
    assert score_example(EXT, answer("150.4 mg/dL, above range.")).unsupported_args is None


def test_over_refusal_precision():
    assert not is_abstention("The fasting glucose is elevated; the cause is unknown at this time.")
    assert is_abstention("The dose is unknown because it is not in the note.")
    assert is_abstention("Allergy information is missing from the record.")


def test_over_call_and_over_refusal_flags():
    s = score_example(EXT, call("calculate_bmi", {"weight_kg": 1, "height_cm": 1}, 10000.0,
                                "Fasting glucose is 150.4 mg/dL, above range."))
    assert s.over_call is True and s.correct is True
    s = score_example(EXT, answer("Not documented."))
    assert s.over_refusal is True and s.over_call is False
    assert score_example(UNC, answer("The dose is not documented.")).over_refusal is None


def test_tool_submetrics_on_wrong_args():
    s = score_example(BMI, call("calculate_bmi", {"weight_kg": 100.0, "height_cm": 189.2}, 27.9, "BMI 27.9."))
    assert s.tool_selected and not s.tool_args_correct and s.tool_executed and not s.tool_e2e


def test_universal_refusal_fails_non_uncertain():
    refusal = answer("The information needed is not documented in the note, so this cannot be determined.")
    assert not score_example(EXT, refusal).correct
    assert not score_example(NUM, refusal).correct
    assert not score_example(BMI, refusal).correct


def test_number_precision():
    assert number_matches(numbers("2.3")[0], numbers("2.345"))
    assert not number_matches(numbers("2.35")[0], numbers("2.4"))
    assert [n.value for n in numbers("range 70-100, 1,200 and -3.5")] == [70, 100, 1200, -3.5]


def test_abstention_phrases():
    assert is_abstention("The weight is not recorded.")
    assert is_abstention("BMI cannot be calculated without height.")
    assert not is_abstention("The heart rate is 117 bpm.")


# Gold answers that themselves state values absent from the input (scorer finding, not a scorer bug).
KNOWN_GOLD_FABRICATIONS = {"train_352", "train_365", "train_913", "train_1670", "train_1801", "val_200"}


@pytest.mark.parametrize("split", ["train", "val"])
def test_gold_copy_scores_correct(split):
    failures = {r["id"]: s.error for r in load_split(split)
                for s in [score_example(r, gold_trajectory(r))] if not s.correct}
    assert set(failures) <= KNOWN_GOLD_FABRICATIONS, failures
