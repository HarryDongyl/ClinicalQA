import pytest

from clinqa import parsing as P
from clinqa.analysis.features import classify_uncertain


def _ms(text):
    return [(m.kind, m.value, m.unit, m.is_delta, m.equivalent) for m in P.extract_body_measurements(text)]


def test_labeled_imperial_and_metric():
    assert _ms("Weight 116.6 lb, Height 150.8 cm.") == [("weight", 116.6, "lb", False, False), ("height", 150.8, "cm", False, False)]


def test_unlabeled_parenthetical_pair():
    assert _ms("Obese male (109.7 kg, 70.9 in), alert.") == [("weight", 109.7, "kg", False, False), ("height", 70.9, "in", False, False)]


def test_equivalent_restatement_is_marked():
    out = _ms("Weight 66.0 kg, Height 66.0 in (167.6 cm).")
    assert out[-1] == ("height", 167.6, "cm", False, True)


@pytest.mark.parametrize(
    "text",
    [
        "approximately 8 lbs of weight gain over the past month",
        "unintentional weight loss of 8 kg over the past 3 months",
        "has gained approximately 5 lbs over 2 weeks",
    ],
)
def test_weight_changes_are_deltas(text):
    assert all(m.is_delta for m in P.extract_body_measurements(text))


@pytest.mark.parametrize(
    "text",
    [
        "Motor strength 5/5 in all extremities.",
        "liver edge palpable 2 cm below the costal margin",
        "BMI 30.7 kg/m² documented",
        "reduce dose if weight ≤60 kg",
        "Right foot with a 3 x 2 cm ulcer",
    ],
)
def test_non_measurements_are_ignored(text):
    assert P.extract_body_measurements(text) == []


def test_bare_in_followed_by_word_is_not_a_height():
    assert P.extract_body_measurements("seen 65 in the clinic") == []


def test_inline_bmi():
    assert P.extract_inline_bmi("Weight 115.1 kg, Height 75.6 in (BMI ~29.3 kg/m²).") == [29.3]
    assert P.extract_inline_bmi("Will calculate BMI and consider nutrition.") == []


@pytest.mark.parametrize(
    "note, expected",
    [
        ("Allergies: NKDA.\n", "nkda"),
        ("**Allergies:** Not documented.\n", "not_documented"),
        ("Allergies: Penicillin (rash).\n", "documented"),
        ("Medications: none.\n", "absent"),
    ],
)
def test_allergy_status(note, expected):
    assert P.allergy_status(note) == expected


def test_medications_single_line_and_bullets():
    note = "Medications: Acetaminophen 500mg PRN, trazodone at bedtime.\n\nAllergies: NKDA."
    meds = P.extract_medications(note)
    assert [(m.name, m.has_dose) for m in meds] == [("acetaminophen", True), ("trazodone", False)]
    note = "MEDICATIONS:\n- Clopidogrel 75mg daily\n- Levetiracetam (dose not confirmed)\n\nVITALS: See table."
    assert [(m.name, m.has_dose) for m in P.extract_medications(note)] == [("clopidogrel", True), ("levetiracetam", False)]


def test_note_analyte_values_skip_differences_and_combined_labels():
    assert P.note_analyte_values("Notable for elevated creatinine at 3.6 mg/dL.", "Creatinine") == ["3.6"]
    assert P.note_analyte_values("calcium 1.3 mg/dL above upper limit of normal", "Calcium") == []
    assert P.note_analyte_values("PT/INR elevated at 17.9 s / 1.8", "INR") == []
    assert P.note_analyte_values("BP 113/83 mmHg, HR 79 bpm", "Blood Pressure") == ["113/83"]
    assert P.note_analyte_values("RR 18/min", "Respiratory Rate") == ["18"]


def test_ref_range_and_panel():
    assert P.parse_ref_range("70-100") == {"low": 70.0, "high": 100.0}
    assert P.parse_ref_range("<200") == {"low": None, "high": 200.0}
    assert P.parse_ref_range(">60") == {"low": 60.0, "high": None}
    assert P.parse_ref_range("n/a") is None
    table = {"type": "labs", "rows": [["TSH", "1", "mIU/L", "x"], ["Free T4", "1", "", "x"], ["Free T3", "1", "", "x"]]}
    assert P.table_panel(table) == "thyroid"


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("The patient's weight is documented at 77.5 kg; however, her height is not recorded in the note. "
         "BMI cannot be calculated without height (BMI = weight in kg / height in m²).", "height"),
        ("BMI cannot be calculated. The height is recorded as 182.2 cm; however, the weight is not documented anywhere. "
         "Both height and weight are required.", "weight"),
        ("The note lists sitagliptin; however, the dose is not documented.", "dose"),
        ("The clinical note does not document any allergy information.", "allergy"),
        ("However, no collection date or time is recorded for these labs.", "timestamp"),
    ],
)
def test_classify_uncertain(answer, expected):
    assert classify_uncertain("", answer) == expected
