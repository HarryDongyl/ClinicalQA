"""Clinical-context check, call-prefix ECE and numeric audit strata (INTERVIEW_PREP.md section 5.13)."""

import importlib.util

import pytest

from clinqa.config import PROJECT_ROOT


def _script(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cc = _script("clinical_context")
an = _script("w3_analyze")
prep = _script("numeric_audit_prep")

LABS = {"type": "labs", "headers": ["Test", "Value", "Unit", "Reference Range"],
        "rows": [["Creatinine", "1.1", "mg/dL", "0.7-1.3"], ["BUN", "18.0", "mg/dL", "7-20"]]}
REC = {"id": "t", "note": "Follow-up.", "table": LABS, "question": "q", "answer": "a", "answer_type": "tool_call"}
BMI = {"name": "calculate_bmi", "arguments": {"weight_kg": 80.0, "height_cm": 170.0}}
CONV = {"name": "unit_convert", "arguments": {"value": 1.1, "from_unit": "mg/dL", "to_unit": "μmol/L",
                                               "substance": "creatinine"}}


@pytest.mark.parametrize("bmi,cats", [(17.0, {"underweight"}), (22.0, {"normal"}), (27.7, {"overweight"}),
                                      (31.0, {"obese"}), (24.9, {"normal", "overweight"})])
def test_who_categories_with_rounding_margin(bmi, cats):
    assert cc.who(bmi) == cats


def test_bmi_category_required_and_commentary_ignored():
    ok = "The BMI is 27.7 kg/m², which classifies the patient as overweight. Obesity is a common driver of HFpEF."
    assert cc.check(REC, BMI, 27.7, ok)[0]
    assert cc.check(REC, BMI, 27.7, "The BMI is 27.7 kg/m².") == (False, "bmi:no_category")
    assert not cc.check(REC, BMI, 18.7, "The BMI is 18.7 kg/m², which falls in the underweight category.")[0]


def test_reference_range_parenthetical_is_not_a_category():
    text = "The BMI is 17.8 kg/m², which classifies him as underweight (normal range 18.5–24.9)."
    assert cc.check(REC, BMI, 17.8, text)[0]


def test_conversion_status_must_match_input():
    good = "The serum creatinine of 1.1 mg/dL converts to 97.26 µmol/L. Despite falling within the normal range, eGFR is low."
    assert cc.check(REC, CONV, 97.26, good)[0]
    assert not cc.check(REC, CONV, 97.26, "Creatinine 1.1 mg/dL converts to 97.26 µmol/L, which is elevated.")[0]
    assert cc.check(REC, CONV, 97.26, "Creatinine 1.1 mg/dL converts to 97.26 µmol/L.") == (False, "convert:no_status")


def test_ece():
    assert an.ece([1.0, 0.0], [True, False]) == 0.0
    assert an.ece([0.9, 0.9], [False, False]) == 0.9
    assert an.ece([], []) is None


def test_numeric_strata():
    agree = {"checks": [{"kind": "diff", "gold_agrees": True}], "coverage": "structured"}
    dispute = {"checks": [{"kind": "diff", "gold_agrees": False}], "coverage": "structured"}
    partial = {"checks": [{"kind": "diff", "gold_agrees": True}], "coverage": "partial"}
    assert prep.stratum(agree) == "key_agrees_gold"
    assert prep.stratum(dispute) == "key_disputes_gold"
    assert prep.stratum(partial) == "open_or_partial"
