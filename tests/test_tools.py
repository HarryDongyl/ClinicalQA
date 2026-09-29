import pytest

from clinqa.data_io import load_split
from clinqa.tools import calculate_bmi, execute_tool, normalize_unit, unit_convert


@pytest.mark.parametrize(
    "value, src, dst, substance, expected",
    [
        (100, "mg/dL", "mmol/L", "glucose", 5.55),
        (1.0, "mg/dL", "μmol/L", "creatinine", 88.42),
        (200, "mg/dL", "mmol/L", "cholesterol", 5.18),
        (100, "lb", "kg", None, 45.36),
        (100, "kg", "lb", None, 220.46),
        (10, "in", "cm", None, 25.4),
        (100, "cm", "in", None, 39.37),
        (212, "°F", "°C", None, 100.0),
        (37, "°C", "°F", None, 98.6),
    ],
)
def test_spec_conversion_table(value, src, dst, substance, expected):
    assert unit_convert(value, src, dst, substance) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("variant", ["μmol/L", "µmol/L", "umol/L", "UMOL/L", " µmol/l "])
def test_micro_sign_variants_are_equivalent(variant):
    assert normalize_unit(variant) == "umol/L"
    assert unit_convert(1.2, "mg/dL", variant, "creatinine") == 106.1


@pytest.mark.parametrize("variant", ["lbs", "LB", "pounds"])
def test_weight_unit_aliases(variant):
    assert unit_convert(234.6, variant, "kg") == 106.41


def test_substance_is_ignored_for_body_measurements():
    assert unit_convert(70, "in", "cm", "height") == unit_convert(70, "in", "cm", None)


@pytest.mark.parametrize(
    "args",
    [
        (100, "mg/dL", "mmol/L", None),  # lab conversions need a substance
        (100, "mg/dL", "mmol/L", "sodium"),
        (100, "kg", "cm", None),
        (100, "furlong", "cm", None),
        ("abc", "lb", "kg", None),
    ],
)
def test_unsupported_conversions_return_error_string(args):
    out = unit_convert(*args)
    assert isinstance(out, str) and out.startswith("Error")


def test_calculate_bmi_formula_and_rounding():
    assert calculate_bmi(104.2, 163.4) == 39.0
    assert calculate_bmi(93.7, 172.6) == 31.5
    assert isinstance(calculate_bmi(0, 170), str)
    assert isinstance(calculate_bmi(70, -1), str)


def test_execute_tool_validates_calls():
    assert execute_tool("calculate_bmi", {"weight_kg": 70, "height_cm": 175}) == 22.9
    assert execute_tool("nope", {}).startswith("Error")
    assert execute_tool("calculate_bmi", {"weight_kg": 70}).startswith("Error")
    assert execute_tool("calculate_bmi", {"weight_kg": 70, "height_cm": 175, "x": 1}).startswith("Error")


# Every gold tool call in the dataset reproduces exactly; any future mismatch must be listed here explicitly.
KNOWN_GOLD_MISMATCHES: set[str] = set()


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_gold_tool_results_reproduce(split):
    mismatches = set()
    for r in load_split(split):
        for tc in r.get("tool_calls", []):
            got = execute_tool(tc["tool"], tc["arguments"])
            if isinstance(got, str) or abs(got - tc["result"]) > 1e-9:
                mismatches.add(r["id"])
    assert mismatches == {i for i in KNOWN_GOLD_MISMATCHES if i.startswith(split)}


def test_micromol_spelling_is_an_alias():
    from clinqa.tools import normalize_unit

    assert normalize_unit("micromol/L") == normalize_unit("μmol/L") == "umol/L"
