import pytest

from clinqa.schemas import TOOL_SCHEMAS, parse_assistant_output, tool_response_content, validate_call


def test_schemas_cover_exactly_two_tools():
    names = [s["function"]["name"] for s in TOOL_SCHEMAS]
    assert names == ["unit_convert", "calculate_bmi"]
    uc = TOOL_SCHEMAS[0]["function"]["parameters"]
    assert uc["required"] == ["value", "from_unit", "to_unit", "substance"]
    assert uc["properties"]["substance"]["type"] == ["string", "null"]


@pytest.mark.parametrize("name,args", [
    ("calculate_bmi", {"weight_kg": 104.2, "height_cm": 163.4}),
    ("calculate_bmi", {"weight_kg": 80, "height_cm": 170}),
    ("unit_convert", {"value": 0.7, "from_unit": "mg/dL", "to_unit": "μmol/L", "substance": "creatinine"}),
    ("unit_convert", {"value": 150, "from_unit": "lb", "to_unit": "kg", "substance": None}),
])
def test_valid_calls(name, args):
    assert validate_call(name, args) == []


@pytest.mark.parametrize("name,args,fragment", [
    ("calculate_bmi", {"weight_kg": True, "height_cm": 170}, "weight_kg"),
    ("calculate_bmi", {"weight_kg": "80", "height_cm": 170}, "weight_kg"),
    ("calculate_bmi", {"weight_kg": float("nan"), "height_cm": 170}, "finite"),
    ("calculate_bmi", {"weight_kg": 80}, "missing"),
    ("calculate_bmi", {"weight_kg": 80, "height_cm": 170, "age": 3}, "unexpected"),
    ("unit_convert", {"value": 1.0, "from_unit": "mg/dL", "to_unit": "mmol/L"}, "missing"),
    ("unit_convert", {"value": 1.0, "from_unit": 5, "to_unit": "mmol/L", "substance": None}, "from_unit"),
    ("bmi", {"weight_kg": 80, "height_cm": 170}, "unknown tool"),
])
def test_invalid_calls(name, args, fragment):
    errors = validate_call(name, args)
    assert errors and any(fragment in e for e in errors)


def test_parse_plain_answer():
    out = parse_assistant_output("The heart rate is 117 bpm.")
    assert out.calls == [] and out.content == "The heart rate is 117 bpm." and out.status == "no_call"


def test_parse_single_call():
    text = '<tool_call>\n{"name": "calculate_bmi", "arguments": {"weight_kg": 106.4, "height_cm": 189.2}}\n</tool_call>'
    out = parse_assistant_output(text)
    assert out.status == "valid"
    assert out.calls[0].name == "calculate_bmi"
    assert out.calls[0].arguments == {"weight_kg": 106.4, "height_cm": 189.2}
    assert out.content == ""


@pytest.mark.parametrize("text,status", [
    ('<tool_call>\n{"name": "calculate_bmi", "arguments": {"weight_kg": 106.4,}}\n</tool_call>', "invalid_json"),
    ('<tool_call>\n{"name": "calculate_bmi", "arguments": {"weight_kg": 106.4, "height_cm": 189.2}}', "unterminated"),
    ('<tool_call>\n{"name": "calculate_bmi", "arguments": {"weight_kg": "106", "height_cm": 189.2}}\n</tool_call>',
     "schema_error"),
    ('<tool_call>\n["calculate_bmi"]\n</tool_call>', "invalid_json"),
    ('<tool_call>\n{"name": "calculate_bmi", "arguments": {"weight_kg": 1, "height_cm": 2}, "x": 1}\n</tool_call>',
     "invalid_json"),
])
def test_parse_rejects_bad_calls(text, status):
    out = parse_assistant_output(text)
    assert out.status == status


def test_parse_rejects_duplicate_keys_and_nan():
    dup = '<tool_call>\n{"name": "calculate_bmi", "arguments": {"weight_kg": 1, "weight_kg": 2, "height_cm": 3}}\n</tool_call>'
    assert parse_assistant_output(dup).status == "invalid_json"
    nan = '<tool_call>\n{"name": "calculate_bmi", "arguments": {"weight_kg": NaN, "height_cm": 3}}\n</tool_call>'
    assert parse_assistant_output(nan).status == "invalid_json"


def test_tool_response_content_shape():
    assert tool_response_content(39.0) == '{"result": 39.0}'
    assert tool_response_content("Error: unsupported conversion") == '{"error": "Error: unsupported conversion"}'
