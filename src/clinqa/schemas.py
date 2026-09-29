"""Tool schemas and the assistant wire format (D-024, D-026).

Single place that knows how a tool call looks on the wire. The model emits Qwen's
native form, one call per block:

    <tool_call>
    {"name": "calculate_bmi", "arguments": {"weight_kg": 104.2, "height_cm": 163.4}}
    </tool_call>

Parsing is strict: malformed JSON, duplicate keys, NaN/Infinity, bool-as-number and
numeric strings are rejected and never repaired. Unit alias normalization belongs to
the executor and the scorer, not to parsing.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

UNIT_CONVERT_DESCRIPTION = (
    "Convert a numeric value between units. Supported conversions: mg/dL -> mmol/L (substance "
    "'glucose' or 'cholesterol'), mg/dL -> umol/L (substance 'creatinine'), lb <-> kg, in <-> cm, "
    "F <-> C (substance null for body measurements and temperature). Returns an error for "
    "unsupported conversions."
)

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "unit_convert",
            "description": UNIT_CONVERT_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Value to convert."},
                    "from_unit": {"type": "string", "description": "Unit of the input value, e.g. 'mg/dL', 'lb'."},
                    "to_unit": {"type": "string", "description": "Target unit, e.g. 'mmol/L', 'kg'."},
                    "substance": {
                        "type": ["string", "null"],
                        "description": "Analyte for lab conversions ('glucose', 'creatinine', 'cholesterol'); "
                                       "null otherwise.",
                    },
                },
                "required": ["value", "from_unit", "to_unit", "substance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_bmi",
            "description": "Body mass index = weight_kg / (height_cm / 100)^2, rounded to 1 decimal. "
                           "Arguments must be metric.",
            "parameters": {
                "type": "object",
                "properties": {
                    "weight_kg": {"type": "number", "description": "Body weight in kilograms."},
                    "height_cm": {"type": "number", "description": "Height in centimetres."},
                },
                "required": ["weight_kg", "height_cm"],
            },
        },
    },
]

def tool_schemas_sha256() -> str:
    """Hash of the schemas the chat template renders into every prompt (provenance)."""
    import hashlib

    return hashlib.sha256(json.dumps(TOOL_SCHEMAS, sort_keys=True).encode("utf-8")).hexdigest()


_SCHEMA_BY_NAME = {s["function"]["name"]: s["function"]["parameters"] for s in TOOL_SCHEMAS}

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
_BLOCK = re.compile(re.escape(TOOL_CALL_OPEN) + r"(.*?)" + re.escape(TOOL_CALL_CLOSE), re.S)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _type_ok(v: Any, spec: str | list[str]) -> bool:
    kinds = spec if isinstance(spec, list) else [spec]
    for kind in kinds:
        if kind == "number" and _is_number(v):
            return True
        if kind == "string" and isinstance(v, str):
            return True
        if kind == "null" and v is None:
            return True
    return False


def validate_call(name: Any, arguments: Any) -> list[str]:
    """Strict JSON-schema check. Returns a list of error strings; empty means valid."""
    if name not in _SCHEMA_BY_NAME:
        return [f"unknown tool {name!r}"]
    if not isinstance(arguments, dict):
        return ["arguments must be an object"]
    schema = _SCHEMA_BY_NAME[name]
    errors = [f"missing required argument {k!r}" for k in schema["required"] if k not in arguments]
    errors += [f"unexpected argument {k!r}" for k in sorted(set(arguments) - set(schema["properties"]))]
    for key, spec in schema["properties"].items():
        if key not in arguments:
            continue
        v = arguments[key]
        if not _type_ok(v, spec["type"]):
            errors.append(f"{key}: expected {spec['type']}, got {type(v).__name__}")
        elif _is_number(v) and not math.isfinite(v):
            errors.append(f"{key}: must be finite")
    return errors


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ParsedOutput:
    """Result of parsing one assistant turn.

    status: no_call | valid | invalid_json | unterminated | schema_error
    """

    status: str
    content: str
    calls: list[ToolCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k!r}")
        out[k] = v
    return out


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite constant {token}")


def _loads_strict(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_strict_object_pairs, parse_constant=_reject_constant)


def parse_assistant_output(text: str) -> ParsedOutput:
    """Parse raw assistant text (special tokens already stripped) into content + tool calls."""
    blocks = list(_BLOCK.finditer(text))
    content = _BLOCK.sub("", text).strip()
    if not blocks:
        if TOOL_CALL_OPEN in text:
            return ParsedOutput("unterminated", content=text.split(TOOL_CALL_OPEN)[0].strip(),
                                errors=["<tool_call> without closing tag"])
        return ParsedOutput("no_call", content=content)
    if TOOL_CALL_OPEN in content:
        return ParsedOutput("unterminated", content=content, errors=["<tool_call> without closing tag"])
    calls: list[ToolCall] = []
    errors: list[str] = []
    for block in blocks:
        try:
            obj = _loads_strict(block.group(1).strip())
        except ValueError as e:
            return ParsedOutput("invalid_json", content=content, errors=[str(e)])
        if not isinstance(obj, dict) or set(obj) != {"name", "arguments"}:
            return ParsedOutput("invalid_json", content=content,
                                errors=["call must be an object with exactly 'name' and 'arguments'"])
        errors += validate_call(obj["name"], obj["arguments"])
        calls.append(ToolCall(obj["name"], obj["arguments"] if isinstance(obj["arguments"], dict) else {}))
    return ParsedOutput("schema_error" if errors else "valid", content=content, calls=calls, errors=errors)


def tool_response_content(result: float | str) -> str:
    """Tool message content: {"result": x} or {"error": "..."} (D-024)."""
    if isinstance(result, str):
        return json.dumps({"error": result}, ensure_ascii=False)
    return json.dumps({"result": result})
