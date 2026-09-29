"""Structural checks before feature extraction; semantic flags remain reviewable."""

from __future__ import annotations

import math
from typing import Any

ANSWER_TYPES = {"extractive", "numeric_reasoning", "tool_call", "uncertain"}
HEADERS = {
    "labs": ["Test", "Value", "Unit", "Reference Range"],
    "vitals": ["Vital", "Value", "Unit"],
}


def _number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def validate_records(records: list[dict[str, Any]], split: str) -> None:
    """Raise ValueError with a record ID for unsafe/malformed dataset structure."""
    seen: set[str] = set()
    for index, record in enumerate(records):
        label = f"{split}[{index}] ({record.get('id', record.get('key', '?'))})"

        def require(condition: bool, message: str) -> None:
            if not condition:
                raise ValueError(f"{label}: {message}")

        if split == "reference":
            for key in ("key", "category", "value"):
                require(isinstance(record.get(key), str) and bool(record[key].strip()), f"invalid {key}")
            require(record["key"] not in seen, "duplicate reference key")
            seen.add(record["key"])
            continue

        for key in ("id", "note", "question", "answer", "answer_type"):
            require(isinstance(record.get(key), str) and bool(record[key].strip()), f"invalid {key}")
        require(record["id"].startswith(split + "_"), "ID does not match split")
        require(record["id"] not in seen, "duplicate ID")
        seen.add(record["id"])
        require(record["answer_type"] in ANSWER_TYPES, "unknown answer_type")
        table = record.get("table")
        require(isinstance(table, dict), "table must be an object")
        require(isinstance(table.get("type"), str) and table["type"] in HEADERS, "unknown table type")
        require(table.get("headers") == HEADERS[table["type"]], "invalid table headers")
        rows = table.get("rows")
        require(isinstance(rows, list) and bool(rows), "table rows must be a nonempty array")
        for row in rows:
            require(isinstance(row, list) and len(row) == len(table["headers"]), "invalid table row width")
            require(all(isinstance(cell, str) for cell in row), "table cells must be strings")
        has_calls = "tool_calls" in record
        require(has_calls == (record["answer_type"] == "tool_call"), "tool_calls inconsistent with answer_type")
        if not has_calls:
            continue
        calls = record["tool_calls"]
        require(isinstance(calls, list) and bool(calls), "tool_calls must be a nonempty array")
        for call in calls:
            require(isinstance(call, dict), "tool call must be an object")
            name = call.get("tool")
            require(isinstance(name, str) and name in {"calculate_bmi", "unit_convert"}, "unknown Core tool")
            args = call.get("arguments")
            require(isinstance(args, dict), "arguments must be an object")
            required = {"weight_kg", "height_cm"} if name == "calculate_bmi" else {"value", "from_unit", "to_unit"}
            allowed = required | ({"substance"} if name == "unit_convert" else set())
            require(required <= args.keys() <= allowed, "missing or unexpected tool arguments")
            for key in ({"weight_kg", "height_cm"} if name == "calculate_bmi" else {"value"}):
                require(_number(args[key]), f"{key} must be a finite JSON number")
                if name == "calculate_bmi":
                    require(args[key] > 0, f"{key} must be positive")
            if name == "unit_convert":
                for key in ("from_unit", "to_unit"):
                    require(isinstance(args[key], str) and bool(args[key].strip()), f"invalid {key}")
                require(args.get("substance") is None or isinstance(args["substance"], str), "invalid substance")
            require(_number(call.get("result")), "result must be a finite JSON number")
