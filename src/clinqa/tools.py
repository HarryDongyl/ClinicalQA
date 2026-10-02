"""Deterministic tools the model is trained to call.

Signatures and formulas follow docs/ASSIGNMENT.md exactly. calculate_egfr is the Stretch A third tool
(docs/STRETCH_A_PLAN.md): it is registered here but only offered to the model by configs that list it. Unit strings are
normalised (e.g. Greek mu, micro sign and "u" are equivalent; case-insensitive)
so that harmless spelling variants do not cause spurious tool errors.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

# P-001: explicit 2 dp data-compatibility policy; BMI separately uses 1 dp.
UNIT_CONVERT_DECIMALS = 2
BMI_DECIMALS = 1

_UNIT_ALIASES = {
    "mg/dl": "mg/dL",
    "mmol/l": "mmol/L",
    "umol/l": "umol/L",
    "micromol/l": "umol/L",
    "kg": "kg",
    "kgs": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "lb": "lb",
    "lbs": "lb",
    "pound": "lb",
    "pounds": "lb",
    "in": "in",
    "inch": "in",
    "inches": "in",
    '"': "in",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "°f": "F",
    "ºf": "F",
    "degf": "F",
    "f": "F",
    "fahrenheit": "F",
    "°c": "C",
    "ºc": "C",
    "degc": "C",
    "c": "C",
    "celsius": "C",
}

_SUBSTANCE_ALIASES = {
    "glucose": "glucose",
    "fasting glucose": "glucose",
    "glucose (fasting)": "glucose",
    "blood glucose": "glucose",
    "creatinine": "creatinine",
    "serum creatinine": "creatinine",
    "cholesterol": "cholesterol",
    "total cholesterol": "cholesterol",
}

# (from_unit, to_unit, substance) -> formula; substance None means substance-independent.
_CONVERSIONS: dict[tuple[str, str, str | None], Callable[[float], float]] = {
    ("mg/dL", "mmol/L", "glucose"): lambda v: v * 0.0555,
    ("mg/dL", "umol/L", "creatinine"): lambda v: v * 88.42,
    ("mg/dL", "mmol/L", "cholesterol"): lambda v: v * 0.0259,
    ("lb", "kg", None): lambda v: v * 0.4536,
    ("kg", "lb", None): lambda v: v * 2.2046,
    ("in", "cm", None): lambda v: v * 2.54,
    ("cm", "in", None): lambda v: v * 0.3937,
    ("F", "C", None): lambda v: (v - 32) * 5 / 9,
    ("C", "F", None): lambda v: v * 9 / 5 + 32,
}


def normalize_unit(unit: str) -> str | None:
    """Map a unit string to its canonical form, or None if unknown."""
    if not isinstance(unit, str):
        return None
    key = unit.strip().replace("\u00b5", "u").replace("\u03bc", "u").replace(" ", "").lower()
    return _UNIT_ALIASES.get(key)


def normalize_substance(substance: str | None) -> str | None:
    if substance is None:
        return None
    if not isinstance(substance, str):
        return None
    key = " ".join(substance.strip().lower().split())
    if key in ("", "null", "none"):
        return None
    return _SUBSTANCE_ALIASES.get(key, key)


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def unit_convert(value: float, from_unit: str, to_unit: str, substance: str | None = None) -> float | str:
    """Convert `value` between units. Returns an error string for unsupported conversions."""
    v = _as_number(value)
    if v is None:
        return f"Error: value must be a finite number, got {value!r}"
    src, dst = normalize_unit(from_unit), normalize_unit(to_unit)
    sub = normalize_substance(substance)
    formula = _CONVERSIONS.get((src, dst, sub)) if src and dst else None
    if formula is None and src and dst:
        # Body-measurement/temperature conversions do not depend on substance.
        formula = _CONVERSIONS.get((src, dst, None))
    if formula is None:
        return f"Error: unsupported conversion from {from_unit!r} to {to_unit!r} (substance={substance!r})"
    return round(formula(v), UNIT_CONVERT_DECIMALS)


def calculate_bmi(weight_kg: float, height_cm: float) -> float | str:
    """BMI = weight_kg / (height_cm / 100)^2, rounded to 1 decimal."""
    w, h = _as_number(weight_kg), _as_number(height_cm)
    if w is None or h is None or w <= 0 or h <= 0:
        return f"Error: weight_kg and height_cm must be positive numbers, got {weight_kg!r}, {height_cm!r}"
    return round(w / (h / 100) ** 2, BMI_DECIMALS)


_SEX_ALIASES = {"male": "male", "m": "male", "man": "male", "female": "female", "f": "female", "woman": "female"}
EGFR_AGE_RANGE = (18, 120)


def calculate_egfr(creatinine_mg_dl: float, age: int, sex: str) -> int | str:
    """Race-free CKD-EPI 2021 creatinine equation, rounded half-up to an integer mL/min/1.73m2 (Stretch A)."""
    cr, years = _as_number(creatinine_mg_dl), _as_number(age)
    norm = _SEX_ALIASES.get(sex.strip().lower()) if isinstance(sex, str) else None
    if cr is None or cr <= 0:
        return f"Error: creatinine_mg_dl must be a positive number, got {creatinine_mg_dl!r}"
    if years is None or years != int(years) or not EGFR_AGE_RANGE[0] <= years <= EGFR_AGE_RANGE[1]:
        return f"Error: age must be an integer between {EGFR_AGE_RANGE[0]} and {EGFR_AGE_RANGE[1]}, got {age!r}"
    if norm is None:
        return f"Error: sex must be 'male' or 'female', got {sex!r}"
    female = norm == "female"
    kappa, alpha = (0.7, -0.241) if female else (0.9, -0.302)
    ratio = cr / kappa
    egfr = 142 * min(ratio, 1) ** alpha * max(ratio, 1) ** -1.200 * 0.9938 ** int(years) * (1.012 if female else 1.0)
    return int(Decimal(str(egfr)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


TOOL_REGISTRY: dict[str, Callable[..., float | str]] = {
    "unit_convert": unit_convert,
    "calculate_bmi": calculate_bmi,
    "calculate_egfr": calculate_egfr,
}

_REQUIRED_ARGS = {
    "unit_convert": ("value", "from_unit", "to_unit"),
    "calculate_bmi": ("weight_kg", "height_cm"),
    "calculate_egfr": ("creatinine_mg_dl", "age", "sex"),
}
_OPTIONAL_ARGS = {"unit_convert": ("substance",), "calculate_bmi": (), "calculate_egfr": ()}


def execute_tool(name: str, arguments: dict[str, Any]) -> float | str:
    """Dispatch a tool call; malformed calls return an error string instead of raising."""
    if name not in TOOL_REGISTRY:
        return f"Error: unknown tool {name!r}"
    if not isinstance(arguments, dict):
        return "Error: arguments must be a JSON object"
    missing = [a for a in _REQUIRED_ARGS[name] if a not in arguments]
    if missing:
        return f"Error: missing required argument(s) {missing} for {name}"
    allowed = set(_REQUIRED_ARGS[name]) | set(_OPTIONAL_ARGS[name])
    unexpected = sorted(set(arguments) - allowed)
    if unexpected:
        return f"Error: unexpected argument(s) {unexpected} for {name}"
    return TOOL_REGISTRY[name](**arguments)
