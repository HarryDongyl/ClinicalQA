"""Heuristic extractors over clinical notes and tables.

These are used for data-quality analysis (and later for synthesising tool-call
traces). They are regex-based and intentionally conservative; every extractor
used by a quality check is spot-checked and its limitations are documented in
docs/FINDINGS.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

NUM = r"(\d+(?:\.\d+)?)"

# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"(?<![\w.])[-\u2212]?\d+(?:,\d{3})*(?:\.\d+)?")


def parse_float(s: Any) -> float | None:
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return float(s)
    if not isinstance(s, str):
        return None
    try:
        return float(s.strip().replace("\u2212", "-").replace(",", ""))
    except ValueError:
        return None


def extract_numbers(text: str) -> list[float]:
    """All decimal numbers in text (thousands separators and unicode minus handled)."""
    out = []
    for m in _NUMBER_RE.finditer(text):
        v = parse_float(m.group(0))
        if v is not None:
            out.append(v)
    return out


def parse_bp(s: str) -> tuple[float, float] | None:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*", s or "")
    return (float(m.group(1)), float(m.group(2))) if m else None


_RANGE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[-\u2013\u2014]\s*(\d+(?:\.\d+)?)\s*$")
_BOUND_RE = re.compile(r"^\s*(<=|>=|<|>|\u2264|\u2265)\s*(\d+(?:\.\d+)?)\s*$")


def parse_ref_range(s: str) -> dict[str, float | None] | None:
    """Parse '70-100', '<200', '>40', '≤5' into {'low', 'high'}; None if unparseable."""
    if not isinstance(s, str):
        return None
    m = _RANGE_RE.match(s)
    if m:
        return {"low": float(m.group(1)), "high": float(m.group(2))}
    m = _BOUND_RE.match(s)
    if m:
        op, v = m.group(1), float(m.group(2))
        if op in ("<", "<=", "\u2264"):
            return {"low": None, "high": v}
        return {"low": v, "high": None}
    return None


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

PANEL_SIGNATURES: dict[str, frozenset[str]] = {
    "metabolic": frozenset({"Glucose (fasting)", "HbA1c", "Creatinine", "BUN", "Sodium", "Potassium"}),
    "lipid": frozenset({"Total Cholesterol", "LDL", "HDL", "Triglycerides"}),
    "cbc": frozenset({"WBC", "Hemoglobin", "Hematocrit", "Platelets"}),
    "liver": frozenset({"ALT", "AST", "Alkaline Phosphatase", "Total Bilirubin", "Albumin", "Total Protein"}),
    "thyroid": frozenset({"TSH", "Free T4", "Free T3"}),
    "coagulation": frozenset({"PT", "INR", "aPTT", "D-dimer"}),
    "iron": frozenset({"Serum Iron", "TIBC", "Ferritin", "Transferrin Saturation"}),
    "renal": frozenset({"Creatinine", "BUN", "eGFR", "Uric Acid", "Calcium", "Phosphorus"}),
}
VITALS_ROWS = ("Blood Pressure", "Heart Rate", "Temperature", "Respiratory Rate", "SpO2")
EXPECTED_HEADERS = {
    "labs": ["Test", "Value", "Unit", "Reference Range"],
    "vitals": ["Vital", "Value", "Unit"],
}


def table_panel(table: dict[str, Any]) -> str:
    """Classify a table as 'vitals', one of the lab panels, 'vitals+body' or 'mixed'."""
    names = frozenset(row[0] for row in table.get("rows", []) if row)
    if table.get("type") == "vitals":
        return "vitals" if names == frozenset(VITALS_ROWS) else "vitals+body"
    for panel, sig in PANEL_SIGNATURES.items():
        if names == sig:
            return panel
    return "mixed"


# ---------------------------------------------------------------------------
# Body measurements (weight / height)
# ---------------------------------------------------------------------------

_BODY_UNIT = {
    "kg": ("weight", "kg"), "kgs": ("weight", "kg"), "kilograms": ("weight", "kg"),
    "lb": ("weight", "lb"), "lbs": ("weight", "lb"), "pounds": ("weight", "lb"),
    "cm": ("height", "cm"), "in": ("height", "in"), "inch": ("height", "in"), "inches": ("height", "in"),
}
_BODY_MENTION_RE = re.compile(NUM + r"\s*(kgs?|kilograms|lbs?|pounds|cm|inches|inch|in)\b(?!\s*/\s*m)", re.I)
PLAUSIBLE_RANGE = {"kg": (20.0, 300.0), "lb": (44.0, 660.0), "cm": (100.0, 230.0), "in": (39.0, 91.0)}
_DELTA_BEFORE_RE = re.compile(
    r"\b(loss|lost|losing|lose|gain|gained|gaining|decrease[ds]?|increase[ds]?|dropped|down|up)\b[^.\n]{0,30}$", re.I
)
_DELTA_AFTER_RE = re.compile(
    r"^\s*(?:\([^)]{0,20}\)\s*)?(?:of\s+)?(?:\w+\s+){0,2}(?:weight\s+)?(?:loss|gain)\b"
    r"|^\s*(?:\([^)]{0,20}\)\s*)?(?:over|in)\s+(?:the\s+)?(?:past|last)\b",
    re.I,
)
_WEIGHT_LABEL_RE = re.compile(r"\b(weight|wt)\b", re.I)
_HEIGHT_LABEL_RE = re.compile(r"\b(height|ht)\b", re.I)


@dataclass(frozen=True)
class Measurement:
    kind: str  # "weight" | "height"
    value: float
    unit: str  # canonical: kg | lb | cm | in
    source: str  # note | question | table
    start: int
    end: int
    labeled: bool  # an explicit "weight"/"height" label precedes it
    is_delta: bool  # a change over time (e.g. "8 lbs of weight gain"), not a body measurement
    equivalent: bool  # parenthetical restatement of the preceding measurement, e.g. "(119.3 kg)"

    @property
    def metric_value(self) -> float:
        return to_metric(self.value, self.unit)


def to_metric(value: float, unit: str) -> float:
    return {"kg": value, "lb": value * 0.4536, "cm": value, "in": value * 2.54}[unit]


def extract_body_measurements(text: str, source: str = "note") -> list[Measurement]:
    """Find weight/height mentions; implausible values (ulcer size, JVP in cm, '5/5 in') are dropped."""
    out: list[Measurement] = []
    prev: Measurement | None = None
    for m in _BODY_MENTION_RE.finditer(text):
        value = float(m.group(1))
        unit_raw = m.group(2).lower()
        kind, unit = _BODY_UNIT[unit_raw]
        lo, hi = PLAUSIBLE_RANGE[unit]
        if not lo <= value <= hi:
            continue
        # Preceded by another digit-slash (e.g. "5/5 in") or part of a dimension ("3 x 2 cm").
        pre_char = text[max(0, m.start() - 1): m.start()]
        if pre_char in ("/", "."):
            continue
        # Thresholds such as "weight <=60 kg" in dosing criteria are not measurements.
        if text[max(0, m.start() - 3): m.start()].strip()[-1:] in ("<", ">", "\u2264", "\u2265", "="):
            continue
        after = text[m.end(): m.end() + 40]
        if unit_raw == "in" and re.match(r"\s+[a-z]", after):
            continue
        before = text[max(0, m.start() - 40): m.start()]
        line_before = before.split("\n")[-1]
        is_delta = bool(_DELTA_BEFORE_RE.search(line_before) or _DELTA_AFTER_RE.match(after))
        label_re = _WEIGHT_LABEL_RE if kind == "weight" else _HEIGHT_LABEL_RE
        labeled = bool(label_re.search(line_before[-30:]))
        equivalent = bool(
            prev is not None
            and prev.kind == kind
            and re.fullmatch(r"\s*\(\s*(?:~|approx\.?|approximately)?\s*", text[prev.end: m.start()])
        )
        meas = Measurement(kind, value, unit, source, m.start(), m.end(), labeled, is_delta, equivalent)
        out.append(meas)
        prev = meas
    return out


def table_body_measurements(table: dict[str, Any]) -> list[Measurement]:
    out = []
    for i, row in enumerate(table.get("rows", [])):
        if len(row) < 3 or row[0] not in ("Weight", "Height"):
            continue
        v = parse_float(row[1])
        unit_info = _BODY_UNIT.get(row[2].strip().lower())
        if v is None or unit_info is None:
            continue
        kind, unit = unit_info
        out.append(Measurement(kind, v, unit, "table", i, i, True, False, False))
    return out


def record_body_measurements(record: dict[str, Any]) -> list[Measurement]:
    """Body measurements from note, table and question (non-delta only)."""
    ms = (
        extract_body_measurements(record["note"], "note")
        + table_body_measurements(record["table"])
        + extract_body_measurements(record["question"], "question")
    )
    return [m for m in ms if not m.is_delta]


# ---------------------------------------------------------------------------
# Inline BMI, demographics
# ---------------------------------------------------------------------------

_BMI_RE = re.compile(r"\bBMI\b[^0-9\n]{0,12}?(\d{2}(?:\.\d+)?)(?!\s*[-\u2013]\s*\d)", re.I)


def extract_inline_bmi(text: str) -> list[float]:
    return [float(m.group(1)) for m in _BMI_RE.finditer(text)]


_AGE_RE = re.compile(r"\b(\d{1,3})[- ](?:year|yr)s?[- ]old\b", re.I)
_SEX_RE = re.compile(r"\b(male|female|man|woman|gentleman|lady)\b", re.I)
_SEX_MAP = {"male": "M", "man": "M", "gentleman": "M", "female": "F", "woman": "F", "lady": "F"}


def extract_age_sex(note: str) -> tuple[int | None, str | None]:
    age_m = _AGE_RE.search(note)
    age = int(age_m.group(1)) if age_m else None
    sex = None
    if age_m:
        m = _SEX_RE.search(note, age_m.end(), age_m.end() + 40)
        if m:
            sex = _SEX_MAP[m.group(1).lower()]
    if sex is None:
        m = _SEX_RE.search(note)
        if m:
            sex = _SEX_MAP[m.group(1).lower()]
    if sex is None:
        if re.search(r"\bMr\.", note):
            sex = "M"
        elif re.search(r"\bM(?:s|rs)\.", note):
            sex = "F"
    return age, sex


# ---------------------------------------------------------------------------
# Sections, allergies, medications
# ---------------------------------------------------------------------------

SECTION_PATTERNS: dict[str, str] = {
    "cc": r"CC|Chief Complaint",
    "hpi": r"HPI|History of Present(?:ing)? Illness",
    "pmh": r"PMH|Past Medical History",
    "medications": r"(?:Current |Home )?Medications?",
    "allergies": r"Allerg(?:y|ies)",
    "vitals": r"Vitals?|Vital Signs",
    "exam": r"(?:Physical )?Exam(?:ination)?",
    "labs": r"Labs?|Laboratory(?: Results| Data| Studies| Findings)?",
    "assessment_plan": r"Assessment(?:\s*(?:/|and|&)\s*Plan)?|A/P|Plan|Impression",
}
_SECTION_RES = {
    k: re.compile(r"(?<![A-Za-z])\**\s*(?:" + v + r")\s*\**\s*:", re.I) for k, v in SECTION_PATTERNS.items()
}


def section_presence(note: str) -> dict[str, bool]:
    return {k: bool(r.search(note)) for k, r in _SECTION_RES.items()}


def note_style(note: str) -> str:
    return "markdown" if "**" in note else "plain"


_ALLERGY_RE = re.compile(r"\ballerg(?:y|ies)(?:\s+history)?\s*\**\s*:\s*\**\s*([^\n]*)", re.I)
_NKDA_RE = re.compile(r"\bNKDA\b|no known (?:drug )?allergies|\bnone known\b|\bno allergies\b", re.I)
_NOT_DOC_RE = re.compile(
    r"not (?:documented|recorded|obtained|available|listed|reviewed|assessed|specified|provided)"
    r"|\bunknown\b|\bunable\b|\bpending\b|\bunclear\b|to be (?:obtained|confirmed|clarified)",
    re.I,
)


def allergy_status(note: str) -> str:
    """One of: nkda, not_documented, documented, absent (no allergy statement at all)."""
    m = _ALLERGY_RE.search(note)
    if m:
        content = m.group(1)
        if _NOT_DOC_RE.search(content):
            return "not_documented"
        if _NKDA_RE.search(content):
            return "nkda"
        return "documented" if content.strip(" .*") else "not_documented"
    if _NKDA_RE.search(note):
        return "nkda"
    return "absent"


_MED_HEADER_RE = re.compile(r"(?<![A-Za-z])\**\s*(?:Current |Home )?Medications?(?:\s+List)?\s*\**\s*:\s*\**", re.I)
_NEXT_SECTION_RE = re.compile(
    r"\n\s*\n|\n\s*[*#-]*\s*\**\s*(?:Allerg|Vital|Physical|Exam|PMH|Past|Social|Family|Review|ROS|Lab|"
    r"Assessment|Plan|HPI|CC|Imaging|Objective|General)\w*[^:\n]{0,25}:"
    r"|(?<=[.)])\s+\**\s*(?:Allerg\w*|Vitals?|Exam|Physical Exam)\s*\**\s*:",
    re.I,
)
_DOSE_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:mg|mcg|\u00b5g|\u03bcg|ug|g|units?|u|ml|%|puffs?|meq)\b", re.I)


@dataclass(frozen=True)
class Medication:
    raw: str
    name: str
    has_dose: bool


def medication_section(note: str) -> str | None:
    m = _MED_HEADER_RE.search(note)
    if not m:
        return None
    rest = note[m.end():]
    end = _NEXT_SECTION_RE.search(rest)
    return rest[: end.start()] if end else rest


_NON_DRUG_WORDS = {
    "none", "no", "denies", "daily", "twice", "once", "bid", "tid", "qid", "prn", "as", "at", "every",
    "with", "po", "qhs", "qd", "and", "per", "dose", "taken", "the", "unknown", "recently",
    "though", "or", "which", "however", "details", "patient", "coagulation", "antihypertensive",
    "antiplatelet", "anxiolytic", "statin", "cpap", "previously", "currently", "not",
}


def extract_medications(note: str) -> list[Medication]:
    section = medication_section(note)
    if section is None:
        return []
    items: list[str] = []
    for ln in section.split("\n"):
        ln = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", ln).strip()
        if not ln:
            continue
        # Split comma/semicolon lists but keep parenthesised remarks intact.
        items.extend(p.strip(" .") for p in re.split(r"[;,](?![^()]*\))", ln) if p.strip(" ."))
    meds = []
    for it in items:
        name_m = re.match(
            r"([A-Za-z][A-Za-z\-]*(?:\s+(?:glargine|lispro|aspart|detemir|carbonate|sulfate|succinate|tartrate|inhaler))?)",
            it,
        )
        if not name_m:
            continue
        name = name_m.group(1).lower()
        if name.split()[0] in _NON_DRUG_WORDS:
            continue
        meds.append(Medication(raw=it, name=name, has_dose=bool(_DOSE_RE.search(it))))
    return meds


# ---------------------------------------------------------------------------
# Analyte mentions in note text (for note-vs-table consistency)
# ---------------------------------------------------------------------------

NOTE_ANALYTE_ALIASES: dict[str, list[str]] = {
    "Glucose (fasting)": ["fasting glucose", "glucose"],
    "HbA1c": ["HbA1c", "hemoglobin A1c", "A1c"],
    "Creatinine": ["creatinine"],
    "BUN": ["BUN"],
    "Sodium": ["sodium"],
    "Potassium": ["potassium"],
    "eGFR": ["eGFR"],
    "Uric Acid": ["uric acid"],
    "Calcium": ["calcium"],
    "Phosphorus": ["phosphorus"],
    "TSH": ["TSH"],
    "Free T4": ["free T4"],
    "Free T3": ["free T3"],
    "Serum Iron": ["serum iron"],
    "TIBC": ["TIBC"],
    "Ferritin": ["ferritin"],
    "Transferrin Saturation": ["transferrin saturation"],
    "PT": ["PT"],
    "INR": ["INR"],
    "aPTT": ["aPTT"],
    "D-dimer": ["D-dimer"],
    "ALT": ["ALT"],
    "AST": ["AST"],
    "Alkaline Phosphatase": ["alkaline phosphatase", "ALP"],
    "Total Bilirubin": ["total bilirubin"],
    "Albumin": ["albumin"],
    "Total Protein": ["total protein"],
    "WBC": ["WBC"],
    "Hemoglobin": ["hemoglobin", "Hgb"],
    "Hematocrit": ["hematocrit", "Hct"],
    "Platelets": ["platelets", "platelet count"],
    "Total Cholesterol": ["total cholesterol"],
    "LDL": ["LDL"],
    "HDL": ["HDL"],
    "Triglycerides": ["triglycerides"],
    "Heart Rate": ["HR", "heart rate"],
    "Temperature": ["Temp", "temperature"],
    "Respiratory Rate": ["RR", "respiratory rate"],
    "SpO2": ["SpO2"],
    "Blood Pressure": ["BP", "blood pressure"],
}
_FILLER = r"(?:\s*(?:level|value|count|of|was|is|at|elevated|reduced|low|high|markedly|mildly|significantly|approximately|now|currently|remains|:|=)\s*)*"


def _analyte_re(alias: str) -> re.Pattern[str]:
    value = r"(\d+(?:\.\d+)?/\d+(?:\.\d+)?)" if alias in ("BP", "blood pressure") else NUM
    flags = re.I if alias.islower() or " " in alias else 0
    # Alias boundaries stop 'PT' matching inside 'aPTT'; the tail rejects '120/80'-style values for non-BP analytes.
    return re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])" + _FILLER + r"\s*" + value + r"(?!\d)(?!\.\d)(?!/\d)",
        flags,
    )


_ANALYTE_RES = {name: [(a, _analyte_re(a)) for a in aliases] for name, aliases in NOTE_ANALYTE_ALIASES.items()}
# "calcium 1.3 mg/dL above upper limit" states a difference, not the analyte value.
_DELTA_TAIL_RE = re.compile(r"^\s*(?:[^\s,;]+\s+)?(?:above|below|higher|lower|over|under|greater|less|more|beyond)\b", re.I)


def note_analyte_values(note: str, analyte: str) -> list[str]:
    """Values written in the note text for a table analyte, e.g. 'creatinine 3.6 mg/dL' -> ['3.6'].

    Skips difference statements ('1.3 mg/dL above the upper limit') and combined labels such as 'PT/INR'.
    """
    vals: list[str] = []
    for alias, r in _ANALYTE_RES.get(analyte, []):
        for m in r.finditer(note):
            before = note[m.start() - 1: m.start()] if m.start() else ""
            after_alias = note[m.start() + len(alias): m.start() + len(alias) + 1]
            if before == "/" or after_alias == "/":
                continue
            if _DELTA_TAIL_RE.match(note[m.end(): m.end() + 40]):
                continue
            vals.append(m.group(1))
    return vals
