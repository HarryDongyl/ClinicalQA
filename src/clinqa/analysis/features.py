"""Per-record derived features shared by statistics and quality checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from clinqa import parsing as P
from clinqa.analysis.common import mask_numbers, sentence_count, word_count

_NEG = (
    r"(?:not (?:been )?(?:documented|recorded|specified|provided|available|listed|stated|included|mentioned|reported|"
    r"noted|confirmed|known|clear|obtained|measured|indicated)"
    r"|missing|unknown|absent|lacks?|lacking|without|unable|"
    r"no (?:documented|recorded|allergy|allergies|height|weight|dose|dosage|timestamp|date|time|information|mention)"
    r"|does not (?:document|include|specify|record|mention|list|state|provide|contain|indicate)"
    r"|cannot be (?:determined|confirmed|calculated|established|assessed|verified|computed))"
)
_NEG_RE = re.compile(_NEG, re.I)
_TERM_RES = {
    "allergy": re.compile(r"\ballerg\w*", re.I),
    "dose": re.compile(r"\b(?:dose|doses|dosage|dosing|strength)\b", re.I),
    "height": re.compile(r"\bheight\b", re.I),
    "weight": re.compile(r"\bweight\b(?![- ](?:loss|gain|based|status|category|classification|management|related|change))", re.I),
    "timing": re.compile(r"\b(?:timestamp|timing|time ?frame|time ?point|dated?|drawn|collected|collection|when)\b", re.I),
}
_Q_TIMING_RE = re.compile(
    r"\b(?:when|timestamp|timing|timeline|prior to|same day|how recent|trend\w*|compared? to prior|comparing to prior"
    r"|over time|at the time of|drawn|collected)\b",
    re.I,
)
_A_TIMING_RE = re.compile(r"\bno (?:\w+\s+){0,3}(?:date|time|timestamp)s?\b|\bwithout (?:a |the )?(?:\w+\s+){0,2}timestamp", re.I)
_Q_BMI_RE = re.compile(r"\bBMI\b|body mass index", re.I)


_POSITIVE_RE = re.compile(
    r"^\W{0,3}(?:is |was |has been )?(?:documented|recorded|listed|noted|measured)\s+(?:at|as)\b"
    r"|^\W{0,3}(?:is|was) (?:documented|recorded|available|provided)\b"
    r"|^\s*(?:of|is|was|as|:)\s*\d",
    re.I,
)
_GENERIC_RE = re.compile(r"\bboth\b.*\band\b|\brequire[sd]?\b|\bneeded to compute\b", re.I)
_NEITHER_RE = re.compile(r"\bneither\b[^.;]{0,30}\bnor\b|\bno (?:height|weight) or (?:height|weight)\b", re.I)


def _missing_terms(answer: str) -> set[str]:
    """Terms the answer says are missing: a negation within ~40 chars after, or ~25 chars before, the term.

    Formula parentheticals ('BMI = weight in kg / height in m2') and generic requirement statements
    ('BMI requires both height and weight') are ignored; terms stated as documented are removed.
    """
    text = re.sub(r"\([^)]*\)", "", answer)
    tags: set[str] = set()
    documented: set[str] = set()
    for clause in re.split(r"[.;]\s+|,\s+(?:but|however|while|whereas)\s+|;\s*however,?\s+", text):
        if _NEITHER_RE.search(clause):
            tags.update({"height", "weight"})
            continue
        generic = bool(_GENERIC_RE.search(clause))
        for tag, r in _TERM_RES.items():
            for m in r.finditer(clause):
                after = clause[m.end(): m.end() + 40]
                if _POSITIVE_RE.search(after):
                    documented.add(tag)
                    break
                if generic and tag in ("height", "weight"):
                    continue
                before = clause[max(0, m.start() - 25): m.start()]
                if _NEG_RE.search(after) or _NEG_RE.search(before):
                    tags.add(tag)
                    break
    return tags - documented


def classify_uncertain(question: str, answer: str) -> str:
    """What the gold answer says is missing: allergy | dose | height | weight | height_and_weight | timestamp | other."""
    tags = _missing_terms(answer)
    if "allergy" in tags:
        return "allergy"
    if "height" in tags and "weight" in tags:
        return "height_and_weight"
    if "height" in tags:
        return "height"
    if "weight" in tags:
        return "weight"
    if "dose" in tags:
        return "dose"
    if "timing" in tags or _A_TIMING_RE.search(answer) or _Q_TIMING_RE.search(question):
        return "timestamp"
    q = question.lower()
    if "allerg" in q:
        return "allergy"
    if re.search(r"\bdos(?:e|age|ing)\b", q):
        return "dose"
    return "other"


_NUMERIC_CATS = [
    ("ratio", re.compile(r"how many times|times (?:the|above|higher|greater)|\bfold\b", re.I)),
    ("percentage", re.compile(r"percent|%|percentage", re.I)),
    ("difference", re.compile(r"by how (?:much|many)|how (?:much|far) (?:higher|lower|above|below)|how far|difference|exceed", re.I)),
    ("ranking", re.compile(r"most (?:significantly|abnormal|deranged|elevated|deviant|outside)|greater deviation|compare|which of these", re.I)),
    ("threshold", re.compile(r"within|outside|above|below|meet|criteria|threshold|stage|normal", re.I)),
]


def classify_numeric(question: str) -> list[str]:
    cats = [name for name, r in _NUMERIC_CATS if r.search(question)]
    return cats or ["other"]


@dataclass
class Features:
    split: str
    id: str
    answer_type: str
    panel: str
    note_words: int
    question_words: int
    answer_words: int
    answer_sentences: int
    note_style: str
    sections: dict[str, bool]
    allergy: str
    medications: list[P.Medication]
    measurements: list[P.Measurement]  # non-delta, from note + table + question
    note_measurements: list[P.Measurement]  # non-delta, note only
    inline_bmi: list[float]
    age: int | None
    sex: str | None
    question_template: str
    is_bmi_question: bool
    uncertain_category: str | None
    numeric_categories: list[str] | None
    approx_prompt_tokens: int
    approx_answer_tokens: int


def _table_text(table: dict[str, Any]) -> str:
    return "\n".join(" | ".join(row) for row in [table.get("headers", [])] + table.get("rows", []))


def compute_features(split: str, r: dict[str, Any]) -> Features:
    note, q, a = r["note"], r["question"], r["answer"]
    measurements = P.record_body_measurements(r)
    age, sex = P.extract_age_sex(note)
    prompt_chars = len(note) + len(_table_text(r["table"])) + len(q)
    return Features(
        split=split,
        id=r["id"],
        answer_type=r["answer_type"],
        panel=P.table_panel(r["table"]),
        note_words=word_count(note),
        question_words=word_count(q),
        answer_words=word_count(a),
        answer_sentences=sentence_count(a),
        note_style=P.note_style(note),
        sections=P.section_presence(note),
        allergy=P.allergy_status(note),
        medications=P.extract_medications(note),
        measurements=measurements,
        note_measurements=[m for m in measurements if m.source == "note"],
        inline_bmi=P.extract_inline_bmi(note),
        age=age,
        sex=sex,
        question_template=mask_numbers(q),
        is_bmi_question=bool(_Q_BMI_RE.search(q)),
        uncertain_category=classify_uncertain(q, a) if r["answer_type"] == "uncertain" else None,
        numeric_categories=classify_numeric(q) if r["answer_type"] == "numeric_reasoning" else None,
        # ~4 characters per token; replaced by exact tokenizer counts once the base model is chosen.
        approx_prompt_tokens=round(prompt_chars / 4),
        approx_answer_tokens=round(len(a) / 4),
    )
