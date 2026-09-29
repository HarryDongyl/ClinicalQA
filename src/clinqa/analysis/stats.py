"""Descriptive statistics over the raw splits."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from clinqa.analysis.common import describe
from clinqa.analysis.features import Features

ANSWER_TYPES = ("extractive", "numeric_reasoning", "tool_call", "uncertain")
SPEC_DISTRIBUTION = {"extractive": 0.40, "numeric_reasoning": 0.20, "tool_call": 0.25, "uncertain": 0.15}


def _pct(n: int, d: int) -> float:
    return round(100 * n / d, 1) if d else 0.0


def _counter_with_pct(c: Counter, total: int, keys: list[str] | None = None) -> dict[str, dict[str, float]]:
    keys = keys if keys is not None else sorted(c)
    return {k: {"n": c.get(k, 0), "pct": _pct(c.get(k, 0), total)} for k in keys}


def compute_stats(
    data: dict[str, list[dict[str, Any]]],
    feats: dict[str, list[Features]],
    top_templates: int,
) -> dict[str, Any]:
    splits = list(data)
    out: dict[str, Any] = {"splits": splits}

    out["counts"] = {s: len(data[s]) for s in splits}
    out["answer_type"] = {
        s: _counter_with_pct(Counter(r["answer_type"] for r in data[s]), len(data[s]), list(ANSWER_TYPES)) for s in splits
    }
    out["answer_type_spec"] = {k: round(100 * v, 1) for k, v in SPEC_DISTRIBUTION.items()}

    out["note_words"] = {s: describe([f.note_words for f in feats[s]]) for s in splits}
    out["note_words_by_type"] = {
        s: {t: describe([f.note_words for f in feats[s] if f.answer_type == t]) for t in ANSWER_TYPES} for s in splits
    }
    out["question_words"] = {s: describe([f.question_words for f in feats[s]]) for s in splits}
    out["answer_words"] = {s: describe([f.answer_words for f in feats[s]]) for s in splits}
    out["answer_words_by_type"] = {
        t: describe([f.answer_words for s in splits for f in feats[s] if f.answer_type == t]) for t in ANSWER_TYPES
    }
    out["answer_sentences"] = {
        s: dict(sorted(Counter(min(f.answer_sentences, 6) for f in feats[s]).items())) for s in splits
    }
    out["approx_tokens"] = {
        "prompt": describe([f.approx_prompt_tokens for s in splits for f in feats[s]]),
        "answer": describe([f.approx_answer_tokens for s in splits for f in feats[s]]),
    }

    out["table_type"] = {
        s: _counter_with_pct(Counter(r["table"]["type"] for r in data[s]), len(data[s]), ["labs", "vitals"]) for s in splits
    }
    out["table_panel"] = {s: dict(sorted(Counter(f.panel for f in feats[s]).items())) for s in splits}
    out["table_rows"] = {s: describe([len(r["table"]["rows"]) for r in data[s]]) for s in splits}
    out["answer_type_by_table_type"] = {
        t: dict(sorted(Counter(r["table"]["type"] for s in splits for r in data[s] if r["answer_type"] == t).items()))
        for t in ANSWER_TYPES
    }
    by_panel: dict[str, Counter] = defaultdict(Counter)
    for s in splits:
        for f in feats[s]:
            by_panel[f.panel][f.answer_type] += 1
    out["answer_type_by_panel"] = {p: {t: c.get(t, 0) for t in ANSWER_TYPES} for p, c in sorted(by_panel.items())}

    tool = {}
    for s in splits:
        rs = data[s]
        with_calls = [r for r in rs if r.get("tool_calls")]
        calls = [tc for r in with_calls for tc in r["tool_calls"]]
        conversions = Counter(
            f"{tc['arguments'].get('from_unit')} -> {tc['arguments'].get('to_unit')} ({tc['arguments'].get('substance')})"
            for tc in calls
            if tc["tool"] == "unit_convert"
        )
        tool[s] = {
            "examples_with_tool_calls": len(with_calls),
            "pct_of_split": _pct(len(with_calls), len(rs)),
            "calls_per_example": dict(sorted(Counter(len(r["tool_calls"]) for r in with_calls).items())),
            "by_tool": dict(sorted(Counter(tc["tool"] for tc in calls).items())),
            "unit_convert_conversions": dict(sorted(conversions.items())),
        }
    out["tool_calls"] = tool

    out["note_style"] = {s: dict(sorted(Counter(f.note_style for f in feats[s]).items())) for s in splits}
    section_keys = list(feats[splits[0]][0].sections) if feats[splits[0]] else []
    out["section_presence_pct"] = {
        s: {k: _pct(sum(f.sections[k] for f in feats[s]), len(feats[s])) for k in section_keys} for s in splits
    }
    out["allergy_status"] = {s: dict(sorted(Counter(f.allergy for f in feats[s]).items())) for s in splits}
    out["inline_bmi_in_note"] = {s: sum(1 for f in feats[s] if f.inline_bmi) for s in splits}
    out["sex"] = {s: dict(sorted(Counter(f.sex or "unknown" for f in feats[s]).items())) for s in splits}
    out["age"] = {s: describe([f.age for f in feats[s] if f.age is not None]) for s in splits}
    out["medications_per_note"] = {s: describe([len(f.medications) for f in feats[s]]) for s in splits}

    def unit_system(f: Features) -> str:
        w = {m.unit for m in f.note_measurements if m.kind == "weight" and not m.equivalent}
        h = {m.unit for m in f.note_measurements if m.kind == "height" and not m.equivalent}
        return f"weight:{'/'.join(sorted(w)) or 'none'} height:{'/'.join(sorted(h)) or 'none'}"

    out["body_measurement_units_in_note"] = {
        t: dict(Counter(unit_system(f) for s in splits for f in feats[s] if f.answer_type == t).most_common())
        for t in ANSWER_TYPES
    }
    out["bmi_questions_by_type"] = {
        t: sum(1 for s in splits for f in feats[s] if f.answer_type == t and f.is_bmi_question) for t in ANSWER_TYPES
    }

    out["uncertain_category"] = {
        s: dict(sorted(Counter(f.uncertain_category for f in feats[s] if f.uncertain_category).items())) for s in splits
    }
    num = Counter()
    for s in splits:
        for f in feats[s]:
            for c in f.numeric_categories or []:
                num[c] += 1
    out["numeric_reasoning_categories"] = dict(sorted(num.items()))

    templates = {}
    for t in ANSWER_TYPES:
        c = Counter(f.question_template for s in splits for f in feats[s] if f.answer_type == t)
        n = sum(c.values())
        templates[t] = {
            "n_examples": n,
            "n_distinct_templates": len(c),
            "top": [{"template": k, "n": v} for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:top_templates]],
        }
    out["question_templates"] = templates
    return out
