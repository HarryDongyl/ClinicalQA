"""Deterministic per-example scoring (D-027, D-028, PLAN section 7).

score_example(record, trajectory) joins a canonical record (input + gold) with one
rollout trajectory from infer.py and returns flat booleans plus one primary error
category. No LLM judge; every rule is a small regex/number rule that the fixture tests
in tests/test_metrics.py pin down. Known limitations are listed in reports/REPORT.md.

Trajectory fields used here (see infer.py):
    turns[i].status    no_call | valid | invalid_json | unterminated | schema_error
    turns[i].calls     [{"name", "arguments"}]
    turns[i].results   executor outputs, aligned with calls
    final_answer       text of the last no-call assistant turn, or None
    stop_reason        answer | budget | max_tokens | parse_error
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from clinqa.analysis.features import classify_uncertain
from clinqa.parsing import extract_inline_bmi, record_body_measurements, to_metric
from clinqa.tools import normalize_substance, normalize_unit

# ---------------------------------------------------------------- numbers

_NUM = re.compile(r"(?<![\w.])[-−]?\d+(?:,\d{3})*(?:\.\d+)?")


@dataclass(frozen=True)
class Num:
    value: float
    decimals: int


def numbers(text: str) -> list[Num]:
    out = []
    for m in _NUM.finditer(text or ""):
        s = m.group(0).replace("−", "-").replace(",", "")
        out.append(Num(float(s), len(s.split(".")[1]) if "." in s else 0))
    return out


# Reference ranges and thresholds quoted as context ("70-100", "<150", ">=30") are not extracted facts.
_CONTEXT_NUM = re.compile(r"\d+(?:\.\d+)?\s*(?:-|–|—|to)\s*\d+(?:\.\d+)?|[<>≤≥]=?\s*\d+(?:\.\d+)?"
                          r"|\b(?:less|greater|more|lower|higher) than\s+\d+(?:\.\d+)?"
                          r"|\b(?:under|over|at least|at most|up to)\s+\d+(?:\.\d+)?", re.I)


def fact_numbers(text: str) -> list[Num]:
    return numbers(_CONTEXT_NUM.sub(" ", text or ""))


def number_matches(gold: Num, candidates: list[Num]) -> bool:
    """A candidate matches if it equals the gold number at the gold's own precision."""
    return any(math.isclose(round(c.value, gold.decimals), gold.value, abs_tol=1e-9) for c in candidates)


_UNIT_AFTER = re.compile(r"\s*([A-Za-z%µμ][\w/%µμ.²³^]*)?")


def numbers_with_units(text: str) -> list[tuple[Num, str]]:
    """Numbers with the unit-like token that follows them ('' if none), e.g. (150.4, 'mg/dl')."""
    out = []
    for m in _NUM.finditer(text or ""):
        s = m.group(0).replace("\u2212", "-").replace(",", "")
        unit = (_UNIT_AFTER.match(text, m.end()).group(1) or "").lower().rstrip(".")
        out.append((Num(float(s), len(s.split(".")[1]) if "." in s else 0), unit))
    return out


def patient_value(n: Num, unit: str, record: dict[str, Any]) -> bool:
    """The number is a patient measurement: a table Value cell, a question number, or a same-unit note number."""
    values = [x for row in record["table"]["rows"] for x in numbers(" ".join(row[1:2]))]
    if number_matches(n, values) or number_matches(n, numbers(record["question"])):
        return True
    return any(number_matches(n, [m]) and (not unit or not u or u == unit) for m, u in numbers_with_units(record["note"]))


def grounded_in_input(n: Num, unit: str, record: dict[str, Any]) -> bool:
    """Patient value or any other table cell (e.g. a reference-range bound)."""
    cells = [x for row in record["table"]["rows"] for x in numbers(" ".join(row[1:]))]
    return patient_value(n, unit, record) or number_matches(n, cells)


def input_text(record: dict[str, Any]) -> str:
    t = record["table"]
    cells = " ".join(" ".join(row) for row in t["rows"])
    return f"{record['note']} {cells} {record['question']}"


# ---------------------------------------------------------------- text classes

_ABSTAIN = re.compile(
    r"not (?:been |yet )?(?:documented|provided|recorded|available|stated|specified|mentioned|included|reported|listed|given|"
    r"known|present|charted|indicated|captured|confirmed|clear|noted|measured|obtained|in the (?:note|chart|record))"
    r"|(?:does|do|did) not (?:include|specify|document|mention|provide|state|contain|list|report|record|indicate|"
    r"note|give|allow)"
    r"|(?:is|are|was|were|has|have|does|do|did|could)n[’']t (?:been )?(?:documented|provided|recorded|available|stated|"
    r"specified|mentioned|included|reported|listed|given|noted|measured|include|specify|document|mention|provide|state|"
    r"contain|list|report|record|be )"
    r"|no (?:\w+ )?(?:documentation|documented|record|recorded|mention|information|data|timestamp|date|time)\b"
    r"|no \w+ (?:is|was|are|were) (?:given|documented|recorded|listed|provided)"
    r"|(?:cannot|could not|can[’']t|couldn[’']t) (?:be )?(?:determin|calculat|assess|confirm|establish|verif|comput|evaluat|"
    r"interpret|answer)"
    r"|(?:unable|impossible) to (?:determin|calculat|assess|confirm|establish|verif|comput)"
    r"|(?:needs?|would need) to be (?:obtained|confirmed|clarified|verified|documented)"
    r"|silent on|unavailable|undocumented|unspecified|not possible to|\bN/A\b"
    # loose words count only near a documentation noun (over-refusal precision)
    r"|(?:unknown|missing|lack\w*|insufficien\w*|absent)\W+(?:\w+\W+){0,4}?(?:document|record|note|chart|provided|"
    r"information|data)"
    r"|(?:document\w*|record\w*|note|chart|information|data)\W+(?:\w+\W+){0,4}?(?:unknown|missing|lack\w*|"
    r"insufficien\w*|absent)",
    re.I,
)

_DIRECTION = {
    "high": re.compile(r"\b(?:elevated|high|higher|above|exceed(?:s|ing|ed)?|increased|raised|hyperkal\w*|"
                       r"hypernatr\w*|hyperglyc\w*|hypercalc\w*|hyperlipid\w*|hypercholest\w*)\b", re.I),
    "low": re.compile(r"\b(?:low|lower|below|decreased|reduced|deficien\w*|hypokal\w*|hyponatr\w*|hypoglyc\w*|"
                      r"hypocalc\w*|hypoxi\w*)\b", re.I),
    "normal": re.compile(r"\bwithin (?:the )?(?:normal|reference|expected|target)\b|\bnormal limits\b"
                         r"|\b(?:is|are|was|were|remains?|which is) normal\b|\bunremarkable\b"
                         r"|\bin (?:the )?(?:normal|reference) range\b|\bwithin range\b|\(normal\)|\bWNL\b", re.I),
}
_CONFLICTS = {("high", "low"), ("low", "high"), ("normal", "high"), ("normal", "low"),
              ("high", "normal"), ("low", "normal")}
# Words that look directional but name a bound or a context, not the patient's value.
_NON_DIRECTIONAL = re.compile(r"\b(?:upper|lower) (?:limit|bound|end|reference)\w*|\bas (?:noted|mentioned) above\b"
                              r"|\b(?:above|below) (?:the )?threshold\b|\breduced ejection\b", re.I)
_NEGATION = re.compile(r"\b(?:not|no longer|neither|nor|never)\s+(?:\w+\s+){0,2}$|n[’']t\s+(?:\w+\s+){0,2}$", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

_MISSING_FIELD = {
    "height": [re.compile(r"height", re.I)],
    "weight": [re.compile(r"weight", re.I)],
    "height_and_weight": [re.compile(r"height", re.I), re.compile(r"weight", re.I)],
    "dose": [re.compile(r"\bdos(?:e|es|age|ing)\b|strength|\bhow much\b", re.I)],
    "allergy": [re.compile(r"allerg", re.I)],
    "timestamp": [re.compile(r"\btim(?:e|es|ing|estamp)\b|\bdates?\b|\bdated\b|\bwhen\b|\bcollected\b", re.I)],
    "other": [],
}

_UNIT_NUM = {
    "body": re.compile(r"(\d+(?:\.\d+)?)\s*(?:kg/m|kg\b|kilograms?\b|lbs?\b|pounds?\b|cm\b|centimet\w+|in\b|"
                       r"inches\b|m\b|met(?:er|re)s?\b|ft\b|feet\b|'|\")"
                       r"|\bBMI (?:of |is |was |would be |near |around |roughly |= ?|≈ ?|: ?)?(?:approximately |about |~)?"
                       r"(\d+(?:\.\d+)?)"
                       r"|\b(?:weight|height) (?:is |was |of |=|:)\s*(?:approximately |about |~)?(\d+(?:\.\d+)?)", re.I),
    "dose": re.compile(r"(\d+(?:\.\d+)?)\s*(?:mg|mcg|µg|μg|g|units?|mL|IU|milligrams?|micrograms?|grams?)\b", re.I),
    "timestamp": re.compile(r"(\d{1,2}:\d{2}|\d{1,4}[/-]\d{1,2}[/-]\d{1,4}|\b\d{1,2}\s*(?:am|pm|a\.m\.|p\.m\.)|"
                            r"\b\d{4}\s*h(?:rs|ours)?\b)", re.I),
}
# Comparators and illustrations ("<=60 kg", "e.g. 81 mg vs 325 mg") are not patient values.
_HEDGE = re.compile(r"[<>≤≥]\s*$|\be\.g\.,?\s*$|\bsuch as\s*$|\bvs\.?\s*$|\bor\s*$|threshold of\s*$", re.I)
_EXAMPLE = re.compile(r"\be\.g\.|\bsuch as\b|\bfor example\b|\b(?:standard|typical|usual|recommended|guideline)\b",
                      re.I)
_NKDA = re.compile(r"\bNKDA\b|no known (?:drug )?allerg|\bno (?:drug |medication )?allergies\b|"
                   r"\bdenies (?:any )?(?:drug |medication )?allergies\b|\ballergies:? none\b", re.I)


def is_abstention(text: str | None) -> bool:
    return bool(text) and bool(_ABSTAIN.search(text))


def directions(text: str) -> set[str]:
    """Direction classes asserted in text; negated mentions ("not elevated") are dropped."""
    text = _NON_DIRECTIONAL.sub(" ", text or "")
    out = set()
    for k, r in _DIRECTION.items():
        for m in r.finditer(text):
            if not _NEGATION.search(text[max(0, m.start() - 30):m.start()]):
                out.add(k)
    return out


def _fact_sentences(text: str, facts: list[Num]) -> str:
    """Sentences mentioning one of the fact numbers (all text if none do), so that direction words in
    unrelated sentences ("history of low back pain") are not read as the answer's direction."""
    sentences = _SENTENCE.split(text or "")
    keep = [x for x in sentences if facts and any(number_matches(n, numbers(x)) for n in facts)]
    return " ".join(keep) if keep else (text or "")


def direction_ok(gold: str, pred: str, facts: list[Num] | None = None) -> bool:
    facts = facts or []
    g, p = directions(_fact_sentences(gold, facts)), directions(_fact_sentences(pred, facts))
    return g <= p and not any((a, b) in _CONFLICTS for a in p - g for b in g)


_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = set("the a an of is are was were and or to in on for with at by as be this that it its patient patient's "
            "s which from has have had".split())


def token_f1(gold: str, pred: str) -> float:
    g = [w for w in _WORD.findall(gold.lower()) if w not in _STOP]
    p = [w for w in _WORD.findall((pred or "").lower()) if w not in _STOP]
    if not g or not p:
        return 0.0
    common = sum(min(g.count(w), p.count(w)) for w in set(g))
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------- answer-type rules

EXTRACTIVE_F1_THRESHOLD = 0.5


def number_dump(gold: str, pred: str) -> bool:
    """Guard against listing many values so that the gold numbers match by accident."""
    return len(numbers(pred)) > max(8, 3 * len(numbers(gold)))


def score_extractive(record: dict[str, Any], pred: str) -> tuple[bool, str]:
    gold = record["answer"]
    inp = numbers(input_text(record))
    grounded = [n for n in fact_numbers(gold) if number_matches(n, inp)]
    pn = numbers(pred)
    if grounded:
        if not all(number_matches(n, pn) for n in grounded):
            return False, "number_mismatch"
    elif token_f1(gold, pred) < EXTRACTIVE_F1_THRESHOLD:
        return False, "low_overlap"
    if not direction_ok(gold, pred, grounded):
        return False, "direction_mismatch"
    return True, "correct"


def score_numeric(record: dict[str, Any], pred: str) -> tuple[bool, str]:
    gold = record["answer"]
    facts = [(n, u) for n, u in numbers_with_units(_CONTEXT_NUM.sub(" ", gold))]
    derived = [n for n, u in facts if not grounded_in_input(n, u, record)]
    grounded = [n for n, u in facts if grounded_in_input(n, u, record)]
    pn = numbers(pred)
    if derived and not all(number_matches(n, pn) for n in derived):
        return False, "number_mismatch"
    patient = [n for n, u in facts if patient_value(n, u, record)]
    other_values = [x for row in record["table"]["rows"] for x in numbers(row[1])
                    if not number_matches(x, patient)]
    if (derived and patient and not any(number_matches(n, pn) for n in patient)
            and any(number_matches(x, pn) for x in other_values)):
        return False, "number_mismatch"  # right result attached to another analyte's value
    if not derived and grounded and sum(number_matches(n, pn) for n in grounded) * 2 < len(grounded):
        return False, "number_mismatch"
    if not direction_ok(gold, pred, derived + grounded):
        return False, "direction_mismatch"
    return True, "correct"


def fabricated_values(record: dict[str, Any], category: str, pred: str) -> list[str]:
    """Values of the missing field that the prediction states but the input does not contain."""
    inp_text = input_text(record)
    inp = numbers(inp_text)
    kind = {"height": "body", "weight": "body", "height_and_weight": "body",
            "dose": "dose", "timestamp": "timestamp"}.get(category)
    out = []
    if kind == "timestamp":
        out = [m.group(0) for m in _UNIT_NUM["timestamp"].finditer(pred) if m.group(0) not in inp_text]
    elif kind:
        # Only same-kind input values exempt a number: a note's heart rate does not license a stated weight.
        if kind == "body":
            ms = record_body_measurements(record)
            allowed = ([Num(m.value, 2) for m in ms] + [Num(round(to_metric(m.value, m.unit), 1), 1) for m in ms]
                       + [Num(v, 1) for v in extract_inline_bmi(record["note"])])
        else:
            allowed = []
        # same-kind mentions anywhere in the input (e.g. a documented "5 kg weight loss", "metformin 500 mg")
        allowed += [numbers(next(g for g in m.groups() if g))[0] for m in _UNIT_NUM[kind].finditer(inp_text)]
        for m in _UNIT_NUM[kind].finditer(pred):
            s = next(g for g in m.groups() if g)
            before = pred[max(0, m.start() - 40):m.start()]
            if number_matches(numbers(s)[0], allowed) or _HEDGE.search(before[-16:]) or _EXAMPLE.search(before):
                continue
            out.append(m.group(0))
    if category == "allergy" and not _NKDA.search(inp_text):
        # "NKDA" is fabricated only when asserted, not when named as the missing documentation.
        for sentence in re.split(r"(?<=[.!?])\s+", pred):
            m = _NKDA.search(sentence)
            if m and not (_ABSTAIN.search(sentence) or re.search(r"\b(?:no|not|without)\b.*" + re.escape(m.group(0)),
                                                                 sentence, re.I)):
                out.append(m.group(0))
    return out


def score_uncertain(record: dict[str, Any], pred: str) -> tuple[bool, str, str]:
    category = classify_uncertain(record["question"], record["answer"])
    if fabricated_values(record, category, pred):
        return False, "fabricated_value", category
    if not is_abstention(pred):
        return False, "no_abstention", category
    if not all(r.search(pred) for r in _MISSING_FIELD[category]):
        return False, "missing_field_not_named", category
    return True, "correct", category


# ---------------------------------------------------------------- tool rules

ARG_HALF_UNIT = 0.05  # half a unit of the last digit of 1-dp metric gold arguments (D-027)


def _close(a: Any, b: float, tol: float = 1e-6) -> bool:
    return isinstance(a, (int, float)) and not isinstance(a, bool) and abs(float(a) - b) <= tol


def bmi_arg_ok(record: dict[str, Any], kind: str, gold: float, pred: Any) -> tuple[bool, str]:
    """(correct, source) for one calculate_bmi argument; source is metric | imperial | ungrounded."""
    imperial_unit = {"weight": "lb", "height": "in"}[kind]
    ms = [m for m in record_body_measurements(record) if m.kind == kind and not m.is_delta]
    imperial = [m for m in ms if m.unit == imperial_unit]
    metric = [m for m in ms if m.unit != imperial_unit]
    source = "metric" if any(abs(m.value - gold) < 1e-6 for m in metric) else "imperial" if imperial else "ungrounded"
    if _close(pred, gold):
        return True, source
    if source == "imperial":
        m = min(imperial, key=lambda m: abs(to_metric(m.value, m.unit) - gold))
        exact = round(to_metric(m.value, m.unit), 1)
        if _close(pred, gold, ARG_HALF_UNIT + 1e-9) or _close(pred, exact, ARG_HALF_UNIT + 1e-9):
            return True, source
    return False, source


def args_ok(record: dict[str, Any], gold_call: dict[str, Any], call: dict[str, Any]) -> tuple[bool, str]:
    """(correct, detail) comparing a parsed call's arguments with the gold call (D-027)."""
    ga, pa = gold_call["arguments"], call.get("arguments") or {}
    if gold_call["tool"] == "calculate_bmi":
        details = []
        ok = True
        for key, kind in (("weight_kg", "weight"), ("height_cm", "height")):
            good, source = bmi_arg_ok(record, kind, ga[key], pa.get(key))
            ok &= good
            details.append(f"{key}:{source}:{'ok' if good else 'wrong'}")
        return ok, ",".join(details)
    ok = (_close(pa.get("value"), ga["value"], 1e-9)
          and normalize_unit(pa.get("from_unit", "")) == normalize_unit(ga["from_unit"])
          and normalize_unit(pa.get("to_unit", "")) == normalize_unit(ga["to_unit"])
          and normalize_substance(pa.get("substance")) == normalize_substance(ga.get("substance")))
    return ok, "unit_convert:" + ("ok" if ok else "wrong")


def unsupported_call(record: dict[str, Any], call: dict[str, Any]) -> bool:
    """True if a call's numeric arguments are not recoverable from the input (D-027, PLAN section 7).

    calculate_bmi: each argument must equal a metric input measurement of its kind or be within
    ARG_HALF_UNIT of an imperial one converted; unit_convert: value must be an input number.
    """
    args = call.get("arguments") or {}
    if call.get("name") == "calculate_bmi":
        ms = [m for m in record_body_measurements(record) if not m.is_delta]
        for key, kind in (("weight_kg", "weight"), ("height_cm", "height")):
            v = args.get(key)
            if not any(_close(v, m.value) if m.unit in ("kg", "cm") else _close(v, to_metric(m.value, m.unit),
                                                                             ARG_HALF_UNIT + 1e-9)
                       for m in ms if m.kind == kind):
                return True
        return False
    if call.get("name") == "unit_convert":
        v = args.get("value")
        return not any(_close(v, n.value, 1e-9) for n in numbers(input_text(record)))
    return True


def _all_calls(traj: dict[str, Any]) -> list[tuple[dict[str, Any], Any, str]]:
    out = []
    for t in traj.get("turns", []):
        results = t.get("results") or [None] * len(t.get("calls", []))
        out += [(c, r, t["status"]) for c, r in zip(t.get("calls", []), results)]
    return out


def _first_bad_status(traj: dict[str, Any]) -> str | None:
    for t in traj.get("turns", []):
        if t["status"] in ("invalid_json", "unterminated", "schema_error"):
            return t["status"]
    return None


# ---------------------------------------------------------------- per-example


@dataclass
class Score:
    id: str
    answer_type: str
    correct: bool  # the headline metric for this answer type (tool: tool E2E)
    error: str
    abstained: bool
    over_refusal: bool | None  # only on non-uncertain examples
    called: bool
    over_call: bool | None  # only on non-tool examples
    tool_selected: bool | None
    tool_args_correct: bool | None
    tool_executed: bool | None
    result_in_answer: bool | None
    tool_e2e: bool | None
    arg_detail: str | None
    uncertain_category: str | None
    unsupported_args: bool | None  # any call with arguments not recoverable from the input; None if no call

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_example(record: dict[str, Any], traj: dict[str, Any]) -> Score:
    at = record["answer_type"]
    final = traj.get("final_answer") or ""
    calls = _all_calls(traj)
    called = bool(calls) or _first_bad_status(traj) is not None
    abstained = is_abstention(final)
    base = dict(id=record["id"], answer_type=at, abstained=abstained, called=called,
                over_refusal=None if at == "uncertain" else abstained,
                over_call=None if at == "tool_call" else called,
                tool_selected=None, tool_args_correct=None, tool_executed=None, result_in_answer=None,
                tool_e2e=None, arg_detail=None, uncertain_category=None,
                unsupported_args=any(unsupported_call(record, c) for c, _, _ in calls) if calls else None)

    if at == "tool_call":
        gold_call = record["tool_calls"][0]
        bad = _first_bad_status(traj)
        first = calls[0] if calls else None
        selected = bool(first) and first[0]["name"] == gold_call["tool"]
        schema_ok = bool(first) and first[2] == "valid"
        a_ok, detail = args_ok(record, gold_call, first[0]) if selected else (False, None)
        a_ok = a_ok and schema_ok
        result = first[1] if first else None
        executed = isinstance(result, (int, float)) and not isinstance(result, bool)
        in_answer = executed and number_matches(numbers(str(result))[0], numbers(final))
        extra = len(calls) > 1
        e2e = selected and a_ok and executed and in_answer and not extra and traj.get("stop_reason") == "answer"
        if e2e:
            err = "correct"
        elif bad:
            err = bad
        elif not calls:
            err = "no_call_abstained" if abstained else "no_call"
        elif not selected:
            err = "wrong_tool"
        elif not schema_ok:
            err = "schema_error"
        elif not a_ok:
            err = "wrong_args_" + ("ungrounded" if "ungrounded" in (detail or "") else
                                   "imperial" if "imperial:wrong" in (detail or "") else "value")
        elif not executed:
            err = "execution_error"
        elif extra:
            err = "extra_call"
        elif traj.get("stop_reason") != "answer":
            err = f"stopped_{traj.get('stop_reason')}"
        else:
            err = "result_not_in_answer"
        return Score(**{**base, "correct": e2e, "error": err, "tool_selected": selected, "tool_args_correct": a_ok,
                        "tool_executed": executed, "result_in_answer": in_answer, "tool_e2e": e2e,
                        "arg_detail": detail})

    if not final.strip():
        return Score(**{**base, "correct": False,
                        "error": "invalid_call" if _first_bad_status(traj) else f"no_answer_{traj.get('stop_reason')}"})
    if at == "uncertain":
        ok, err, cat = score_uncertain(record, final)
        return Score(**{**base, "correct": ok, "error": err, "uncertain_category": cat})
    if number_dump(record["answer"], final):
        return Score(**{**base, "correct": False, "error": "number_dump"})
    ok, err = (score_extractive if at == "extractive" else score_numeric)(record, final)
    if not ok and abstained:
        err = "over_refusal"
    return Score(**{**base, "correct": ok, "error": err})


def gold_trajectory(record: dict[str, Any]) -> dict[str, Any]:
    """The gold demonstration expressed as a trajectory (scorer self-check, never a prediction)."""
    if record["answer_type"] == "tool_call":
        c = record["tool_calls"][0]
        turns = [{"status": "valid", "calls": [{"name": c["tool"], "arguments": c["arguments"]}],
                  "results": [c["result"]]},
                 {"status": "no_call", "calls": [], "results": []}]
    else:
        turns = [{"status": "no_call", "calls": [], "results": []}]
    return {"id": record["id"], "turns": turns, "final_answer": record["answer"], "stop_reason": "answer"}
