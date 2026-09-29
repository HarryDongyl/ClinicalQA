"""Question-contract scoring, v2. No model calls, lexical F1, or test-set reads.

This is a bounded semantic checker, not a universal natural-language judge.
Unknown semantics remain review, never an implicit failure or success. Contracts
are derived before seeing predictions. The legacy scorer is retained as evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ast
import math
import re
from typing import Any

from clinqa import metrics as legacy
from clinqa.analysis.features import classify_uncertain
from clinqa.parsing import NOTE_ANALYTE_ALIASES, parse_ref_range, record_body_measurements

VERSION = "2.0.0"
N = r"(?<![\w.])[-−]?\d+(?:,\d{3})*(?:\.\d+)?"
ALL_STATES = {"low", "normal", "high"}


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("**", "").replace("__", "").replace("−", "-")).strip()


def contains(text: str, phrase: str) -> bool:
    return bool(re.search(r"(?<![\w])" + re.escape(phrase) + r"(?![\w])", text, re.I))


def aliases(name: str) -> list[str]:
    extra = {"WBC": ["white blood cell count", "white blood count"], "SpO2": ["oxygen saturation"],
             "eGFR": ["estimated glomerular filtration rate"], "Serum Iron": ["iron level"],
             "Total Cholesterol": ["cholesterol level"], "BMI": ["body mass index"]}
    return list(dict.fromkeys([name, *NOTE_ANALYTE_ALIASES.get(name, []), *extra.get(name, [])]))


@dataclass
class Fact:
    name: str
    value: float
    unit: str
    reference: str | None
    low: float | None = None
    high: float | None = None
    state: str | None = None
    source: str = "table"


def table_facts(record: dict) -> list[Fact]:
    facts = []
    rows = [(row, "table") for row in record["table"]["rows"]]
    # Many encounters contain a vitals table and explicitly ranged labs in prose.
    # These are input facts, not external reference ranges or gold-answer facts.
    note = record["note"].replace("**", "")
    for name in NOTE_ANALYTE_ALIASES:
        for alias in aliases(name):
            pattern = (r"(?<!\w)" + re.escape(alias) + r"(?!\w)\s*:?\s*(\d+(?:\.\d+)?)"
                       r"\s*([^(),\n]*?)\s*\(\s*ref(?:erence)?\s*:?\s*"
                       r"([<>≤≥]=?\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?)")
            for m in re.finditer(pattern, note, re.I):
                rows.append(([name, m.group(1), m.group(2).strip(), m.group(3)], "note"))
    seen = set()
    for row, source in rows:
        try:
            value = float(row[1])
        except (ValueError, TypeError):
            continue
        ref = row[3] if len(row) > 3 else None
        bounds = parse_ref_range(ref) if ref else None
        identity = (row[0], value, ref)
        if identity in seen:
            continue
        seen.add(identity)
        f = Fact(row[0], value, row[2], ref, source=source)
        if bounds:
            f.low, f.high = bounds["low"], bounds["high"]
            f.state = "normal"
            # Respect strict inequalities at the boundary, unlike interval-only parsing.
            if f.low is not None and (value < f.low or (ref.strip().startswith(">") and not ref.strip().startswith(">=") and value == f.low)):
                f.state = "low"
            if f.high is not None and (value > f.high or (ref.strip().startswith("<") and not ref.strip().startswith("<=") and value == f.high)):
                f.state = "high"
        facts.append(f)
    return facts


@dataclass
class Slot:
    kind: str
    subject: str
    expected: Any
    unit: str = ""
    relation: str | None = None
    bound: float | None = None
    source: str = "input"


@dataclass
class Contract:
    id: str
    answer_type: str
    slots: list[Slot] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    policy: str = "answer"


def compile_contract(record: dict) -> Contract:
    """Compile from question + input only (gold used solely for uncertainty category)."""
    c = Contract(record["id"], record["answer_type"], facts=table_facts(record))
    q = clean(record["question"]).lower()
    for name in {f.name for f in c.facts}:
        if len({(f.value, f.low, f.high) for f in c.facts if f.name == name}) > 1:
            c.review_reasons.append("conflicting_input_facts:" + name)
    if c.answer_type == "tool_call":
        c.policy = "tool"
        if record["tool_calls"][0]["tool"] == "calculate_bmi":
            ms = record_body_measurements(record)
            c.missing_fields = [k for k in ("weight", "height") if not any(m.kind == k for m in ms)]
            if c.missing_fields:
                c.policy = "missing_tool_inputs"
        return c
    if c.answer_type == "uncertain":
        c.policy = "uncertain"
        category = classify_uncertain(record["question"], record["answer"])
        c.missing_fields = category.split("_and_")
        return c
    mentioned = [f for f in c.facts if any(contains(q, a) for a in aliases(f.name))]
    mentioned.sort(key=lambda f: min(m.start() for a in aliases(f.name)
                                    for m in re.finditer(r"(?<!\w)"+re.escape(a)+r"(?!\w)", q, re.I)))
    if c.answer_type == "extractive":
        if not mentioned:
            c.review_reasons.append("non_table_or_unresolved_question: semantic adjudication required")
            return c
        # Only closed table-value/range questions are automatically complete.
        if re.search(r"\b(action|plan|medication|taking|prescribed|finding|cause|concern|assessment|correlate)\b", q):
            c.review_reasons.append("additional_note_fact_requested")
        for f in mentioned:
            if not q.startswith(("is ", "was ", "does ", "are ")):
                c.slots.append(Slot("value", f.name, f.value, f.unit))
            if re.search(r"\b(normal|reference|range|elevated|low|high)\b", q):
                if f.state:
                    binary = bool(re.search(r"\bwithin\b", q))
                    c.slots.append(Slot("membership" if binary else "state", f.name, f.state))
                else:
                    c.review_reasons.append("reference_not_supplied")
        return c
    # Numeric BMI questions require a calculation, not all intermediate gold numbers.
    if "bmi" in q or "body mass index" in q:
        ms = record_body_measurements(record)
        values = {k: {round(m.metric_value, 6) for m in ms if m.kind == k} for k in ("weight", "height")}
        if all(len(v) == 1 for v in values.values()):
            bmi = next(iter(values["weight"])) / (next(iter(values["height"])) / 100) ** 2
            c.slots.append(Slot("value", "BMI", round(bmi, 1), "kg/m²"))
            threshold = re.search(r"threshold of (\d+(?:\.\d+)?)", q)
            if threshold:
                b = float(threshold.group(1))
                c.slots.append(Slot("threshold", "BMI", "high" if bmi > b else "normal", bound=b))
        else:
            c.review_reasons.append("ambiguous_or_missing_body_measurements")
        return c
    # Clinical significance/staging is a distinct required obligation, not a keyword test.
    if re.search(r"clinical|ckd stage|explanation|consistent with|low.grade or high.grade", q):
        c.review_reasons.append("clinical_interpretation_requires_adjudication")
    if not mentioned:
        desired_states = {"low"} if "below" in q else {"high"} if "elevated" in q or "exceed" in q else {"high", "low"}
        abnormal = [f for f in c.facts if f.state in desired_states]
        if len(abnormal) == 1:
            mentioned = abnormal
        elif re.search(r"times|factor|percentage|proportion|relative", q) and abnormal:
            def relative(f):
                b = f.high if f.state == "high" else f.low
                return abs(f.value - b) / b if b else -1
            largest = max(relative(f) for f in abnormal)
            mentioned = [f for f in abnormal if math.isclose(relative(f), largest)]
            if len(mentioned) != 1:
                c.review_reasons.append("tied_comparison")
        else:
            c.review_reasons.append("ambiguous_ranking_scale_or_target")
    if not mentioned:
        return c
    # Determine which operation each question clause requests for each subject.
    for idx, f in enumerate(mentioned):
        local_q = q
        if len(mentioned) > 1:
            positions = sorted((m.start(), g.name) for g in mentioned for a in aliases(g.name)
                               for m in re.finditer(r"(?<!\w)" + re.escape(a.lower()) + r"(?!\w)", q))
            starts = [p for p, name in positions if name == f.name]
            pos = starts[0] if starts else 0
            # Include the operation preceding the analyte, but not the preceding clause.
            start = max(q.rfind(" and ", 0, pos), q.rfind(",", 0, pos))
            end = min([p for p, name in positions if p > pos and name != f.name] or [len(q)])
            separator = q.rfind(" and ", pos, end)
            if separator != -1:
                end = separator
            local_q = q[max(0, start):end]
        # Later questions about OTHER tests do not change this subject's boundary.
        operation_q = re.split(r"\band which other\b|\band is (?:any |the )?other\b", local_q)[0]
        b = f.low if re.search(r"below|lower limit|lower threshold", operation_q) else f.high
        if b is None:
            # A question may explicitly supply a reference when a vitals table does not.
            threshold = re.search(r"(?:threshold|upper limit).*?\b(\d+(?:\.\d+)?)", q)
            rng = re.search(r"\((\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s*bpm", q)
            if rng:
                b = float(rng.group(2))
            elif threshold:
                b = float(threshold.group(1))
        difference = bool(re.search(r"how (?:much|far|close)|how many (?!times)|by how (?!many times)", local_q))
        ratio = bool(re.search(r"times|factor", local_q))
        if idx and "compare" in q and re.search(r"times|factor", q) and not re.search(r"below|lower", local_q):
            ratio = True
        percent = "percentage" in local_q and "percentage points" not in local_q
        if idx and re.search(r"also below|remain within", local_q):
            difference = False
        if b is not None:
            relation = "high" if f.value > b else "low" if f.value < b else "normal"
            if difference:
                c.slots.append(Slot("difference", f.name, round(abs(f.value-b), 6), f.unit, relation, b))
            if ratio:
                c.slots.append(Slot("ratio", f.name, f.value/b, "", relation, b))
            if percent:
                c.slots.append(Slot("percent", f.name, abs(f.value-b)/b*100, "%", relation, b))
        elif difference or ratio or percent:
            c.review_reasons.append("requested_boundary_not_in_input")
        if idx and re.search(r"also below|within.*range", local_q) and f.state:
            c.slots.append(Slot("state", f.name, f.state))
    if re.search(r"bun.to.creatinine ratio", q):
        by = {f.name: f for f in c.facts}
        if "BUN" in by and "Creatinine" in by:
            c.slots.append(Slot("ratio", "BUN", by["BUN"].value/by["Creatinine"].value, source="input:BUN/Creatinine"))
    if re.search(r"which other|is any other|are any|only .*outside|which.*remain within", q):
        requested = [f for f in c.facts if f.state and f.name not in {g.name for g in mentioned}]
        if "which other" in q and re.search(r"(?:which other).*?below", q):
            requested = [f for f in requested if f.state == "low"]
        elif "which other" in q and re.search(r"(?:which other).*?(?:exceed|above)", q):
            requested = [f for f in requested if f.state == "high"]
        for f in requested:
            # Listing normal comparators is optional if an equivalent collective answer is present.
            c.slots.append(Slot("panel_state", f.name, f.state))
    if "most" in q:
        abnormal = [f for f in c.facts if f.state in {"high", "low"}]
        if len(abnormal) == 1:
            c.slots.append(Slot("identity", abnormal[0].name, abnormal[0].name))
        elif not re.search(r"times|factor|percentage|proportion|relative", q):
            c.review_reasons.append("ambiguous_ranking_scale")
        else:
            c.review_reasons.append("comparative_conclusion_requires_adjudication")
    if len(mentioned) > 1 and "compare" in q:
        c.review_reasons.append("comparative_conclusion_requires_adjudication")
    if not c.slots:
        c.review_reasons.append("no_supported_numeric_obligation")
    c.review_reasons = list(dict.fromkeys(c.review_reasons))
    return c


def contexts(text: str, facts: list[Fact], subject: str, default_subject: str | None) -> str:
    """Track a discourse subject across sentences; switch at another explicit analyte.

    This is deliberately bounded coreference, not whole-answer number matching.
    Cross-analyte comparisons with ambiguous scope are left for review.
    """
    names = {f.name for f in facts} | {subject}
    spans = []
    for name in names:
        for a in aliases(name):
            for m in re.finditer(r"(?<!\w)" + re.escape(a) + r"(?!\w)", text, re.I):
                spans.append((m.start(), m.end(), name))
    spans.sort(key=lambda s: (s[0], -(s[1]-s[0]), s[2]))
    selected = []
    for span in spans:
        if not selected or span[0] >= selected[-1][1]:
            selected.append(span)
    if not selected:
        return text if default_subject == subject else ""
    pieces = []
    if selected[0][0] > 0 and default_subject == subject:
        pieces.append(text[:selected[0][0]])
    for i, (start, _, name) in enumerate(selected):
        if name == subject:
            end = selected[i+1][0] if i+1 < len(selected) else len(text)
            piece = text[start:end]
            # A predicate introducing the NEXT named analyte is not about this one.
            piece = re.sub(r"(?:Therefore|Thus|Hence)[^.!?]*\b(?:only|most|least)[^.!?]*\b(?:is|are)\s*$", "", piece, flags=re.I)
            pieces.append(piece)
    return " ".join(pieces)


def state_evidence(text: str, fact: Fact | None = None) -> list[set[str]]:
    """Each assertion denotes possible states. Negation complements its predicate.

    'not within' means {low, high}, not the empty set and not normal.
    'not high' means {low, normal}; it cannot prove normal by itself.
    """
    t = clean(text).lower()
    # Comparisons to a LOWER versus UPPER boundary entail different state sets.
    boundary_assertions = []
    if fact:
        explicit_bound = re.compile(r"\b(above|below)\s+(?:(?:the |its )?reference (?:threshold|limit) of\s*[<>]?\s*)?(\d+(?:\.\d+)?)")
        def resolve_bound(m):
            value = float(m.group(2))
            if fact.high is not None and math.isclose(value, fact.high):
                boundary_assertions.append({"high"} if m.group(1) == "above" else {"low", "normal"})
                return ""
            if fact.low is not None and math.isclose(value, fact.low):
                boundary_assertions.append({"normal", "high"} if m.group(1) == "above" else {"low"})
                return ""
            return m.group()
        t = explicit_bound.sub(resolve_bound, t)
    boundary = re.compile(r"\b(above|below|exceeds?|higher than|lower than)\s+(?:the |its )?(upper|lower)\s+(?:normal |reference )?(?:limit|bound|threshold|end)(?: of (?:the )?(?:normal|reference)(?: range)?)?")
    for m in boundary.finditer(t):
        high = m.group(1) in {"above", "exceed", "exceeds", "higher than"}
        states = ({"high"} if high else {"low", "normal"}) if m.group(2) == "upper" else ({"normal", "high"} if high else {"low"})
        before = t[max(0, m.start()-35):m.start()]
        if re.search(r"\b(?:not|no|isn't|doesn't)\s+(?:\w+\s+){0,2}$", before):
            states = ALL_STATES-states
        boundary_assertions.append(states)
    t = boundary.sub("", t)
    t = re.sub(r"\b(?:upper|lower)\s+(?:normal\s+|reference\s+)?(?:limit|bound|threshold|end)(?: of (?:the )?(?:normal|reference)(?: range)?)?\b", "boundary", t)
    t = re.sub(r"\b(?:high|low) end of (?:the )?normal", "normal", t)
    t = re.sub(r"\b(?:iron|factor|vitamin) deficiency\b|reduced ejection fraction|lower extremit\w*|high risk|low risk", "", t)
    # A distance to a boundary is not a range-membership assertion.
    t = re.sub(N + r"\s*(?:[a-z%µμ/²³]+\s+)?(?:above|below|over|under)\s+(?:the\s+)?boundary", "", t)
    t = re.sub(r"\b(?:above|below)\s+(?:the\s+|its\s+)?(?:reference\s+)?(?:target|boundary|threshold|limit)\b", "", t)
    patterns = [
        (r"\b(?:outside|out of)\s+(?:the\s+)?(?:normal\s+|reference\s+)?(?:range|limits)\b|\babnormal\b", {"low", "high"}),
        (r"\b(?:within|inside|in)\s+(?:the\s+|their\s+)?(?:respective\s+)?(?:normal\s+|reference\s+|expected\s+)*(?:range|limits)\b|\bnormal\b|\bwnl\b|\bunremarkable\b", {"normal"}),
        (r"\b(?:elevated|raised|high|higher|above|exceeds?|exceeded|exceeding|increased|prolonged)\b", {"high"}),
        (r"\b(?:low|lower|below|decreased|reduced|depressed|shortened)\b", {"low"}),
    ]
    assertions, occupied = boundary_assertions, []
    for pattern, states in patterns:
        for m in re.finditer(pattern, t):
            if any(a <= m.start() < b for a, b in occupied):
                continue
            before, after = t[max(0, m.start()-40):m.start()], t[m.end():m.end()+35]
            # Naming a normal range/limit does not assert normal patient physiology.
            if m.group() == "normal" and re.match(r"\s+(?:reference\s+)?(?:range|limit|threshold)", after):
                continue
            if m.group() == "normal" and not (re.search(r"(?:\b(?:is|are|was|were|be|remains|all|and|not|considered)|[(:,])\s*$", before)
                                               or not before.strip()):
                continue
            if re.search(r"\b(?:range|reference|boundary)\s+(?:of\s+)?$", before):
                continue
            negated = bool(re.search(r"\b(?:not|no|never|no longer|isn't|isn’t|doesn't|doesn’t)\s+(?:\w+\s+){0,2}$", before))
            assertions.append(ALL_STATES - states if negated else states)
            occupied.append((m.start(), m.end()))
    return assertions


def strict_abstention(text: str) -> bool:
    # Remove clinical inference negations, which do not deny available information.
    text = re.sub(r"\b(?:does|do|did) not indicate\b[^.;]*", "", text, flags=re.I)
    return legacy.is_abstention(text)


def check(kind: str, subject: str, status: str, reason: str, **kwargs) -> dict:
    return {"kind": kind, "subject": subject, "status": status, "reason": reason, **kwargs}


def number_candidates(text: str, kind: str, unit: str = "") -> list[tuple[float, int, str]]:
    """Extract role-bearing numbers; a matching number elsewhere is not sufficient."""
    candidates = []
    for m in re.finditer(N, text):
        value = float(m.group().replace(",", "").replace("−", "-"))
        decimals = len(m.group().split(".")[1]) if "." in m.group() else 0
        before, after = text[max(0, m.start()-65):m.start()].lower(), text[m.end():m.end()+65].lower()
        # Reference bounds, identifiers and denominator exponents are not answers.
        if re.search(r"(?:range|limit|threshold|boundary)\s+(?:of\s+)?$|[<>≤≥]\s*$", before):
            continue
        if kind == "value":
            if re.search(r"\bby\s*$", before):
                continue
        elif kind == "difference":
            if not (re.search(r"\b(?:by|difference(?: of| is|:)?|deviation(?: of| is|:)?|shortfall(?: of| is|:)?)\s*(?:about |approximately |~|only )?$", before)
                    or re.match(r"\s*(?:\S+\s+){0,3}(?:above|below|higher|lower|over|under|short of)\b", after)
                    or re.search(r"[=≈]\s*$", before)):
                continue
            if (unit != "%" and re.match(r"\s*%\s*(?:above|below)", after)) or re.match(r"\s*(?:times|fold)", after):
                continue
        elif kind == "ratio":
            if not (re.match(r"\s*(?:times|[- ]?fold|:1\b|×|x\b)", after)
                    or re.search(r"\bratio\s*(?:is|of|=|:)?\s*(?:approximately\s*)?$|[=≈]\s*$", before)):
                continue
        elif kind == "percent":
            if not re.match(r"\s*%(?!\s*(?:points|percentage points))", after):
                continue
            if not (re.match(r"\s*%\s*(?:above|below|higher|lower|deviation|elevation|deficit)", after)
                    or re.search(r"[=≈]\s*$|\b(?:approximately|about|percentage(?: of| is)?)\s*$", before)):
                continue
        candidates.append((value, decimals, m.group()))
    return candidates


def numeric_match(value: float, expected: float, decimals: int, kind: str) -> bool:
    if kind in {"ratio", "percent"}:
        # Presentation rounding to at least one decimal for ratios; whole percentages permitted.
        # No broad relative tolerance or test-derived tolerance changes.
        precision = max(1, decimals) if kind == "ratio" else decimals
        return abs(value - expected) <= 0.5 * 10**(-precision) + 1e-8
    return abs(value-expected) <= 0.05000001


def evaluate_slot(slot: Slot, text: str, contract: Contract) -> dict:
    subjects = {s.subject for s in contract.slots if s.kind != "panel_state"}
    default = next(iter(subjects)) if len(subjects) == 1 else None
    scoped = contexts(text, contract.facts, slot.subject, default)
    if slot.kind == "identity":
        return check(slot.kind, slot.subject, "pass" if any(contains(text, a) for a in aliases(slot.subject)) else "review",
                     "named_subject" if scoped else "identity_not_resolved")
    if slot.kind in {"membership", "state", "panel_state", "threshold"}:
        fact = next((f for f in contract.facts if f.name == slot.subject), None)
        states = state_evidence(scoped, fact)
        if slot.kind == "panel_state" and not states:
            if re.search(r"\b(?:all|other|remaining|both)\b.{0,300}\b(?:normal|within)\b", text, re.I):
                states = [{"normal"}]
        if slot.kind == "membership":
            accepted = {"normal"} if slot.expected == "normal" else {"low", "high"}
        else:
            accepted = {slot.expected}
        if any(not (s & accepted) for s in states):
            if re.search(r"\bcorrection\s*:", scoped, re.I):
                return check(slot.kind, slot.subject, "review", "explicit_self_correction", evidence=scoped)
            return check(slot.kind, slot.subject, "fail", "contradicted_range", expected=sorted(accepted), evidence=scoped)
        if any(s <= accepted for s in states) or (states and set.intersection(*states) and set.intersection(*states) <= accepted):
            return check(slot.kind, slot.subject, "pass", "entailed_range", expected=sorted(accepted), evidence=scoped)
        # A direct yes/no answers a binary question without repeating any numbers.
        direct = re.match(r"\s*(yes|no)\b", text, re.I)
        if direct and slot.kind == "membership" and default == slot.subject:
            yes = direct.group(1).lower() == "yes"
            return check(slot.kind, slot.subject, "pass" if yes == (slot.expected == "normal") else "fail", "direct_binary_answer")
        return check(slot.kind, slot.subject, "review", "range_assertion_unresolved", evidence=scoped)
    candidates = number_candidates(scoped, slot.kind, slot.unit)
    matched = [(v, d) for v, d, _ in candidates if numeric_match(v, slot.expected, d, slot.kind)]
    if matched:
        if slot.bound is not None:
            boundary = re.compile(r"\b(upper|lower)\s+(?:normal\s+|reference\s+)?(?:limit|threshold)"
                                  r"(?:\s+of\s+(?:the\s+)?(?:normal|reference)(?:\s+range)?)?"
                                  r"\s*(?:of\s+|is\s+|[(:]\s*)?[<>≤≥]?\s*(\d+(?:\.\d+)?)", re.I)
            fact = next((f for f in contract.facts if f.name == slot.subject), None)
            for m in boundary.finditer(scoped):
                if re.match(r"\s*(?:[-+*/÷×=]|vs\.?\b)", scoped[m.end():], re.I):
                    continue  # parenthetical arithmetic/comparison, not a bound assertion
                expected_bound = (fact.high if m.group(1).lower() == "upper" else fact.low) if fact else slot.bound
                if expected_bound is not None and abs(float(m.group(2))-expected_bound) > .000001:
                    return check(slot.kind, slot.subject, "fail", "wrong_reference_boundary", expected=expected_bound, evidence=m.group())
        # Explicit incompatible units are errors; omitted units are not fabricated.
        unit_tokens = r"mg/dl|mmol/l|µmol/l|umol/l|ng/ml|pg/ml|ng/dl|g/dl|miu/l|meq/l|kg/m²|kg|cm|lb|inches|bpm|seconds|percent|%"
        unit_aliases = {"seconds": "s", "second": "s", "sec": "s", "percent": "%", "umol/l": "µmol/l"}
        expected_unit = unit_aliases.get(slot.unit.lower().replace("μ", "µ"), slot.unit.lower().replace("μ", "µ"))
        for v, d in matched:
            for m in re.finditer(re.escape(f"{v:.{d}f}") + r"\s*("+unit_tokens+r")\b", scoped, re.I):
                u = m.group(1).lower().replace("μ", "µ")
                u = unit_aliases.get(u, u)
                if expected_unit and slot.kind in {"value", "difference"} and u != expected_unit:
                    return check(slot.kind, slot.subject, "fail", "incompatible_unit", expected=slot.unit, evidence=m.group())
        # Wrong direction attached to a requested difference remains a real error.
        if slot.kind == "difference" and slot.relation:
            for word, direction in (("above", "high"), ("below", "low"), ("higher", "high"), ("lower", "low")):
                for v, d in matched:
                    pattern = re.escape(f"{v:.{d}f}") + r"\s*(?:\S+\s+){0,3}" + word + r"\b"
                    if direction != slot.relation and re.search(pattern, scoped, re.I):
                        return check(slot.kind, slot.subject, "fail", "wrong_difference_direction", expected=slot.expected, evidence=scoped)
        return check(slot.kind, slot.subject, "pass", "computed_from_input", expected=slot.expected, evidence=scoped)
    if candidates:
        return check(slot.kind, slot.subject, "fail", "wrong_requested_number", expected=slot.expected,
                     observed=[v for v, _, _ in candidates], evidence=scoped)
    return check(slot.kind, slot.subject, "review", "requested_number_unresolved", expected=slot.expected, evidence=scoped)


def factual_assertion_checks(text: str, c: Contract) -> list[dict]:
    """Check explicit value and arithmetic assertions, including optional additions.

    Unsupported sentence shapes are not interpreted as false. This supplements,
    rather than replaces, the mandatory review of open clinical assertions.
    """
    checks = []
    for f in c.facts:
        scoped = contexts(text, c.facts, f.name, None)
        for alias in aliases(f.name):
            pattern = (r"(?<!\w)"+re.escape(alias)+r"(?!\w)(?:\s+(?:level|value|count))?"
                       r"\s*(?:is|was|of|at|:|=)\s*(\d+(?:\.\d+)?)")
            for m in re.finditer(pattern, text, re.I):
                before, after = text[max(0,m.start()-55):m.start()], text[m.end():m.end()+70]
                if re.search(r"\b(?:reference|normal|range|threshold|stage)\b[^.;]*$", before, re.I):
                    continue
                if re.match(r"\s*[-–—]\s*\d", after):
                    continue
                if re.match(r"\s*(?:\S+\s+){0,4}(?:above|below|higher|lower|times|fold)\b", after):
                    continue  # a derived value, not an explicit measurement
                if abs(float(m.group(1))-f.value) > .05000001:
                    checks.append(check("extra_claim", f.name, "fail", "wrong_asserted_measurement", expected=f.value, evidence=m.group()))
        # Unrequested differences remain factual claims when explicitly asserted.
        for m in re.finditer(r"\b(?:by)\s*(\d+(?:\.\d+)?)\s*([^,.;]*)", scoped, re.I):
            before = scoped[max(0,m.start()-110):m.start()].lower()
            after = m.group(2).lower()
            b = f.low if "lower limit" in before else f.high if "upper limit" in before else None
            if b is not None and not re.match(r"times|fold|%|percentage|[-+*/÷×]", after):
                if abs(float(m.group(1))-abs(f.value-b)) > .05000001:
                    checks.append(check("extra_claim", f.name, "fail", "wrong_asserted_difference", expected=abs(f.value-b), evidence=m.group()))
        # Explicit local equations can be verified without interpreting prose.
    equation = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?(?:\s*[-+*/÷×]\s*\d+(?:\.\d+)?)+)\s*[=≈]\s*(\d+(?:\.\d+)?)(%)?")
    for m in equation.finditer(text):
        expression, displayed, percent = m.groups()
        if text[max(0,m.start()-1):m.start()] in {"/", "*", "×", "÷", "(", "²", "^"}:
            continue
        def calculate(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (float, int)):
                return node.value
            if isinstance(node, ast.BinOp):
                a, b = calculate(node.left), calculate(node.right)
                if isinstance(node.op, ast.Add): return a+b
                if isinstance(node.op, ast.Sub): return a-b
                if isinstance(node.op, ast.Mult): return a*b
                if isinstance(node.op, ast.Div): return a/b
            raise ValueError("Unsupported arithmetic")
        try:
            expected = calculate(ast.parse(expression.replace("×", "*").replace("÷", "/"), mode="eval").body)
        except (ValueError, SyntaxError, ZeroDivisionError):
            continue
        decimals = len(displayed.split(".")[1]) if "." in displayed else 0
        candidates = [expected, expected*100] if percent else [expected]
        if all(abs(float(displayed)-v) > .5*10**(-decimals)+1e-8 for v in candidates):
            checks.append(check("extra_claim", "equation", "fail", "false_equation", expected=expected, evidence=m.group()))
    return checks


def combine(checks: list[dict]) -> str:
    if any(c["status"] == "fail" for c in checks):
        return "fail"
    if not checks or any(c["status"] == "review" for c in checks):
        return "review"
    return "pass"


def score_v2(record: dict, trajectory: dict, contract: Contract | None = None) -> dict:
    c = contract or compile_contract(record)
    old = legacy.score_example(record, trajectory).as_dict()
    text = clean(trajectory.get("final_answer") or "")
    calls = legacy._all_calls(trajectory)
    behavior = {k: old[k] for k in ("called", "over_call", "tool_selected", "tool_args_correct", "tool_executed", "result_in_answer", "tool_e2e", "unsupported_args")}
    behavior.update({"abstention_expression": strict_abstention(text), "finished_answer": trajectory.get("stop_reason") == "answer",
                     "schema_valid": all(t.get("status") not in {"invalid_json", "unterminated", "schema_error"} for t in trajectory.get("turns", [])),
                     "call_budget_ok": len(calls) <= 1})
    checks = []
    if not text:
        checks.append(check("completion", "answer", "fail", "no_final_answer"))
    elif c.policy in {"uncertain", "missing_tool_inputs"}:
        if calls:
            checks.append(check("abstention", "tool", "fail", "call_despite_missing_inputs"))
        cat = "_and_".join(c.missing_fields)
        fabricated = legacy.fabricated_values(record, cat, text)
        if fabricated:
            checks.append(check("grounding", cat, "fail", "fabricated_value", evidence=fabricated))
        named = all(any(p.search(text) for p in legacy._MISSING_FIELD.get(f, [])) for f in c.missing_fields)
        checks.append(check("abstention", cat, "pass" if strict_abstention(text) and named else "review", "missing_field_and_abstention" if strict_abstention(text) and named else "missing_information_response_unresolved"))
    elif c.policy == "tool":
        checks.append(check("tool_e2e", "tool", "pass" if old["tool_e2e"] else "fail", old["error"]))
        if old["unsupported_args"]:
            checks.append(check("grounding", "tool", "review" if old["tool_args_correct"] else "fail",
                                "argument_and_grounding_verifiers_disagree" if old["tool_args_correct"] else "unsupported_arguments"))
    else:
        checks.extend(evaluate_slot(s, text, c) for s in c.slots)
        checks.extend(factual_assertion_checks(text, c))
        # Explicit false collective statements are independently falsifiable.
        main = {s.subject for s in c.slots if s.kind != "panel_state"}
        others = [f for f in c.facts if f.name not in main and f.state in {"high", "low"}]
        if others and re.search(r"\b(?:all (?:the )?other|other .{0,35}all|remaining)\b.{0,70}\b(?:normal|within .{0,20}range)\b", text, re.I):
            checks.append(check("extra_claim", "other_results", "fail", "false_all_others_normal", evidence=[f.name for f in others]))
        if c.slots and strict_abstention(text) and not any(x["status"] == "pass" for x in checks):
            checks.append(check("answerability", "answer", "fail", "refused_answerable_question"))
    if any(reason.startswith("conflicting_input_facts:") for reason in c.review_reasons):
        checks = [check("input", "record", "review", "conflicting_input_requires_adjudication")]
    core = combine(checks)
    complete_checks = checks + [check("scope", "question", "review", reason) for reason in c.review_reasons]
    status = combine(complete_checks)
    return {"id": record["id"], "answer_type": record["answer_type"], "scorer_version": VERSION,
            "policy": c.policy,
            "status": status, "correct": True if status == "pass" else False if status == "fail" else None,
            "core_status": core, "checks": complete_checks, "behavior": behavior, "legacy": old}
