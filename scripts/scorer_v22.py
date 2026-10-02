"""Scorer v2.2 prototype (not adopted, D-096): frozen v2.1 plus two global contradiction guards on numeric answers.

Kept in scripts/, not src/clinqa/, because evaluation protocol hashes cover every src/clinqa/*.py file.

v2.1 (``scorer_v2.py``, sha256-frozen, untouched) checks only what the question asks, so a false extra claim passes
(docs/SCORER_V2.md:36 already says such claims are errors "when the checker can establish them"; v2.1 implemented this
only inside set checks). v2.2 runs v2.1 first and can only turn a pass into a fail:

1. state guard: every analyte the answer calls abnormal (low, high or out of range) must have that state in the input.
   "Called normal but abnormal" is left to v2.1's set checks: plural and blanket sentences ("X and Y are within
   range") are where v2.1's reader binds states least reliably (gold self-check). A flagged analyte must also be named
   with an explicit state word within 40 characters of the same clause, so cross-clause bindings ("the Wells score is
   expected to be low, and the D-dimer ...") do not count.
   Only explicit state words count (elevated, low, within normal, hyper-/hypo- terms ...): bound-relative
   prepositions ("below the upper limit", "above the 100 mg/dL cutoff") are neutralised first, because they place a
   value against a bound rather than assert a state. Skipped for vitals (conventions differ), blood pressure,
   analytes without a range and self-corrected analytes.
2. amount guard: every "[analyte] ... by X" or "X <unit> above/below the limit/threshold" must equal |value - bound|
   for some input bound (the analyte's range ends or a number stated in the question). Bounds quoted only in the
   answer are not trusted. Percent / fold / times amounts are skipped. To limit binding errors, an amount that is the
   correct deviation of any analyte in the input is accepted. Division and multiplication ("dividing 4273.7 by 500"),
   equation operands ("60 - 51.2 = 8.8"), parenthesised restatements in another unit ("(or 12 g/L)"), fold amounts ("2.8 times above") and blood pressure are skipped.

Guards apply to numeric_reasoning only; other answer types are scored exactly as v2.1.
"""

from __future__ import annotations

import re
from typing import Any

from clinqa import scorer_v2 as v21

VERSION = "2.2"

_BY = re.compile(r"\bby\s+(?:only\s+|just\s+|approximately\s+|about\s+|roughly\s+|nearly\s+|almost\s+|~\s*)?"
                 r"(\d+(?:\.\d+)?)(?!\s*(?:%|percent|-?fold|times|x\b)|\.\d|\d)", re.I)
_REL = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)(?!\s*%)(?:\s+(?!above|below|over|under|higher|lower|short|beyond)[^\s\d,;]+){0,3}"
                  r"\s+(?:above|below|over|under|higher than|lower than|short of|beyond)\s+(?:the\s+|its\s+|their\s+)?"
                  r"(?:upper|lower|normal|reference|threshold|limit|cutoff|cut-off|ULN|LLN)", re.I)
_SKIP_STATE = set(v21._VITAL_RANGES) | {"Blood Pressure"}
_RELATIVE = re.compile(r"\b(?:above|below|over|under|exceed\w*|surpass\w*|higher than|lower than|less than|greater than|"
                       r"short of|beyond)(?=\s+(?:the\s+|its\s+|their\s+|his\s+|her\s+|a\s+)?(?:(?:stage|desirable|"
                       r"normal|reference|respective|recommended|diagnostic|goal)\s+)?(?:upper|lower|limit|threshold|cutoff|"
                       r"cut-off|bound|ULN|LLN|\d)|\s*[,;.)\u2014\u2013-]|\s*$)", re.I)
_LIMIT_OF_NORMAL = re.compile(r"\b(?:upper |lower )?limits? of normal\b", re.I)
_OPERAND = re.compile(r"\s*[-\u2212\u2013+\u00d7x*/\u00f7]\s*\d")  # "60 - 51.2 = 8.8": an equation operand
_FOLD = re.compile(r"\s*(?:-?fold|times|\u00d7|x\b)", re.I)  # "2.8 times above the upper limit": a ratio
_RESTATED = re.compile(r"\(\s*(?:or|i\.e\.,?|=|approximately|about|~)?\s*$", re.I)  # "1.2 g/dL (or 12 g/L)": other unit
_ARITH = re.compile(r"\b(?:divid\w*|multipl\w*|ratio)\b[^;]{0,40}$", re.I)


def _facts(record: dict[str, Any]) -> dict[str, v21.Fact]:
    return v21.note_facts(record, v21.question_facts(record, v21.table_facts(record)))


_HIGH_WORD = r"(?:elevat\w*|high|raised|increased|abnormal\w*|outside)"
_LOW_WORD = r"(?:low|decreased|reduced|deficien\w*|depressed|abnormal\w*|outside)"


def _local_claim(text: str, name: str, said: frozenset[str]) -> bool:
    """The analyte is named and an explicit state word follows within 40 characters of the same clause."""
    word = _HIGH_WORD if said == {"high"} else _LOW_WORD if said == {"low"} else f"(?:{_HIGH_WORD}|{_LOW_WORD})"
    for alias in v21._ALIASES.get(name, [name]):
        flags = 0 if v21._CASE_SENSITIVE.match(alias) else re.I
        if re.search(rf"(?<![\w-]){re.escape(alias)}(?![\w-])[^.;,\u2014]{{0,40}}?\b{word}\b", text, flags):
            return True
    return False


def state_guard(record: dict[str, Any], reading: v21.Reading, text: str = "") -> list[str]:
    facts = _facts(record)
    out = []
    for name, said in reading.states.items():
        f = facts.get(name)
        if not said <= {"low", "high"} or name in reading.conflicts or name in _SKIP_STATE or not f or f.bp:
            continue
        true = f.state()
        if true is not None and true not in said and _local_claim(text, name, said):
            out.append(f"state({name}: said {'/'.join(sorted(said))}, input {true})")
    return out


def _deviations(f: v21.Fact, bounds: list[float]) -> list[float]:
    if f.value is None or f.bp:
        return []
    return [abs(f.value - b) for b in bounds]


def amount_guard(record: dict[str, Any], final: str, names: list[str]) -> list[str]:
    facts = _facts(record)
    qnums = [v21._num(m.group(0)) for m in v21.NUM_RE.finditer(record["question"])]
    bounds = {n: [b for b in (f.low, f.high) if b is not None] + qnums for n, f in facts.items()}
    anywhere = [d for n, f in facts.items() for d in _deviations(f, bounds[n])]
    out = []
    subject: str | None = None
    for sent in v21.sentences(v21.normalize(final)):
        ms = v21.mentions(sent, names)
        for m in list(_BY.finditer(sent)) + list(_REL.finditer(sent)):
            x = float(m.group(1))
            prev = [n for s, _, n in ms if s < m.start()]
            owner = prev[-1] if prev else subject
            f = facts.get(owner) if owner else None
            if m.re is _BY and _ARITH.search(sent[:m.start()]):
                continue
            if _OPERAND.match(sent, m.end(1)) or _FOLD.match(sent, m.end(1)) or _RESTATED.search(sent[:m.start(1)]):
                continue
            if f is None or owner == "Blood Pressure" or f.value is None or f.bp or abs(x - f.value) < 1e-9:
                continue
            ok = any(lo <= x <= hi for d in _deviations(f, bounds[owner]) for lo, hi in [v21._band(d, "diff")])
            if not ok and not any(lo <= x <= hi for d in anywhere for lo, hi in [v21._band(d, "diff")]):
                want = ", ".join(f"{d:g}" for d in sorted(set(round(d, 4) for d in _deviations(f, bounds[owner][:2]))))
                out.append(f"amount({owner}: said {x:g}, input deviation {want or '?'})")
        if ms:
            subject = ms[-1][2]
    return list(dict.fromkeys(out))


def score(record: dict[str, Any], traj: dict[str, Any], key: v21.Key | None = None) -> dict[str, Any]:
    key = key or v21.build_key(record)
    out = {**v21.score(record, traj, key), "scorer": VERSION}
    if record["answer_type"] != "numeric_reasoning":
        return out
    final = traj.get("final_answer") or ""
    if not final.strip():
        return out
    text = _LIMIT_OF_NORMAL.sub("limit", _RELATIVE.sub("vs", v21.normalize(final)))
    guards = state_guard(record, v21.read(text, key.names), text) + amount_guard(record, final, key.names)
    out["v21_correct"] = out["correct"]
    out["guards"] = guards
    if out["correct"] and guards:
        out["correct"] = False
        out["error"] = "guard:" + guards[0]
    return out


build_key = v21.build_key
gold_trajectory = v21.gold_trajectory
