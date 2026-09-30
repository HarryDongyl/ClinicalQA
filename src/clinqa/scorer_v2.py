"""Scorer v2: MedCalc-Bench-style answer keys with type-specific checks.

Design (docs/SCORER_V2.md):
  1. build_key(record) compiles an answer key from the question and the INPUT (table, note), never
     from a prediction. Like MedCalc-Bench, each check carries a ground truth and, for computed
     numbers, a [lower, upper] tolerance band. The gold answer is only a cross-check (gold_agrees).
  2. score(record, trajectory, key) runs every check against the final answer and returns
     pass/fail per check; the example passes iff all checks pass. There is no "review" outcome.

Check kinds and tolerance policy (MedCalc: exact for integer/categorical, band for decimals):
  value   table/vital lookup                 exact numeric equality
  status  low | normal | high of an analyte  categorical, negation-aware, bound to the analyte
  diff    distance to a bound/threshold      exact for integer operands, else +-max(0.051, 0.5%)
  ratio   value / upper limit                +-5% relative (floor 0.051)
  pct     % above/below a bound              +-5% relative (floor 0.51)
  bmi     BMI from note measurements         +-0.15 absolute
  entity  "which value is most abnormal"     any of an accepted set (BFCL possible_answer style)
  set     "which other / are any ..."        required members asserted, no false members
  yesno   threshold / "is this the only"     categorical
  text    free-text facts from the note      DROP-style number gate + grounded key-term recall
Tool calls are scored on the executed outcome (tau-bench style); uncertain and Q5 examples use
deterministic abstention gates (AbstentionBench keyword layer) plus a no-unsupported-call gate.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from clinqa import metrics as v1
from clinqa.parsing import extract_inline_bmi, parse_bp, parse_float, parse_ref_range, record_body_measurements, to_metric
from clinqa.tools import execute_tool

VERSION = "2.1"
NUM_RE = re.compile(r"(?<![\d.])(?<![A-WYZa-wyz])[-−]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")
SENT_RE = re.compile(r"(?<=[.!?:])\s+(?=[A-Z(*\-•])|\s*\n+\s*|\s{2,}(?=[A-Z\-•])")
STATES = frozenset({"low", "normal", "high"})

# ---------------------------------------------------------------- analytes

# alias -> canonical table name; longest aliases are matched first and mask shorter ones.
_ALIASES: dict[str, list[str]] = {
    "Glucose (fasting)": ["fasting glucose", "fasting blood glucose", "glucose (fasting)", "blood glucose", "glucose",
                          "FBG"],
    "Fasting Glucose": ["fasting glucose", "glucose"],
    "HbA1c": ["hemoglobin a1c", "haemoglobin a1c", "hba1c", "a1c"],
    "Creatinine": ["serum creatinine", "creatinine", "Cr"],
    "BUN": ["blood urea nitrogen", "urea nitrogen", "BUN"],
    "Sodium": ["sodium", "Na"],
    "Potassium": ["potassium", "K"],
    "Total Cholesterol": ["total cholesterol", "cholesterol"],
    "LDL": ["ldl cholesterol", "ldl-c", "ldl"],
    "HDL": ["hdl cholesterol", "hdl-c", "hdl"],
    "Triglycerides": ["triglycerides", "triglyceride", "TG"],
    "WBC": ["white blood cell count", "white blood cells", "white blood cell", "white count", "wbc", "leukocyte count"],
    "Hemoglobin": ["hemoglobin", "haemoglobin", "Hgb", "Hb"],
    "Hematocrit": ["hematocrit", "haematocrit", "Hct"],
    "Platelets": ["platelet count", "platelets", "platelet", "PLT"],
    "ALT": ["alanine aminotransferase", "ALT"],
    "AST": ["aspartate aminotransferase", "AST"],
    "Alkaline Phosphatase": ["alkaline phosphatase", "alk phos", "ALP"],
    "Total Bilirubin": ["total bilirubin", "bilirubin", "T bili"],
    "Albumin": ["serum albumin", "albumin"],
    "Total Protein": ["total protein", "protein"],
    "TSH": ["thyroid-stimulating hormone", "thyroid stimulating hormone", "TSH"],
    "Free T4": ["free t4", "free thyroxine", "FT4", "T4"],
    "Free T3": ["free t3", "free triiodothyronine", "FT3", "T3"],
    "PT": ["prothrombin time", "PT"],
    "INR": ["INR"],
    "aPTT": ["activated partial thromboplastin time", "partial thromboplastin time", "aptt", "PTT"],
    "D-dimer": ["d-dimer", "d dimer", "ddimer"],
    "Serum Iron": ["serum iron", "iron level", "iron"],
    "TIBC": ["total iron-binding capacity", "total iron binding capacity", "TIBC"],
    "Ferritin": ["serum ferritin", "ferritin"],
    "Transferrin Saturation": ["transferrin saturation", "iron saturation", "TSAT", "transferrin sat"],
    "eGFR": ["egfr", "gfr", "glomerular filtration rate"],
    "Uric Acid": ["uric acid", "urate"],
    "Calcium": ["serum calcium", "calcium", "Ca"],
    "Phosphorus": ["phosphorus", "phosphate"],
    "Blood Pressure": ["blood pressure", "BP", "systolic", "diastolic"],
    "Heart Rate": ["resting heart rate", "heart rate", "pulse", "HR"],
    "Temperature": ["body temperature", "temperature", "temp"],
    "Respiratory Rate": ["respiratory rate", "respiration rate", "RR"],
    "SpO2": ["oxygen saturation", "spo2", "o2 saturation", "o2 sat", "pulse oximetry", "SaO2"],
}
# Clinical terms that name an analyte and its state at once.
_STATE_TERMS: list[tuple[str, str, str]] = [
    (r"hyperkal\w*", "Potassium", "high"), (r"hypokal\w*", "Potassium", "low"),
    (r"hypernatr\w*", "Sodium", "high"), (r"hyponatr\w*", "Sodium", "low"),
    (r"hyperglyc\w*", "Glucose (fasting)", "high"), (r"hypoglyc\w*", "Glucose (fasting)", "low"),
    (r"hypercalc\w*", "Calcium", "high"), (r"hypocalc\w*", "Calcium", "low"),
    (r"hyperphosphat\w*", "Phosphorus", "high"), (r"hypophosphat\w*", "Phosphorus", "low"),
    (r"hyperuric\w*", "Uric Acid", "high"), (r"hypoalbumin\w*", "Albumin", "low"),
    (r"hyperbilirubin\w*", "Total Bilirubin", "high"), (r"hypertriglycerid\w*", "Triglycerides", "high"),
    (r"thrombocytopeni\w*", "Platelets", "low"), (r"thrombocytos\w*", "Platelets", "high"),
    (r"leukocytos\w*", "WBC", "high"), (r"leukopeni\w*", "WBC", "low"),
    (r"tachycard\w*", "Heart Rate", "high"), (r"bradycard\w*", "Heart Rate", "low"),
    (r"febrile|fever\w*|pyrex\w*", "Temperature", "high"), (r"afebrile", "Temperature", "normal"),
    (r"tachypn\w*", "Respiratory Rate", "high"), (r"hypoxemi\w*|hypoxi\w*|desaturat\w*", "SpO2", "low"),
    (r"hypertensi\w*", "Blood Pressure", "high"), (r"hypotensi\w*", "Blood Pressure", "low"),
    (r"normotensi\w*", "Blood Pressure", "normal"),
]
_CASE_SENSITIVE = re.compile(r"^[A-Z0-9]{1,4}$|^[A-Z][a-z]$")  # PT, K, Na, Hb, T4 ...


def _alias_patterns(names: list[str]) -> list[tuple[re.Pattern[str], str]]:
    pats = []
    for name in names:
        for a in _ALIASES.get(name, [name.lower()]):
            flags = 0 if _CASE_SENSITIVE.match(a) else re.I
            pats.append((len(a), re.compile(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])", flags), name))
    pats.sort(key=lambda p: -p[0])
    return [(p, n) for _, p, n in pats]


_MODIFIER_USE = re.compile(r"\s*(?:-to-|/)|\s+(?:production|clearance|excretion|synthesis|kinase|ratio|supplement\w*|"
                           r"therapy|replacement|infusion|deficiency|binding|metabolism|stones?|carbonate|citrate|"
                           r"gluconate|sulfate|chloride|channel|antagonist|agonist|inhibitor|monitoring|trend)\b", re.I)
_PANEL_WORD = re.compile(r"\s*(?:stud(?:y|ies)|panel|parameters|indices|profile|tests?|function)\b", re.I)


def mentions(text: str, names: list[str]) -> list[tuple[int, int, str]]:
    """Non-overlapping analyte mentions (start, end, canonical name), longest alias first."""
    taken = [False] * len(text)
    out = []
    for pat, name in _alias_patterns(names):
        for m in pat.finditer(text):
            if any(taken[m.start():m.end()]):
                continue
            if name == "Serum Iron" and _PANEL_WORD.match(text, m.end()):
                continue  # "iron studies" names the panel, not the analyte
            if _MODIFIER_USE.match(text, m.end()):
                continue  # "creatinine production", "BUN-to-creatinine ratio": not the analyte's value
            for i in range(m.start(), m.end()):
                taken[i] = True
            out.append((m.start(), m.end(), name))
    return sorted(out)


# ---------------------------------------------------------------- input facts


@dataclass
class Fact:
    name: str
    value: float | None
    raw: str
    unit: str
    low: float | None = None
    high: float | None = None
    low_strict: bool = False  # ">40": value must be > 40
    high_strict: bool = False  # "<200": value must be < 200
    bp: tuple[float, float] | None = None

    def state(self, value: float | None = None) -> str | None:
        v = self.value if value is None else value
        if v is None or (self.low is None and self.high is None):
            return None
        if self.high is not None and (v >= self.high if self.high_strict else v > self.high):
            return "high"
        if self.low is not None and (v <= self.low if self.low_strict else v < self.low):
            return "low"
        return "normal"


# Conventional adult resting thresholds for vitals (the table gives no range). Used only when the
# question names no threshold itself; status keys for vitals also accept the gold's stated state.
_VITAL_RANGES = {"Heart Rate": (60.0, 100.0), "Respiratory Rate": (12.0, 20.0), "SpO2": (95.0, None),
                 "Temperature": (97.0, 100.3)}


def table_facts(record: dict[str, Any]) -> dict[str, Fact]:
    facts: dict[str, Fact] = {}
    for row in record["table"]["rows"]:
        name, raw = row[0], row[1]
        unit = row[2] if len(row) > 2 else ""
        f = Fact(name, parse_float(raw), raw, unit)
        if name == "Blood Pressure":
            f.bp = parse_bp(raw)
            f.value = f.bp[0] if f.bp else None
        if len(row) > 3 and (rng := parse_ref_range(row[3])):
            f.low, f.high = rng["low"], rng["high"]
            f.low_strict = row[3].strip().startswith(">") and not row[3].strip().startswith(">=")
            f.high_strict = row[3].strip().startswith("<") and not row[3].strip().startswith("<=")
        elif name in _VITAL_RANGES:
            f.low, f.high = _VITAL_RANGES[name]
        facts[name] = f
    return facts


# ---------------------------------------------------------------- state assertions in text

_NOT_STATE = re.compile(
    r"(?:\s*-?\s*|\s+)(?:limit|bound|end|reference|threshold|range|risk|dose|doses|dosing|intensity|suspicion|"
    r"likelihood|probability|concern|index|priority|yield|normal limit|muscle|mass|intake|"
    r"output|volume|body|side|normal range|portion|half|part)\b|\s+(?:the\s+|its\s+)?(?:midpoint|mean|average|median|"
    r"baseline|prior|previous|middle)\b", re.I)
_STATE_PATTERNS: list[tuple[re.Pattern[str], frozenset[str]]] = [
    (re.compile(r"\b(?:outside|out of|beyond) (?:of )?(?:the |its |their |his |her )?(?:normal |reference |expected |"
                r"respective )*(?:range|ranges|limits?|interval)\b|\bout of range\b|\babnormal(?:ly)?\b|\bderanged\b",
                re.I), frozenset({"low", "high"})),
    (re.compile(r"\b(?:above|over|exceed\w*|greater than|higher than|at or above) (?:the |its )?lower (?:limit|bound|"
                r"end|threshold|reference)", re.I), frozenset({"normal", "high"})),
    (re.compile(r"\b(?:below|under|less than|lower than|at or below) (?:the |its )?upper (?:limit|bound|end|threshold|"
                r"reference)|\b(?:does|did|do) not exceed\b|\bdoesn[’']t exceed\b", re.I), frozenset({"low", "normal"})),
    (re.compile(r"\b(?:above|over|exceed\w*|higher than|greater than) (?:the )?normal\b(?! range| limit)", re.I),
     frozenset({"high"})),
    (re.compile(r"\b(?:below|under|less than|lower than) (?:the )?normal\b(?! range| limit)", re.I), frozenset({"low"})),
    (re.compile(r"\b(?:within|in) (?:the |its |their |his |her |a )?(?:normal|reference|expected|target|desirable|"
                r"optimal|acceptable|healthy|goal|respective)(?: reference| normal)?(?: range| ranges| limits?| "
                r"interval)?\b|\bnormal limits\b|\bWNL\b|\bunremarkable\b|\bin range\b|\bwithin range\b"
                r"|\b(?:is|are|was|were|remains?|remained|appears?|be|considered|otherwise|all|both|still|entirely|"
                r"completely|also) (?:\w+ )?normal\b(?! (?:range|reference|limit|upper|lower|threshold|value))"
                r"|(?<!of )(?<!of the )\bnormal\s*(?=[.,;)]|$)|\bwithin\s*\(?\s*(?:ref\w*\s*:?\s*)?[<>≤≥]?\s*\d", re.I),
     frozenset({"normal"})),
    (re.compile(r"\b(?:elevat\w*|raised|increased|high|higher|above|exceed\w*|greater than|surpass\w*|"
                r"supranormal)\b", re.I), frozenset({"high"})),
    (re.compile(r"\b(?:low|lower|below|decreased|reduced|depressed|diminished|subnormal|short of|"
                r"falls? short)\b", re.I), frozenset({"low"})),
]
_NEGATOR = re.compile(r"\b(?:not|no|never|without|neither|nor|none)\b(?:\s+\w+){0,2}\s*$|n[’']t(?:\s+\w+){0,2}\s*$",
                      re.I)
_PLURAL = re.compile(r"\b(?:are|were|all|both|remain|each|respective|their|these|those|others)\b[^.;]{0,40}$", re.I)
_STATE_TERM_RES = [(re.compile(r"\b(?:" + p + r")\b", re.I), a, s) for p, a, s in _STATE_TERMS]


def _state_phrases(sentence: str) -> list[tuple[int, int, frozenset[str]]]:
    taken = [False] * len(sentence)
    out = []
    for pat, states in _STATE_PATTERNS:
        for m in pat.finditer(sentence):
            if any(taken[m.start():m.end()]) or _NOT_STATE.match(sentence, m.end()):
                continue
            if states == frozenset({"high"}) and re.match(r"\s+than\s+(?:the\s+)?lower", sentence[m.end():], re.I):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            s = states
            if _NEGATOR.search(sentence[max(0, m.start() - 30):m.start()]):
                s = STATES - states if states != frozenset({"low", "high"}) else frozenset({"normal"})
            out.append((m.start(), m.end(), frozenset(s)))
    return sorted(out)


_HYPOTHETICAL = re.compile(r"\bunlikely\b|\brul(?:e|ed|ing) out\b|\bno evidence\b|\bnot consistent\b|\bwould\b|"
                           r"\bif (?:the|it|this)\b|\bmay be\b|\bcould be\b|\brisk of\b|\bconcern for\b", re.I)


def normalize(text: str) -> str:
    return re.sub(r"\*\*|__|`|^\s*[-•*]\s+", "", text or "", flags=re.M)


def sentences(text: str) -> list[str]:
    return [s for s in SENT_RE.split(text or "") if s and s.strip()]


@dataclass
class Reading:
    """What a text asserts: per analyte, the intersection of asserted state sets, plus numbers per analyte."""

    states: dict[str, frozenset[str]] = field(default_factory=dict)
    conflicts: set[str] = field(default_factory=set)
    numbers: dict[str, list[float]] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)  # analytes in order of first mention
    most: list[str] = field(default_factory=list)  # analytes in sentences claiming "most ..."
    blanket: bool = False  # "all (other) values are within range" / "none of the others is abnormal"

    def assert_state(self, name: str, s: frozenset[str]) -> None:
        cur = self.states.get(name, STATES)
        new = cur & s
        if not new:
            self.conflicts.add(name)
            new = s  # keep the latest assertion but remember the contradiction
        self.states[name] = new


_MOST = re.compile(r"\bmost\b|\bclosest\b|\bnearest\b|\bgreatest\b|\blargest\b|\bhighest (?:degree|relative|proportional)|\bfurthest\b|"
                   r"\bmore (?:significantly|markedly|severely|substantially|profoundly|abnormal|elevated|deranged)\b|"
                   r"\b(?:greater|larger) (?:deviation|elevation|degree)\b", re.I)
_BLANKET = re.compile(r"\ball\b[^;]{0,200}?\b(?:are|were|fall|falls|remain|remained|lie|is)?\s*(?:\w+\s+){0,2}"
                      r"(?:within|normal\b|unremarkable)|\bnone of\b[^;]{0,160}?\b(?:exceed|fall|outside|abnormal|"
                      r"elevated|below|above|are)|\b(?:no|none) other\b[^.;]{0,40}\b(?:abnormal|outside|elevated)|"
                      r"\bis the only\b|\bare the only\b|\bonly (?:abnormal|lab|value|result)|"
                      r"\b(?:values|labs|tests|results|studies|parameters)\b[^.;]{0,40}\b(?:are|were|fall|remain)\s+(?:all\s+)?"
                      r"(?:within|normal)|\bno (?:\w+ ){0,3}(?:values?|labs?|results?|tests?|parameters?) (?:are|were|is|was|fall)"
                      r" (?:outside|abnormal|elevated)", re.I)
_PREMOD = re.compile(r"\b(normal|elevated|high|low|reduced|decreased|increased|raised|abnormal)\s+(?:\w+\s+){0,2}$",
                     re.I)
_PREMOD_STATE = {"normal": frozenset({"normal"}), "elevated": frozenset({"high"}), "high": frozenset({"high"}),
                 "raised": frozenset({"high"}), "increased": frozenset({"high"}), "low": frozenset({"low"}),
                 "reduced": frozenset({"low"}), "decreased": frozenset({"low"}), "abnormal": frozenset({"low", "high"})}
_PRONOUN = re.compile(r"\s*(?:(?:yes|no)\s*,?\s*)?(?:this|it|that|these|which|the (?:value|level|result|latter|former))\b",
                      re.I)


def read(text: str, names: list[str]) -> Reading:
    text = normalize(text)
    r = Reading()
    subject: list[str] = []
    for sent in sentences(text):
        ms = mentions(sent, names)
        for _, _, n in ms:
            if n not in r.order:
                r.order.append(n)
        if _MOST.search(sent) and (ms or subject):
            mpos = _MOST.search(sent).start()
            before = [n for s, _, n in ms if s < mpos]
            after = [n for s, _, n in ms if s > mpos]
            pronoun = _PRONOUN.match(sent)
            true_pronoun = re.match(r"\s*(?:(?:yes|no)\s*,?\s*)?(?:this|it|that|these|which|such)\b", sent, re.I)
            r.most += before[-1:] or (subject[-1:] if true_pronoun and subject else after[:1])
        # numbers belong to the most recent mention before them (else the first mention / inherited subject)
        for m in NUM_RE.finditer(sent):
            prev = [n for s, _, n in ms if s <= m.start()]
            if prev:
                owner = prev[-1]
            elif subject and (_PRONOUN.match(sent) or not ms):
                owner = subject[-1]
            else:
                owner = ms[0][2] if ms else None
            if owner:
                r.numbers.setdefault(owner, []).append(_num(m.group(0)))
        for p, a, s in [(pat, a, s) for pat, a, s in _STATE_TERM_RES]:
            for m in p.finditer(sent):
                if a in names and not _NEGATOR.search(sent[max(0, m.start() - 30):m.start()]):
                    r.assert_state(a, frozenset({s}))
        bound_upto = -1
        bm = _BLANKET.search(sent)
        if bm and not _HYPOTHETICAL.search(sent[:bm.end()]):
            r.blanket = True
            only = re.search(r"\bonly\b", sent, re.I)
            for s0, _, n in ms:
                if not only and s0 >= bm.start():
                    r.assert_state(n, frozenset({"normal"}))
            if not only:
                subject = [n for _, _, n in ms] or subject
                continue
        for s0, _, n in ms:
            pm = _PREMOD.search(sent[max(0, s0 - 30):s0])
            if pm and not _NEGATOR.search(sent[max(0, s0 - 60):s0 - len(pm.group(0))]):
                r.assert_state(n, _PREMOD_STATE[pm.group(1).lower()])
        for ps, pe, st in _state_phrases(sent):
            lo_ = max(sent.rfind(",", 0, ps), sent.rfind(";", 0, ps), 0)
            hi_ = min([x for x in (sent.find(",", pe), sent.find(";", pe)) if x >= 0] or [len(sent)])
            if _HYPOTHETICAL.search(sent[lo_:hi_]):
                continue
            prev = [(s, n) for s, _, n in ms if s < ps]
            if _PLURAL.search(sent[max(0, ps - 60):ps]) and prev:
                targets = [n for s, n in prev if s > bound_upto] or ([prev[-1][1]] if bound_upto < 0 else [])
            else:
                nxt = [(s, n) for s, _, n in ms if pe <= s <= pe + 25 and "," not in sent[pe:s]
                       and len(sent[pe:s].split()) <= 3]
                adjective = re.match(r"(?:elevat|raised|increased|high|low|decreased|reduced|abnormal|normal)",
                                     sent[ps:pe], re.I)
                if adjective and nxt:
                    targets = [nxt[0][1]]  # "markedly reduced eGFR"
                elif prev:
                    targets = [prev[-1][1]]
                elif subject and _PRONOUN.match(sent):
                    targets = [subject[-1]]
                elif (later := [(s_, n) for s_, _, n in ms if pe <= s_ <= pe + 80 and ";" not in sent[pe:s_]]):
                    targets = [later[0][1]]  # "another value that exceeds its range is phosphorus"
                else:
                    targets = []
            for t in targets:
                r.assert_state(t, st)
            bound_upto = ps
        if ms:
            subject = [n for _, _, n in ms]
    return r


def _num(s: str) -> float:
    s = s.replace("−", "-").replace(",", "")
    return float(s.split("/")[0])


def yes_no(text: str) -> bool | None:
    m = re.match(r"\s*(yes|no)\b", text or "", re.I)
    return None if not m else m.group(1).lower() == "yes"


# ---------------------------------------------------------------- answer keys


@dataclass
class Check:
    kind: str
    subject: str | None
    gt: Any
    lo: float | None = None
    hi: float | None = None
    accept: list[Any] = field(default_factory=list)  # extra acceptable answers (entity / status)
    polarity: str = "abnormal"  # set checks: abnormal | high | low | normal
    exclude: list[str] = field(default_factory=list)
    values: dict[str, float] = field(default_factory=dict)  # entity: per-candidate deviation (value / bound)
    gold_agrees: bool | None = None
    note: str = ""


@dataclass
class Key:
    id: str
    answer_type: str
    expected: str  # answer | call | abstain
    checks: list[Check]
    names: list[str]
    coverage: str  # structured | partial | text

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_CLAUSE_SPLIT = re.compile(r"\?\s*|;\s*|,?\s+and\s+(?=(?:which|what|is|are|does|do|did|by|how|if|whether|can|could|"
                           r"has|have|was|were)\b)|(?<=\.)\s+(?=[A-Z])", re.I)
_THRESH = re.compile(r"(?:threshold|limit|cutoff|cut-off|target|goal|of)(?:[^\d]|(?<=[A-Za-z])\d){0,40}?\(?\s*(?:≥|>=|>|<|≤|<=)?\s*"
                     r"(?<![\w.])(?<!stage )(?<!class )(?<!grade )(?<!type )(\d+(?:\.\d+)?)(?:\s*[-–—]\s*(?P<hi>\d+(?:\.\d+)?))?(?:\s*/\s*(\d+(?:\.\d+)?))?\s*(%|bpm|mmHg|mg/dL|kg/m|°F|breaths)?", re.I)
_OP = {
    "pp": re.compile(r"percentage points?", re.I),
    "ratio": re.compile(r"how many times|\btimes (?:the|its|their|higher|greater|above|over|as)|\bfold\b|multiple of",
                        re.I),
    "pct": re.compile(r"(?:what|which|how much|how many|by what)\W+(?:\w+\W+){0,2}percent(?:age)?\b(?! points?)|"
                      r"\bpercent(?:age)? (?:above|below|higher|lower|over|greater)\b|in percentage terms", re.I),
    "diff": re.compile(r"\bby how (?:much|many|far)\b|how (?:much|far) (?:higher|lower|above|below|greater|less|over|"
                       r"under)|how many \S+(?: \S+)? (?:above|below|over|under|higher|lower|greater|less|exceed|"
                       r"short)|\bdifference\b|how (?:far|much) (?:is|does|do)\b", re.I),
    "entity": re.compile(r"\b(?:which|what)\b[^?]{0,70}\b(?:most|greatest|largest|furthest|highest|greater|"
                         r"larger|further|more (?:significant|abnormal|elevated|severe))\b|\bhow does (?:this|it) compare\b|"
                         r"\bcompared? (?:to|with)\b[^?]{0,40}\b(?:elevation|deviation)", re.I),
    "set": re.compile(r"\b(?:which|what|any) other\b|\bis any\b|\bany (?:\w+ ){0,3}(?:outside|abnormal|elevated)\b|\bis (?:this|it|that) the only\b|\bthe only\b|"
                      r"\bare (?:any|all|there any)\b|\bwhich (?:\w+ ){0,3}(?:values?|tests?|parameters?|labs?|results?|"
                      r"studies|markers?|components?)\b|\bwhich of the (?:patient's )?\b|\bany of the\b", re.I),
    "status": re.compile(r"\b(?:within|outside|in) (?:the |its |their |a )?(?:normal|reference|expected|target)"
                         r"|\bis (?:it|this|that|the value) (?:normal|elevated|abnormal|high|low)\b"
                         r"|\bnormal\s*(?:range)?\s*$|\b(?:elevated|abnormal|high|low|normal)\s*$"
                         r"|\bfall within\b|\b(?:above|below|elevated|low|high|abnormal)\b[^?]{0,25}\b(?:reference|normal) "
                         r"range|\bconsidered (?:normal|abnormal|high|low)",
                         re.I),
    "yesno_thr": re.compile(r"^\s*(?:is|are|does|do)\b[^?]{0,80}\b(?:above|below|exceed\w*|meet|meets|over|under)\b"
                            r"[^?]{0,40}\bthreshold\b", re.I),
    "value": re.compile(r"\bwhat (?:is|was|are|were)\b|\bwhat (?:\w+ ){0,3}(?:value|level|reading|count)\b|"
                        r"\breport\b|\bstate\b", re.I),
    "bmi": re.compile(r"\bBMI\b|body mass index", re.I),
}


_STATED = re.compile(r"(?:\s+(?:level|value|count|reading))?\s+(?:is|was|of|at|=|:)\s*(?:recorded at\s*|measured at\s*)?"
                     r"(\d+(?:\.\d+)?(?:\s*/\s*\d+)?)")


def question_facts(record: dict[str, Any], facts: dict[str, Fact]) -> dict[str, Fact]:
    """Values stated explicitly in the question ("The patient's heart rate is 111 bpm") are input facts too."""
    out = dict(facts)
    q = record["question"]
    for s_, e, name in mentions(q, list(_ALIASES)):
        if name in out:
            continue
        m = _STATED.match(q, e)
        if not m:
            continue
        raw = m.group(1).replace(" ", "")
        f = Fact(name, parse_float(raw.split("/")[0]), raw, "")
        if "/" in raw:
            f.bp = parse_bp(raw)
        if name in _VITAL_RANGES:
            f.low, f.high = _VITAL_RANGES[name]
        rng = re.search(r"(?:reference|normal) range (?:of |is |\()?\s*(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)", q, re.I)
        if rng:
            f.low, f.high = float(rng.group(1)), float(rng.group(2))
        out[name] = f
    return out


# Reference ranges used by the dataset's lab tables; applied to labs that appear only in the note.
_CANON_RANGES = {"Creatinine": "0.7-1.3", "BUN": "7-20", "eGFR": ">60", "Uric Acid": "3.5-7.2", "Calcium": "8.5-10.5",
                 "Phosphorus": "2.5-4.5", "TSH": "0.27-4.20", "Free T4": "0.93-1.70", "Free T3": "2.0-4.4",
                 "Serum Iron": "60-170", "TIBC": "250-370", "Ferritin": "12-300", "Transferrin Saturation": "20-50",
                 "PT": "11.0-13.5", "INR": "0.8-1.1", "aPTT": "25-35", "D-dimer": "<500", "ALT": "7-56",
                 "AST": "10-40", "Alkaline Phosphatase": "44-147", "Total Bilirubin": "0.1-1.2", "Albumin": "3.5-5.5",
                 "Total Protein": "6.0-8.3", "WBC": "4.5-11.0", "Hemoglobin": "12.0-17.5", "Hematocrit": "36-51",
                 "Platelets": "150-400", "Total Cholesterol": "<200", "LDL": "<100", "HDL": ">40",
                 "Triglycerides": "<150", "HbA1c": "<5.7", "Glucose (fasting)": "70-100", "Sodium": "136-145",
                 "Potassium": "3.5-5.0"}
_NOTE_VALUE = re.compile(r"\s*(?:(?:level|value|count|was|is|of|at|measured|recorded|reading|elevated at|low at)\s+|"
                         r"[:=(]\s*|[-–]\s+){0,3}(\d+(?:\.\d+)?(?:\s*/\s*\d+)?)(?!\s*(?:years?|y/?o|days?|weeks?|months?|"
                         r"hours?|mg\b(?!/)|mcg|units?|times|x\b))", re.I)


def note_facts(record: dict[str, Any], facts: dict[str, Fact]) -> dict[str, Fact]:
    """Labs/vitals written in the note when the table does not hold them. Ambiguous (several values) -> skipped."""
    out = dict(facts)
    note = record["note"]
    found: dict[str, set[str]] = {}
    for _, e, name in mentions(note, list(_ALIASES)):
        if name in facts or name in ("Blood Pressure",) and "Blood Pressure" in facts:
            continue
        m = _NOTE_VALUE.match(note, e)
        if m:
            found.setdefault(name, set()).add(m.group(1).replace(" ", ""))
    for name, vals in found.items():
        if len(vals) != 1:
            continue
        raw = next(iter(vals))
        f = Fact(name, parse_float(raw.split("/")[0]), raw, "")
        if "/" in raw:
            if name != "Blood Pressure":
                continue
            f.bp = parse_bp(raw)
        if name in _VITAL_RANGES:
            f.low, f.high = _VITAL_RANGES[name]
        elif name in _CANON_RANGES:
            rng = _CANON_RANGES[name]
            r = parse_ref_range(rng)
            f.low, f.high = r["low"], r["high"]
            f.low_strict, f.high_strict = rng.startswith(">"), rng.startswith("<")
        out[name] = f
    return out


_LAB_WORDS = re.compile(r"\blab(?:s|oratory)?\b|\bpanel\b|\bstud(?:y|ies)\b|\btests?\b|\bparameters?\b|"
                        r"\bfunction\b|\bcoagulation\b|\blipid\b|\bthyroid\b|\biron\b|\bliver\b|\brenal\b|\bCBC\b",
                        re.I)
_VITAL_WORDS = re.compile(r"\bvital", re.I)


def panel_pool(clause: str, record: dict[str, Any], table: dict[str, Fact], facts: dict[str, Fact]) -> dict[str, Fact]:
    """Candidates for set/entity/closest checks: the table if it matches the question's domain, else labs from the note."""
    if _domain_ok(clause, record):
        return table
    if record["table"]["type"] == "vitals" and _LAB_WORDS.search(clause):
        return {n: f for n, f in facts.items() if n not in table and n not in _VITAL_RANGES and f.value is not None}
    return {}


def _domain_ok(clause: str, record: dict[str, Any]) -> bool:
    """A set/entity question about labs cannot be keyed from a vitals table (labs are then in the note)."""
    t = record["table"]["type"]
    return not ((t == "vitals" and _LAB_WORDS.search(clause)) or (t == "labs" and _VITAL_WORDS.search(clause)))


_EXPLAIN = re.compile(r"\b(?:cause|reason|why|explain|significance|suggest|implication|interpret|management|plan|"
                      r"recommend|etiology|attribut|associated|contribut|concern|mean|indicate|clinical)\w*", re.I)


def _band(gt: float, kind: str, integer: bool = False) -> tuple[float, float]:
    if kind == "diff":
        tol = 1e-9 if integer else max(0.051, 0.005 * abs(gt))
    elif kind == "ratio":
        tol = max(0.051, 0.05 * abs(gt))
    elif kind in ("pct",):
        tol = max(0.51, 0.05 * abs(gt))
    elif kind == "bmi":
        tol = 0.15
    else:
        tol = 1e-9
    return gt - tol, gt + tol


def _gold_numbers(gold: str) -> list[float]:
    return [_num(m.group(0)) for m in NUM_RE.finditer(gold)]


def _in_band(x: float, lo: float, hi: float) -> bool:
    return lo - 1e-9 <= x <= hi + 1e-9


def _bmi_from_note(record: dict[str, Any]) -> float | None:
    ms = [m for m in record_body_measurements(record) if not m.is_delta]
    w = [m for m in ms if m.kind == "weight"]
    h = [m for m in ms if m.kind == "height"]
    if not w or not h:
        stated = extract_inline_bmi(record["note"])
        return stated[0] if len(set(stated)) == 1 else None
    wk, hc = to_metric(w[0].value, w[0].unit), to_metric(h[0].value, h[0].unit)
    return execute_tool("calculate_bmi", {"weight_kg": round(wk, 1), "height_cm": round(hc, 1)})


def _deviation(f: Fact) -> float:
    """Relative distance outside the range (0 if normal); comparable across units."""
    st = f.state()
    if st == "high" and f.high:
        return f.value / f.high
    if st == "low" and f.low and f.value:
        return f.low / f.value
    return 0.0


def build_key(record: dict[str, Any]) -> Key:
    at = record["answer_type"]
    table = table_facts(record)
    facts = note_facts(record, question_facts(record, table))
    names = list(facts)
    if at == "tool_call":
        gc = {"name": record["tool_calls"][0]["tool"], "arguments": record["tool_calls"][0]["arguments"]}
        grounded = not (_unsupported_loose(record, gc) if gc["name"] == "calculate_bmi" else v1.unsupported_call(record, gc))
        exp = "call" if grounded else "abstain"
        return Key(record["id"], at, exp, [Check("tool", None, record["tool_calls"][0]["result"],
                                                 note="" if grounded else "Q5: gold arguments absent from input")],
                   names, "structured")
    if at == "uncertain":
        return Key(record["id"], at, "abstain", [Check("abstain", None, v1.classify_uncertain(record["question"],
                                                                                              record["answer"]))],
                   names, "structured")

    q = record["question"]
    gold = record["answer"]
    gnums = _gold_numbers(gold)
    gread = read(gold, names)
    checks: list[Check] = []
    unrecognized = False
    last_subject: str | None = None
    for clause in [c for c in _CLAUSE_SPLIT.split(q) if c and c.strip()]:
        cm = [n for _, _, n in mentions(clause, names)]
        subject = cm[0] if cm else last_subject
        if cm:
            last_subject = cm[0]
        f = facts.get(subject) if subject else None
        made = False
        thr = next((m for m in _THRESH.finditer(clause)
                    if not (f and f.value is not None and abs(float(m.group(1)) - f.value) < 1e-9)), None)
        if _EXPLAIN.search(clause) and not any(_OP[o].search(clause) for o in ("diff", "ratio", "pct", "pp")):
            unrecognized = True
            continue
        if last_subject == "BMI" and not cm and checks and checks[-1].kind == "bmi":
            thr_b = re.search(r"(\d+(?:\.\d+)?)\s*kg/m", clause)
            if thr_b:
                t = float(thr_b.group(1))
                above = bool(re.search(r"exceed|above|over|meet|greater", clause, re.I))
                checks.append(Check("yesno", "BMI", checks[-1].gt >= t if above else checks[-1].gt < t))
                continue
        if _OP["bmi"].search(clause) and not cm:
            bmi = _bmi_from_note(record)
            if bmi is not None and isinstance(bmi, float):
                lo, hi = _band(bmi, "bmi")
                checks.append(Check("bmi", "BMI", bmi, lo, hi, gold_agrees=any(_in_band(g, lo, hi) for g in gnums)))
                made = True
                last_subject = "BMI"
            thr_m = re.search(r"(?:threshold|exceed|above|below)\D{0,30}(\d+(?:\.\d+)?)\s*kg/m", clause, re.I)
            if bmi is not None and thr_m and isinstance(bmi, float):
                t = float(thr_m.group(1))
                checks.append(Check("yesno", "BMI", bmi >= t if re.search(r"exceed|above|over|meet", clause, re.I)
                                    else bmi < t, gold_agrees=None))
            if made:
                continue
        pool_tbl = panel_pool(clause, record, table, facts)
        if re.search(r"\b(?:closest|nearest)\b", clause, re.I) and pool_tbl and not cm:
            side_hi = not re.search(r"lower (?:limit|bound|end)", clause, re.I)
            pool_ = [x for x in pool_tbl.values() if x.state() == "normal" and (x.high if side_hi else x.low)]
            if pool_:
                rel = min(pool_, key=lambda x: (x.high - x.value) / x.high if side_hi else (x.value - x.low) / x.low).name
                acc = [rel]
                if len({x.unit for x in pool_}) == 1:
                    ab = min(pool_, key=lambda x: x.high - x.value if side_hi else x.value - x.low).name
                    acc += [ab] if ab not in acc else []
                checks.append(Check("entity", None, rel, accept=acc, note="closest"))
                last_subject = rel
                continue
        if f is not None and f.low is not None and f.high is not None and re.search(r"\bmidpoint\b|\bmiddle of\b", clause,
                                                                                    re.I):
            mid = (f.low + f.high) / 2
            checks.append(Check("diff", subject, round(abs(f.value - mid), 4), *_band(abs(f.value - mid), "diff"),
                                note="midpoint"))
            continue
        if _OP["entity"].search(clause) and pool_tbl:
            pool = [facts[n] for n in dict.fromkeys(n for _, _, n in mentions(q, names)) if n in facts]
            pool = pool if re.search(r"\bthese\b|\bboth\b|\btwo\b|\bor\b|compare", clause, re.I) and len(pool) >= 2 \
                else list(pool_tbl.values())
            if re.search(r"\bother\b", clause, re.I) and last_subject:
                pool = [x for x in pool if x.name != last_subject]
            abn = [x for x in pool if x.state() in ("high", "low")]
            compare = len(pool) == 2 and pool != list(pool_tbl.values())
            if not compare and not re.search(r"outside|abnormal", clause, re.I):
                if re.search(r"elevated|above|high|exceed", clause, re.I):
                    abn = [x for x in abn if x.state() == "high"]
                elif re.search(r"below|\blow|decreased|reduced", clause, re.I):
                    abn = [x for x in abn if x.state() == "low"]
            if abn:
                best = max(abn, key=_deviation).name
                accept = [best]
                if len({x.unit for x in abn}) == 1:  # same unit: absolute distance is a defensible scale too
                    ab = max(abn, key=lambda x: x.value - x.high if x.state() == "high" else x.low - x.value).name
                    if ab not in accept:
                        accept.append(ab)
                gm = [n for n in gread.most if n in facts] or gread.order[:1]
                if gm and gm[0] not in accept and facts[gm[0]].state() in ("high", "low"):
                    accept.append(gm[0])  # a different but defensible scale (absolute vs relative)
                excl_e = [last_subject] if last_subject and re.search(r"\bother\b", clause, re.I) else []
                checks.append(Check("entity", None, best, accept=accept, gold_agrees=bool(gm) and gm[0] == best, exclude=excl_e,
                                    note="compare" if compare else "",
                                    values={x.name: round(_deviation(x), 4) for x in abn} if compare else {}))
                made = True
                last_subject = best
        if not made and subject and subject in facts and re.search(r"\bis (?:this|it|that|the \w+) the (?:most|only)\b",
                                                                    clause, re.I) and _domain_ok(clause, record):
            only = "only" in clause.lower()
            abn = [x for x in (pool_tbl or table).values() if x.state() in ("high", "low")]
            if only:
                gt = [x.name for x in abn] == [subject]
            else:
                gt = bool(abn) and max(abn, key=_deviation).name == subject
            checks.append(Check("yesno", subject, gt, note="is_most" if not only else "is_only"))
            made = True
        if not made and pool_tbl and (_OP["set"].search(clause) and not _OP["diff"].search(clause)
                         and not _OP["ratio"].search(clause)):
            pol = "abnormal" if re.search(r"outside|abnormal|not within|out of range", clause, re.I) else \
                "high" if re.search(r"\belevated|\babove|\bhigh|\bexceed", clause, re.I) else \
                "low" if re.search(r"\bbelow|\blow\b|\bdecreased|\breduced", clause, re.I) else \
                "normal" if re.search(r"within|normal", clause, re.I) else "abnormal"
            excl = [last_subject] if (last_subject and re.search(r"\bother\b|\bonly\b|\balso\b", clause, re.I)
                                      and len(checks)) else []
            want = {"abnormal": {"low", "high"}, "high": {"high"}, "low": {"low"}, "normal": {"normal"}}[pol]
            members = sorted(n for n, x in pool_tbl.items() if x.state() in want and n not in excl)
            only = bool(re.search(r"\bonly\b", clause, re.I))
            singular = bool(re.search(r"\bwhich (?:other |one )?(?:\w+ ){0,2}(?:value|result|test|parameter|lab|marker)\b"
                                      r"(?! values| results| tests)[^?]{0,20}\b(?:is|was|falls|exceeds|lies)\b", clause, re.I))
            checks.append(Check("set", None, members, polarity=pol, exclude=excl,
                                note="only" if only else "singular" if singular else "",
                                gold_agrees=all(gread.states.get(n, STATES) & want for n in members)))
            made = True
            if len(members) == 1 and not excl:
                last_subject = members[0]
        if not made and f is not None and f.value is not None:
            val = f.value
            if f.bp and re.search(r"diastolic", clause, re.I):
                val = f.bp[1]
            if (m := re.search(r"systolic|diastolic", clause, re.I)) and thr and thr.group(3):
                t = float(thr.group(1)) if m.group(0).lower() == "systolic" else float(thr.group(3))
            below = bool(re.search(r"\bbelow\b|\blower limit\b|\bunder\b|\bless\b|\bshort\b", clause, re.I))
            if not (m and thr and thr.group(3)):
                t = None if not thr else float(thr.group("hi")) if thr.group("hi") and not below else float(thr.group(1))
            bound = t if t is not None else (f.low if below else f.high)
            for op in ("pp", "ratio", "pct", "diff"):
                if not _OP[op].search(clause) or bound is None:
                    continue
                if op == "ratio":
                    gt = val / bound
                elif op == "pct":
                    gt = (bound - val) / bound * 100 if below else (val - bound) / bound * 100
                else:
                    gt = abs(val - bound)
                kind = "diff" if op == "pp" else op
                integer = float(val).is_integer() and float(bound).is_integer() and "." not in f.raw
                lo, hi = _band(gt, kind, integer)
                checks.append(Check(kind, subject, round(gt, 4), lo, hi, note=op,
                                    gold_agrees=any(_in_band(g, lo, hi) for g in gnums)))
                made = True
                if op in ("pp", "ratio", "pct"):
                    break
            if not made and _OP["yesno_thr"].search(clause) and t is not None:
                above = bool(re.search(r"above|exceed|over|meet", clause, re.I))
                ans = (val >= t) if above else (val < t)
                if f.bp and thr and thr.group(3) and not re.search(r"systolic|diastolic", clause, re.I):
                    ans = (f.bp[0] >= t or f.bp[1] >= float(thr.group(3))) if above else \
                        (f.bp[0] < t and f.bp[1] < float(thr.group(3)))
                checks.append(Check("yesno", subject, ans, gold_agrees=yes_no(gold) in (None, ans),
                                    note="above" if above else "below"))
                made = True
            if not made and _OP["status"].search(clause) and not f.bp:
                st = f.state()
                if st:
                    accept = [st]
                    gs = gread.states.get(subject)
                    if subject in _VITAL_RANGES and gs and len(gs) == 1 and next(iter(gs)) not in accept:
                        accept.append(next(iter(gs)))  # vitals: conventions differ; gold's stated state accepted
                    checks.append(Check("status", subject, st, accept=accept, gold_agrees=bool(gs) and st in gs))
                    made = True
            if not made and _OP["value"].search(clause):
                raw = f.raw
                checks.append(Check("value", subject, raw, gold_agrees=raw in gold or f.raw.split("/")[0] in gold))
                made = True
                if re.search(r"\b(?:normal|elevated|abnormal|within|outside|high|low)\b", clause, re.I) and \
                        not f.bp and (st := f.state()):
                    checks.append(Check("status", subject, st, accept=[st],
                                        gold_agrees=st in gread.states.get(subject, STATES)))
        if not made:
            unrecognized = True
    if not checks:  # nothing verifiable was recognised: DROP-style text check against the gold
        checks.append(Check("text", None, gold, note="fallback: gold key terms + grounded numbers"))
    coverage = "text" if len(checks) == 1 and checks[0].kind == "text" else "partial" if unrecognized else "structured"
    return Key(record["id"], at, "answer", checks, names, coverage)


# ---------------------------------------------------------------- checking predictions

_STOP = set("""a an the and or of to in on at for with by from as is are was were be been being this that these those
it its their his her he she they them patient patients patient's which what who whom whose when where why how
there here also both all any each other than then into over under about per not no yes noted note documented
reported recorded given currently current daily twice once""".split())


def _terms(text: str) -> list[str]:
    return [w[:6] for w in re.findall(r"[a-z][a-z\-]{3,}", text.lower()) if w not in _STOP]


def check_text(record: dict[str, Any], gold: str, pred: str) -> tuple[bool, str]:
    """DROP-style: gold numbers grounded in the input must appear; >= 50% of grounded gold key terms recalled."""
    inp = v1.input_text(record)
    qnums = {_num(m.group(0)) for m in NUM_RE.finditer(record["question"])}
    inums = {_num(m.group(0)) for m in NUM_RE.finditer(inp)}
    pnums = {_num(m.group(0)) for m in NUM_RE.finditer(pred)}
    first_g = sentences(gold)[0] if sentences(gold) else gold
    need = [g for g in _gold_numbers(first_g) if g in inums and g not in qnums]
    if any(g not in pnums for g in need):
        return False, "text:number_missing"
    if need:
        return True, "text:numbers"  # the grounded numbers are the answer (score, dose, degrees ...)
    inp_terms, q_terms = set(_terms(inp)), set(_terms(record["question"]))
    first = sentences(gold)[0] if sentences(gold) else gold
    key = {t for t in _terms(first) if t in inp_terms and t not in q_terms}
    if not key:
        return True, "text:no_key_terms"
    got = key & set(_terms(pred))
    return (len(got) / len(key) >= 0.4), f"text:recall={len(got)}/{len(key)}"


def _pred_numbers(reading: Reading, subject: str | None, pred: str) -> list[float]:
    if subject and subject in reading.numbers:
        return reading.numbers[subject]
    return [_num(m.group(0)) for m in NUM_RE.finditer(pred)]


def run_check(c: Check, record: dict[str, Any], pred: str, reading: Reading) -> tuple[bool, str]:
    if c.kind == "text":
        return check_text(record, c.gt, pred)
    if c.kind == "value":
        want = parse_bp(c.gt) if "/" in str(c.gt) else None
        if want:
            ok = bool(re.search(rf"(?<![\d.]){int(want[0])}\s*/\s*{int(want[1])}(?![\d.])", pred))
            return ok, "value" if ok else "value:missing"
        v = parse_float(c.gt)
        ok = any(abs(x - v) < 1e-9 for x in [_num(m.group(0)) for m in NUM_RE.finditer(pred)])
        return ok, "value" if ok else "value:missing"
    if c.kind in ("diff", "ratio", "pct", "bmi"):
        cands = [abs(x) for x in _pred_numbers(reading, c.subject, pred)]
        if c.kind == "pct":
            alt = (100 + c.gt) if "below" not in c.note else None
            ok = any(_in_band(x, c.lo, c.hi) for x in cands) or (
                alt is not None and any(abs(x - alt) <= max(0.51, 0.05 * alt) for x in cands))
        else:
            ok = any(_in_band(x, c.lo, c.hi) for x in cands)
        if not ok and c.subject and c.subject in reading.numbers:  # value stated outside the subject's clauses
            ok = any(_in_band(abs(_num(m.group(0))), c.lo, c.hi) for m in NUM_RE.finditer(pred)) and \
                not _band_owned_by_other(reading, c)
        return ok, c.kind if ok else f"{c.kind}:out_of_band"
    if c.kind == "status":
        s = reading.states.get(c.subject)
        if s is None:
            yn = yes_no(pred)
            if yn is not None and re.search(r"within|normal", record["question"], re.I):
                s = frozenset({"normal"}) if yn else frozenset({"low", "high"})
        if s is None:
            return False, "status:not_stated"
        ok = any(a in s for a in c.accept) and s != STATES  # last assertion wins (MedCalc: the final answer counts)
        why = "status" if ok else "status:wrong"
        return ok, why + (":self_corrected" if c.subject in reading.conflicts else "")
    if c.kind == "entity":
        if c.note == "compare" and not reading.most:  # open comparison without a "which is more" claim: unscored
            return True, "entity:compare_unscored"
        cand = [n for n in reading.most if n not in c.exclude][:1] or [n for n in reading.order if n not in c.exclude][:1]
        ok = bool(cand) and cand[0] in c.accept
        return ok, "entity" if ok else f"entity:wrong({cand[0] if cand else None})"
    if c.kind == "set":
        want = {"abnormal": {"low", "high"}, "high": {"high"}, "low": {"low"}, "normal": {"normal"}}[c.polarity]
        facts = note_facts(record, question_facts(record, table_facts(record)))
        claimed = {n for n, s in reading.states.items() if n not in c.exclude and s <= want and s != STATES}
        truly = set(c.gt)
        wrong = {n for n in claimed if facts.get(n) and facts[n].state() not in want}
        if wrong:
            return False, f"set:false_member({','.join(sorted(wrong))})"
        if not truly:
            yn = yes_no(pred)
            none_said = bool(re.search(r"\b(?:no other|none|all (?:other )?(?:\w+ ){0,4}(?:are |were |fall |remain )?"
                                       r"within|only\b|all (?:\w+ ){0,5}normal|no additional)\b", pred, re.I))
            ok = not claimed and (none_said or reading.blanket or yn is not None or c.note == "only")
            return ok, "set" if ok else "set:not_stated"
        missing = truly - claimed
        if c.note == "singular" and truly & claimed:
            missing = set()  # "which other value is also abnormal": one correct member answers it
        if c.polarity == "normal" and reading.blanket:
            missing = set()  # "all values are within normal limits" covers the members it does not name
        return (not missing), "set" if not missing else f"set:missing({','.join(sorted(missing))})"
    if c.kind == "yesno":
        yn = None
        if c.note == "is_most":
            yn = next((yes_no(x) for x in sentences(pred) if _MOST.search(x) and yes_no(x) is not None), None)
            if yn is None and reading.most:
                yn = reading.most[0] == c.subject
        elif c.note == "is_only":
            others = [n for n, st in reading.states.items() if n != c.subject and st <= {"low", "high"}]
            yn = False if others else True if re.search(r"\bonly\b|\bno other\b", pred, re.I) or reading.blanket \
                else None
        if yn is None:
            yn = yes_no(pred)
        if yn is None and c.subject in reading.states:
            s = reading.states[c.subject]
            side = "low" if c.note == "below" else "high"
            yn = True if s == frozenset({side}) else False if side not in s else None
        if yn is None:
            neg = re.search(r"\b(?:not|does not|doesn[’']t|below|under|less than)\b[^.]{0,25}\b(?:exceed\w*|above|"
                            r"threshold|obes\w*)", pred, re.I)
            pos = re.search(r"\b(?:exceed\w*|above|over|meets?|greater than|obese|obesity|class (?:i|ii|iii|1|2|3))\b",
                            pred, re.I)
            yn = False if neg else True if pos else None
        if yn is None:
            return False, "yesno:not_stated"
        return yn == c.gt, "yesno" if yn == c.gt else "yesno:wrong"
    raise ValueError(c.kind)


def _band_owned_by_other(reading: Reading, c: Check) -> bool:
    """True if every in-band number in the text is attributed to a different analyte."""
    owners = [n for n, xs in reading.numbers.items() if any(_in_band(abs(x), c.lo, c.hi) for x in xs)]
    return bool(owners) and c.subject not in owners


# ---------------------------------------------------------------- tool / abstention


def _bmi_tolerance(result: Any, gold: float) -> bool:
    return isinstance(result, (int, float)) and abs(float(result) - gold) <= 0.15


def score_tool(record: dict[str, Any], traj: dict[str, Any], key: Key) -> dict[str, Any]:
    calls = v1._all_calls(traj)
    bad = v1._first_bad_status(traj)
    final = traj.get("final_answer") or ""
    out: dict[str, Any] = {"parse_error": bad in ("invalid_json", "unterminated", "schema_error"),
                           "called": bool(calls) or bad is not None, "n_calls": len(calls)}
    if key.expected == "abstain":  # Q5: no weight/height in the input; the right behaviour is to say so
        unsupported = any(_unsupported_loose(record, c) for c, _, _ in calls)
        ok = not out["called"] and v1.is_abstention(final) and bool(re.search(r"weight|height", final, re.I))
        out.update(correct=ok, error="correct" if ok else "q5_called_unsupported" if unsupported else
                   "q5_no_abstention", relevance_ok=not out["called"])
        return out
    gold_call = record["tool_calls"][0]
    out["relevance_ok"] = out["called"]
    if not calls:
        out.update(correct=False, error="parse_error" if out["parse_error"] else "no_call")
        return out
    call, result, status = calls[0]
    name_ok = call["name"] == gold_call["tool"]
    args = call.get("arguments") or {}
    executed = isinstance(result, (int, float)) and not isinstance(result, bool)
    if gold_call["tool"] == "calculate_bmi":
        outcome_ok = executed and _bmi_tolerance(result, gold_call["result"])
        grounded = not _unsupported_loose(record, call)
    else:
        outcome_ok = executed and math.isclose(float(result), float(gold_call["result"]), rel_tol=0.01, abs_tol=0.011)
        grounded = not v1.unsupported_call(record, call)
    reported = executed and any(abs(_num(m.group(0)) - float(result)) <= max(0.051, 0.005 * abs(float(result)))
                                for m in NUM_RE.finditer(final))
    args_exact, _ = v1.args_ok(record, gold_call, call) if name_ok else (False, None)
    single = len(calls) == 1 and traj.get("stop_reason") == "answer"
    contradicts = False
    analyte = {"creatinine": "Creatinine", "glucose": "Glucose (fasting)", "cholesterol": "Total Cholesterol"}.get(
        str(args.get("substance") or "").lower())
    if analyte:
        facts = note_facts(record, question_facts(record, table_facts(record)))
        true_state = facts[analyte].state() if analyte in facts else None
        said = read(final, list(facts)).states.get(analyte)
        contradicts = bool(true_state and said and said != STATES and true_state not in said)
    ok = name_ok and status == "valid" and grounded and outcome_ok and reported and single and not contradicts
    err = "correct" if ok else "wrong_tool" if not name_ok else "schema_error" if status != "valid" else \
        "ungrounded_args" if not grounded else "outcome_out_of_band" if not outcome_ok else \
        "result_not_reported" if not reported else "status_contradicts_input" if contradicts else "extra_call_or_budget"
    out.update(correct=ok, error=err, name_ok=name_ok, grounded=grounded, outcome_ok=outcome_ok,
               result_reported=reported, args_exact_v1=args_exact)
    return out


def _unsupported_loose(record: dict[str, Any], call: dict[str, Any]) -> bool:
    """calculate_bmi arguments must each be within 0.5% of an input measurement (converted to metric)."""
    args = call.get("arguments") or {}
    ms = [m for m in record_body_measurements(record) if not m.is_delta]
    for key, kind in (("weight_kg", "weight"), ("height_cm", "height")):
        v = args.get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return True
        if not any(abs(v - to_metric(m.value, m.unit)) <= max(0.051, 0.005 * to_metric(m.value, m.unit))
                   for m in ms if m.kind == kind):
            return True
    return False


def score_abstain(record: dict[str, Any], traj: dict[str, Any], key: Key) -> dict[str, Any]:
    final = traj.get("final_answer") or ""
    calls = v1._all_calls(traj)
    unsupported = [c for c, _, _ in calls if v1.unsupported_call(record, c)]
    if unsupported:
        return {"correct": False, "error": "unsupported_call", "called": True}
    if not final.strip():
        return {"correct": False, "error": f"no_answer_{traj.get('stop_reason')}", "called": bool(calls)}
    ok, err, cat = v1.score_uncertain(record, final)
    if ok and _DEFINITE.search(final) and not any(_HEDGED.search(x) for x in sentences(final) if _DEFINITE.search(x)):
        ok, err = False, "definite_conclusion_despite_missing"
    return {"correct": ok, "error": err, "category": cat, "called": bool(calls)}


_DEFINITE = re.compile(r"\b(?:it is|is|are|would be|should be) (?:considered |generally )?safe\b|\bsafe to (?:administer|"
                       r"prescribe|give|use|start|continue)\b|\bcan (?:be )?safely\b|\bno contraindication", re.I)
_HEDGED = re.compile(r"\b(?:cannot|can't|unable to|not possible to|difficult to|impossible to)\b[^.]{0,60}\b(?:determin|"
                     r"confirm|assess|conclud|say|establish|know)\w*[^.]{0,40}\bsafe|\bwhether\b[^.]{0,60}\bsafe|"
                     r"\b(?:may|might|could) (?:be|not be) safe|\bsafe\b[^.]{0,40}\b(?:cannot|unless|until|only if|pending)\b",
                     re.I)


# ---------------------------------------------------------------- entry point


def score(record: dict[str, Any], traj: dict[str, Any], key: Key | None = None) -> dict[str, Any]:
    key = key or build_key(record)
    base = {"id": record["id"], "answer_type": record["answer_type"], "expected": key.expected,
            "coverage": key.coverage, "scorer": VERSION}
    if record["answer_type"] == "tool_call":
        return {**base, **score_tool(record, traj, key)}
    if record["answer_type"] == "uncertain":
        return {**base, **score_abstain(record, traj, key)}
    final = traj.get("final_answer") or ""
    calls = v1._all_calls(traj)
    if not final.strip():
        return {**base, "correct": False, "error": f"no_answer_{traj.get('stop_reason')}", "checks": [],
                "called": bool(calls)}
    final = normalize(final)
    reading = read(final, key.names)
    results = []
    for c in key.checks:
        ok, why = run_check(c, record, final, reading)
        results.append({"kind": c.kind, "subject": c.subject, "ok": ok, "why": why})
    ok = all(r["ok"] for r in results)
    abstained = v1.is_abstention(final)
    err = "correct" if ok else ("over_refusal" if abstained else next(r["why"] for r in results if not r["ok"]))
    return {**base, "correct": ok, "error": err, "checks": results, "called": bool(calls),
            "over_refusal": abstained and not ok}


def gold_trajectory(record: dict[str, Any]) -> dict[str, Any]:
    return v1.gold_trajectory(record)
