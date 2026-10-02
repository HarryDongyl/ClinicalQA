"""Stretch A data generation: calculate_egfr examples (docs/STRETCH_A_PLAN.md sections 5-6).

    uv run python scripts/stretch_a_data.py            # writes data/stretch_a/ and reports/stretch_a/review_sheet.md

Outputs (refuses to overwrite):
  data/stretch_a/train_additions.jsonl   40 positives + 6 age-removed + 6 sex-removed negatives (train notes only)
  data/stretch_a/val_egfr.jsonl          19 val positives (needs user review before freezing)
  data/stretch_a/val_egfr_age_probes.jsonl  the same 19 with age removed (expected: abstain)
  data/stretch_a/manifest.json           seed, formula, source ids, sha256 of every output
  reports/stretch_a/review_sheet.md      human review sheet for the 19 + 19 evaluation items

Records use the canonical schema (id, note, table, question, answer, answer_type, tool_calls) plus `source_id` and
`stretch_a` metadata. Only records whose table has Creatinine (mg/dL) and no eGFR row are used: the synthetic table
eGFR is not derivable from creatinine, age and sex (STRETCH_A_PLAN.md section 2), so tool and table never coexist.
The CKD-EPI 2021 function here is the reference implementation that `tools.calculate_egfr` must reproduce (SA1).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from clinqa.parsing import extract_age_sex, parse_float

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_POS, N_NEG_AGE, N_NEG_SEX = 40, 6, 6

QUESTIONS = [
    "Estimate the patient's eGFR from the available serum creatinine, age and sex, and state the CKD stage.",
    "Using the documented creatinine, age and sex, what is this patient's estimated GFR and corresponding CKD stage?",
    "What is the patient's estimated glomerular filtration rate (CKD-EPI) based on their creatinine, age and sex?",
    "Calculate this patient's eGFR from the creatinine in the lab table and the patient's age and sex.",
]


def ckd_epi_2021(creatinine_mg_dl: float, age: int, sex: str) -> int:
    """Race-free CKD-EPI 2021 creatinine equation, rounded half-up to an integer mL/min/1.73m2."""
    female = sex == "female"
    kappa, alpha = (0.7, -0.241) if female else (0.9, -0.302)
    ratio = creatinine_mg_dl / kappa
    egfr = 142 * min(ratio, 1) ** alpha * max(ratio, 1) ** -1.200 * 0.9938 ** age * (1.012 if female else 1.0)
    return int(Decimal(str(egfr)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def g_stage(egfr: int) -> str:
    for cut, stage in ((90, "G1"), (60, "G2"), (45, "G3a"), (30, "G3b"), (15, "G4")):
        if egfr >= cut:
            return stage
    return "G5"


# ---------------------------------------------------------------- note edits

_AGE_PATTERNS = [
    re.compile(r"\b\d{1,3}[- ](?:year|yr)s?[- ]old\s+", re.I),        # "62-year-old woman" -> "woman"
    re.compile(r",?\s*\(?\b(?:aged?|age:)\s*\d{1,3}\)?", re.I),           # "aged 62", "age: 62"
    re.compile(r"\s*\(?\b\d{1,3}\s*(?:yo|y/o|y\.o\.)\)?", re.I),           # "62 yo"
]
# Hard flag: a numeric age cue survives. Durations ("2 years ago", "for 10 years") are not age cues.
AGE_RESIDUE = re.compile(r"\b\d{1,3}[- ](?:year|yr)s?[- ]old\b|\b(?:twenties|thirties|forties|fifties|sixties|seventies|"
                         r"eighties|nineties)\b|\b\d0s\b|\baged?\s*:?\s*\d", re.I)
# Soft note for the reviewer: the concept of age is mentioned without a value ("given age", "age-appropriate").
AGE_SOFT = re.compile(r".{0,40}\bage\b.{0,30}", re.I)
_SEX_WORDS = [(re.compile(r"\b(?:male|female|man|woman|gentleman|lady|boy|girl)\b", re.I), "patient"),
              (re.compile(r"\bM(?:r|rs|s)\.\s*", re.I), "Patient ")]
_PRONOUNS = [("himself", "themselves"), ("herself", "themselves"), ("he", "the patient"), ("she", "the patient"),
             ("him", "the patient"), ("his", "the patient's")]
_HER_OBJECT = re.compile(r"^\s*(?:[.,;:)!?]|to\b|at\b|with\b|for\b|and\b|on\b|in\b|of\b|from\b|by\b|about\b|as\b|that\b|"
                         r"this\b|$)", re.I)
SEX_RESIDUE = re.compile(r"\b(?:male|female|man|woman|men|women|gentleman|lady|he|she|him|his|her|hers|himself|herself|"
                         r"mr|mrs|ms|husband|wife|boyfriend|girlfriend)\b", re.I)


def _cap(match_text: str, repl: str) -> str:
    return repl[0].upper() + repl[1:] if match_text[:1].isupper() else repl


_BARE_SEX = re.compile(r"(?P<lead>(?:^|[:.](?:\*\*)?\s+|\n)\s*(?:\*\*)?)(?P<sex>male|female)\b(?!\s+patient)", re.I)
_ARTICLE_SEX = re.compile(r"\b(?P<art>an?)\s+(?P<sex>male|female)\b(?!\s+patient)", re.I)


def remove_age(note: str) -> str:
    """Drop the numeric age; keep the sentence grammatical ("HPI: 58-year-old female with" -> "HPI: Female patient with")."""
    for pat in _AGE_PATTERNS:
        note = pat.sub("", note)
    note = _BARE_SEX.sub(lambda m: f"{m.group('lead')}{m.group('sex').capitalize()} patient", note)
    note = _ARTICLE_SEX.sub(lambda m: f"{'A' if m.group('art')[0].isupper() else 'a'} {m.group('sex').lower()} patient", note)
    return note


def remove_sex(note: str) -> str:
    for pat, repl in _SEX_WORDS:
        note = pat.sub(lambda m, r=repl: _cap(m.group(0), r), note)
    for word, repl in _PRONOUNS:
        note = re.sub(rf"\b{word}\b", lambda m, r=repl: _cap(m.group(0), r), note, flags=re.I)

    def her(m: re.Match[str]) -> str:
        return _cap(m.group(0), "the patient" if _HER_OBJECT.match(note[m.end():]) else "the patient's")

    note = re.sub(r"\bher\b", her, note, flags=re.I)
    return re.sub(r"\ban patient\b", "a patient", re.sub(r"\bAn patient\b", "A patient", note))


# ---------------------------------------------------------------- records


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def eligible(split: str) -> list[dict]:
    out = []
    for r in read_jsonl(ROOT / "data" / f"{split}.jsonl"):
        rows = {x[0]: x for x in r["table"]["rows"]}
        if "Creatinine" not in rows or "eGFR" in rows or rows["Creatinine"][2] != "mg/dL":
            continue
        age, sex = extract_age_sex(r["note"])
        cr = parse_float(rows["Creatinine"][1])
        if age is None or sex is None or cr is None or not 18 <= age <= 120:
            continue
        out.append({"src": r, "age": age, "sex": "female" if sex == "F" else "male", "cr": cr})
    return out


def positive(e: dict, rid: str, q: str) -> dict:
    egfr = ckd_epi_2021(e["cr"], e["age"], e["sex"])
    answer = (f"Using the patient's serum creatinine of {e['cr']} mg/dL, age {e['age']} and sex ({e['sex']}), the "
              f"estimated GFR (CKD-EPI 2021) is {egfr} mL/min/1.73m² (KDIGO GFR category {g_stage(egfr)}, assuming stable "
              f"kidney function).")
    return {"id": rid, "source_id": e["src"]["id"], "note": e["src"]["note"], "table": e["src"]["table"], "question": q,
            "answer": answer, "answer_type": "tool_call",
            "tool_calls": [{"tool": "calculate_egfr",
                            "arguments": {"creatinine_mg_dl": e["cr"], "age": e["age"], "sex": e["sex"]},
                            "result": egfr}],
            "stretch_a": {"kind": "positive", "egfr": egfr, "stage": g_stage(egfr)}}


def negative(e: dict, rid: str, q: str, missing: str) -> dict:
    note = remove_age(e["src"]["note"]) if missing == "age" else remove_sex(e["src"]["note"])
    documented = "sex" if missing == "age" else "age"
    answer = (f"The serum creatinine ({e['cr']} mg/dL) and {documented} are documented, but the patient's {missing} is "
              f"not recorded, so eGFR cannot be calculated.")
    residue = (AGE_RESIDUE if missing == "age" else SEX_RESIDUE).findall(note)
    still = extract_age_sex(note)[0 if missing == "age" else 1]
    return {"id": rid, "source_id": e["src"]["id"], "note": note, "table": e["src"]["table"], "question": q,
            "answer": answer, "answer_type": "uncertain", "tool_calls": None,
            "stretch_a": {"kind": f"negative_{missing}", "residue": residue, "extractor_still_finds": still is not None,
                          "soft_age_mentions": [m.group(0).strip() for m in AGE_SOFT.finditer(note)]
                          if missing == "age" else []}}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    data, rep = ROOT / "data" / "stretch_a", ROOT / "reports" / "stretch_a"
    for d in (data, rep):
        if d.exists():
            raise SystemExit(f"{d} exists; generated data is never overwritten")
    rng = random.Random(SEED)
    train = eligible("train")
    rng.shuffle(train)
    assert len(train) >= N_POS + N_NEG_AGE + N_NEG_SEX, len(train)
    pos = [positive(e, f"sa_train_{i:03d}", QUESTIONS[i % 4]) for i, e in enumerate(train[:N_POS])]
    neg_age = [negative(e, f"sa_train_{N_POS + i:03d}", QUESTIONS[i % 4], "age")
               for i, e in enumerate(train[N_POS:N_POS + N_NEG_AGE])]
    neg_sex = [negative(e, f"sa_train_{N_POS + N_NEG_AGE + i:03d}", QUESTIONS[i % 4], "sex")
               for i, e in enumerate(train[N_POS + N_NEG_AGE:N_POS + N_NEG_AGE + N_NEG_SEX])]
    val = eligible("val")
    val_pos = [positive(e, f"sa_val_{i:03d}", QUESTIONS[i % 4]) for i, e in enumerate(val)]
    val_probe = [negative(e, f"sa_val_age_{i:03d}", QUESTIONS[i % 4], "age") for i, e in enumerate(val)]
    data.mkdir(parents=True)
    rep.mkdir(parents=True)
    files = {"train_additions.jsonl": pos + neg_age + neg_sex, "val_egfr.jsonl": val_pos,
             "val_egfr_age_probes.jsonl": val_probe}
    for name, rows in files.items():
        (data / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    flagged = [r["id"] for rows in files.values() for r in rows
               if r["stretch_a"].get("residue") or r["stretch_a"].get("extractor_still_finds")]
    manifest = {"seed": SEED, "formula": "CKD-EPI 2021 race-free, integer half-up", "status": "generated; val items "
                "await user review (STRETCH_A_PLAN.md section 6)", "counts": {k: len(v) for k, v in files.items()},
                "eligible_pool": {"train": len(train), "val": len(val)}, "flagged_for_residue": flagged,
                "stage_distribution_val": {s: sum(r["stretch_a"]["stage"] == s for r in val_pos)
                                           for s in ("G1", "G2", "G3a", "G3b", "G4", "G5")},
                "sha256": {k: sha(data / k) for k in files}}
    (data / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    lines = ["# Stretch A evaluation review sheet (19 positives + 19 age-removed probes)", "",
             "Check for each item: age/sex/creatinine extracted correctly; eGFR and stage plausible; the probe note reads "
             "naturally with no residual age cue; the question is sensible. Mark approve / fix / drop.", ""]
    for p_, q_ in zip(val_pos, val_probe):
        src = p_["note"]
        span = re.search(r".{0,40}\b\d{1,3}[- ](?:year|yr)s?[- ]old\b.{0,40}", src)
        lines += [f"## {p_['id']} (source {p_['source_id']})", "",
                  f"- age **{p_['tool_calls'][0]['arguments']['age']}**, sex **{p_['tool_calls'][0]['arguments']['sex']}**, "
                  f"creatinine **{p_['tool_calls'][0]['arguments']['creatinine_mg_dl']} mg/dL** → eGFR "
                  f"**{p_['stretch_a']['egfr']}**, stage **{p_['stretch_a']['stage']}**",
                  f"- age span: `{span.group(0).strip() if span else 'NOT FOUND'}`",
                  f"- question: {p_['question']}",
                  f"- probe residue flags: {q_['stretch_a']['residue'] or 'none'}; extractor still finds age: "
                  f"{q_['stretch_a']['extractor_still_finds']}",
                  f"- soft age mentions (no value; reviewer decides): {q_['stretch_a']['soft_age_mentions'] or 'none'}",
                  "- probe note (first 400 chars):", "", "  > " + q_["note"][:400].replace("\n", " "), "",
                  "- decision: [ ] approve  [ ] fix  [ ] drop", ""]
    (rep / "review_sheet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("counts", "eligible_pool", "flagged_for_residue",
                                                "stage_distribution_val")}, indent=1))


if __name__ == "__main__":
    main()
