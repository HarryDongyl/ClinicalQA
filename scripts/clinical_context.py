"""Clinical-context check (CC) for tool answers: assignment item "final answer incorporates the tool result with brief
clinical context" (INTERVIEW_PREP.md section 5.13). Post-hoc diagnostic added 2026-10-02; rules developed on train gold.

    uv run python scripts/clinical_context.py gold --split train          # rule self-check on gold answers
    uv run python scripts/clinical_context.py score --labels <label> ... --out reports/w3/results_2026-10-02/clinical_context

Rules (ground truth from the executed tool result and the input, never from gold text):
  BMI         the WHO category of the executed result (<18.5 underweight, <25 normal, <30 overweight, >=30 obese)
              must be stated in a sentence about the BMI, with no conflicting category asserted. Within 0.15 of a
              cut-off, either neighbouring category is accepted (rounding of reported values).
  conversion  the converted analyte's status (low / normal / high against the input reference range) must be stated
              and agree with the input; an unparseable or missing status fails.
Only tool records whose input supports the call (scorer v2.1 key expected == "call") are applicable. A model answer is
scored only when the call executed; otherwise it is reported as not applicable (the tool metrics already fail it).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from clinqa import scorer_v2 as s2
from clinqa.evaluate import wilson

ROOT = Path(__file__).resolve().parents[1]
CUTS = ((18.5, "underweight"), (25.0, "normal"), (30.0, "overweight"))
CATEGORY = {
    "underweight": re.compile(r"\bunder-?weight\b", re.I),
    "normal": re.compile(r"\b(?:normal|healthy|ideal)(?: body)? weight\b|\bnormal(?: BMI)? (?:range|category)\b|"
                         r"\bwithin (?:the )?normal(?: BMI)?(?: range)?\b|\bnormal BMI\b", re.I),
    "overweight": re.compile(r"\bover-?weight\b|\bpre-?obes\w*", re.I),
    "obese": re.compile(r"\bobes(?:e|ity)\b|\bclass (?:I{1,3}|[123])\b", re.I),
}
# A category word that names a threshold, a risk or history is not a classification of this BMI.
NOT_ASSERTED = re.compile(r"(?:below|under|short of|approach\w*|near|threshold|cut-?off|risk|history|histor\w* of|"
                          r"not|no longer|rather than|than|from|contribut\w*|related|associated|despite)\W+"
                          r"(?:\w+\W+){0,3}$", re.I)
CONVERT_ANALYTE = {"creatinine": "Creatinine", "glucose": "Glucose (fasting)", "cholesterol": "Total Cholesterol"}


def who(bmi: float) -> set[str]:
    cat = next((c for cut, c in CUTS if bmi < cut), "obese")
    near = {c for cut, c in CUTS if abs(bmi - cut) <= 0.15}  # the lower-side category of a nearby cut-off
    near |= {next((c for cut, c in CUTS if x < cut), "obese") for x in (bmi - 0.15, bmi + 0.15)}
    return {cat} | near


PAREN = re.compile(r"\([^)]*\)")
NUM = re.compile(r"(?<![\d.])\d+(?:\.\d+)?")
STATUS_NORMALISE = [
    (re.compile(r"\b(?:conventional|standard|typical|usual|general|adult|expected)\s+(?=normal|reference)", re.I), ""),
    (re.compile(r"\b(?:at|near|toward|towards) the (?:upper|lower|high|low) end of the (?:normal|reference)(?: reference)? range",
                re.I), "within the normal range"),
    (re.compile(r"\b(?:just )?below the (?:desirable|recommended|target|optimal) (?:threshold|limit|level|cut-?off)", re.I),
     "within the normal range"),
    (re.compile(r"\b(?:near|close to|approaching) the (?:upper limit of (?:the )?normal|desirable range)", re.I),
     "within the normal range"),
]


def _mentions_value(sent: str, value: float) -> bool:
    return any(abs(float(m.group(0)) - value) <= max(0.051, 0.005 * abs(value)) for m in NUM.finditer(sent))


REFERS_BACK = re.compile(r"^\W*(?:\w+\W+){0,2}?(?:this|it|which|that|the value|this value|this level|the result)\b|"
                         r"^\W*(?:despite|although|though|while)\s+(?:\w+ing|being)\b", re.I)
CLASSIFIES = re.compile(r"\b(?:classif\w*|categor\w*|places? (?:him|her|the patient)|falls? (?:in|within|into)|range)\b", re.I)


def claim_sentences(text: str, value: float, follow: re.Pattern[str] | None = None,
                    other_analytes: list[str] | None = None) -> list[str]:
    """Sentences that state the result, plus a directly following sentence that refers back to it.

    A follow-up sentence is kept only if it matches `follow` (when given), and only up to the first mention of another
    analyte, so commentary such as "obesity is a common driver of HFpEF" or "discordant with the elevated HbA1c" is not
    read as a claim about this result.
    Parentheticals (reference ranges, cut-off lists) are removed.
    """
    sents = s2.sentences(s2.normalize(text))
    out = []
    for i, sent in enumerate(sents):
        if not _mentions_value(sent, value):
            continue
        out.append(sent)
        if i + 1 < len(sents):
            nxt = sents[i + 1]
            others = s2.mentions(nxt, other_analytes) if other_analytes else []
            if others:
                nxt = nxt[:others[0][0]]  # keep only the clause before another analyte is named
            if REFERS_BACK.match(nxt) and (follow is None or follow.search(nxt)):
                out.append(nxt)
    return [PAREN.sub("", x) for x in out]


def stated_categories(text: str, result: float) -> set[str]:
    out: set[str] = set()
    for sent in claim_sentences(text, result, follow=CLASSIFIES):
        for cat, pat in CATEGORY.items():
            for m in pat.finditer(sent):
                if not NOT_ASSERTED.search(sent[max(0, m.start() - 40):m.start()]):
                    out.add(cat)
    if "obese" in out and "overweight" in out and re.search(r"pre-?obes", text, re.I):
        out.discard("obese")
    return out


def check(record: dict, call: dict, result: float, text: str) -> tuple[bool, str]:
    if call["name"] == "calculate_bmi":
        want = who(float(result))
        got = stated_categories(text, float(result))
        if not got:
            return False, "bmi:no_category"
        if got - want:
            return False, f"bmi:conflict({','.join(sorted(got))} vs {','.join(sorted(want))})"
        return True, "bmi:ok"
    analyte = CONVERT_ANALYTE.get(str((call.get("arguments") or {}).get("substance") or "").lower())
    if not analyte:
        return False, "convert:unknown_substance"
    facts = s2.note_facts(record, s2.question_facts(record, s2.table_facts(record)))
    truth = facts[analyte].state() if analyte in facts else None
    if truth is None:
        return False, "convert:no_input_range"
    said = None
    for sent in claim_sentences(text, float(result), other_analytes=[n for n in facts if n != analyte]):
        for pat, rep in STATUS_NORMALISE:
            sent = pat.sub(rep, sent)
        phrases = [st for _, _, st in s2._state_phrases(sent) if st != s2.STATES]
        if phrases:
            said = phrases[0]  # the first status asserted about the converted value
            break
    if said is None:
        return False, "convert:no_status"
    return (truth in said), "convert:ok" if truth in said else f"convert:wrong({','.join(sorted(said))} vs {truth})"


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def applicable(records: dict[str, dict]) -> list[str]:
    return [i for i, r in records.items() if r["answer_type"] == "tool_call" and s2.build_key(r).expected == "call"]


def summarize(rows: list[dict]) -> dict:
    out = {}
    for tool in ("calculate_bmi", "unit_convert", "all"):
        sub = [r for r in rows if r["ok"] is not None and (tool == "all" or r["tool"] == tool)]
        k = sum(r["ok"] for r in sub)
        out[tool] = {"k": k, "n": len(sub), "rate": round(k / len(sub), 4) if sub else None, "ci95": wilson(k, len(sub))}
    out["not_applicable_no_executed_call"] = sum(r["ok"] is None for r in rows)
    return out


def cmd_gold(a: argparse.Namespace) -> None:
    records = {r["id"]: r for r in read_jsonl(ROOT / "data" / f"{a.split}.jsonl")}
    rows = []
    for i in applicable(records):
        r = records[i]
        gc = r["tool_calls"][0]
        ok, why = check(r, {"name": gc["tool"], "arguments": gc["arguments"]}, gc["result"], r["answer"])
        rows.append({"id": i, "tool": gc["tool"], "ok": ok, "why": why})
    print(json.dumps(summarize(rows)))
    for r in rows:
        if not r["ok"]:
            print(r["id"], r["why"], "|", records[r["id"]]["answer"][:160].replace("\n", " "))


def cmd_score(a: argparse.Namespace) -> None:
    records = {r["id"]: r for r in read_jsonl(ROOT / "data" / "val.jsonl")}
    ids = applicable(records)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Clinical-context check (post-hoc diagnostic, rules developed on train gold)", "",
          "| label | BMI | conversion | all | no executed call |", "|---|---|---|---|---|"]
    for label in a.labels:
        traj = {t["id"]: t for t in read_jsonl(ROOT / "outputs" / label / "val" / "trajectories.jsonl")}
        rows = []
        for i in ids:
            t = traj[i]
            calls = [(c, res) for turn in t["turns"] for c, res in zip(turn.get("calls", []), turn.get("results", []))]
            gc = records[i]["tool_calls"][0]
            hit = next(((c, res) for c, res in calls if c["name"] == gc["tool"] and isinstance(res, (int, float))), None)
            if hit is None or not t.get("final_answer"):
                rows.append({"id": i, "tool": gc["tool"], "ok": None, "why": "no_executed_call_or_answer"})
                continue
            ok, why = check(records[i], hit[0], hit[1], t["final_answer"])
            rows.append({"id": i, "tool": gc["tool"], "ok": ok, "why": why})
        summ = summarize(rows)
        (out / f"{label}.json").write_text(json.dumps({"label": label, "summary": summ, "rows": rows}, indent=1) + "\n")
        f = lambda s: f"{s['k']}/{s['n']}"  # noqa: E731
        md.append(f"| {label} | {f(summ['calculate_bmi'])} | {f(summ['unit_convert'])} | {f(summ['all'])} | "
                  f"{summ['not_applicable_no_executed_call']} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gold")
    g.add_argument("--split", default="train", choices=["train", "val"])
    s = sub.add_parser("score")
    s.add_argument("--labels", nargs="+", required=True)
    s.add_argument("--out", required=True)
    a = p.parse_args()
    {"gold": cmd_gold, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
