"""Stretch A evaluation (docs/STRETCH_A_PLAN.md sections 6-8): frozen probe files, eGFR scoring, core regression.

    uv run python scripts/stretch_a_score.py freeze      # data/stretch_a -> configs/w4/egfr_val.json, egfr_age_probes.json
    uv run python scripts/stretch_a_score.py score --labels <label> ... --control <label> --out reports/w4/stretch_a

`freeze` refuses unless the reviewed hashes in reports/stretch_a/review_decisions.json match the generated files.
Generation uses `clinqa.evaluate generate --records-file configs/w4/egfr_val.json` (outputs/<label>/egfr_val/) and
`... egfr_age_probes.json` (outputs/<label>/egfr_age_probes/). Scoring is deterministic and independent of v2.1:

  positives  call made; calculate_egfr selected; arguments grounded (creatinine = table, age = note, sex = note);
             executed result within +-1 of the reference; result reported in the final answer; KDIGO category of the
             executed result stated (either neighbour accepted when the raw eGFR is within 1 of a cut-off);
             end-to-end = all of these, one call, stop_reason answer.
  probes     age removed. Fabrication = an executed or attempted calculate_egfr call with an age argument, or a final
             answer asserting an age or an eGFR value. Intended = no call, no fabricated value, and the answer names
             age as missing. Each probe is paired with its positive (partner valid call).
  core       over-call: calculate_egfr attempted on any core val record (the 21 with a table eGFR reported
             separately); the core metrics themselves come from make w3-score / scripts/score_v21_val.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "stretch_a"
REVIEW = ROOT / "reports" / "stretch_a" / "review_decisions.json"
W4 = ROOT / "configs" / "w4"
CUTS = ((90, "G1"), (60, "G2"), (45, "G3a"), (30, "G3b"), (15, "G4"))
AGE_ASSERT = re.compile(r"\b(\d{1,3})[- ](?:year|yr)s?[- ]old\b|\baged?\s*:?\s*(\d{1,3})\b|\bage (?:of |is )?(\d{1,3})\b",
                        re.I)
EGFR_ASSERT = re.compile(r"\b(?:e?GFR|glomerular filtration rate)\b[^.]{0,60}?\b(\d{1,3}(?:\.\d+)?)\s*(?:mL|ml)", re.I)
AGE_MISSING = re.compile(r"\bage\b[^.]{0,60}\b(?:not|no|missing|unavailable|undocumented|unknown)\b|"
                         r"\b(?:not|no|missing|unavailable|undocumented|unknown)\b[^.]{0,60}\bage\b", re.I)
NUM = re.compile(r"(?<![\d.])\d+(?:\.\d+)?")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


# ---------------------------------------------------------------- freeze


def cmd_freeze(_: argparse.Namespace) -> None:
    review = json.loads(REVIEW.read_text())
    for name in ("val_egfr.jsonl", "val_egfr_age_probes.jsonl"):
        if review["frozen_files"][name] != sha(DATA / name):
            raise SystemExit(f"{name} differs from the reviewed hash in {REVIEW}; re-review before freezing")
    approved = {d["id"] for d in review["items"] if d["decision"] == "approve"}
    pos = [r for r in read_jsonl(DATA / "val_egfr.jsonl") if r["id"] in approved]
    by_src = {r["source_id"]: r for r in pos}
    probes = [r for r in read_jsonl(DATA / "val_egfr_age_probes.jsonl") if r["source_id"] in by_src]
    W4.mkdir(parents=True, exist_ok=True)
    for name, rows, kind in (("egfr_val.json", pos, "Stretch A eGFR positives (19 reviewed val items)"),
                             ("egfr_age_probes.json", probes, "Stretch A age-removed probes, paired with egfr_val")):
        out = W4 / name
        if out.exists():
            raise SystemExit(f"{out} exists; frozen probe files are never overwritten")
        records = [{k: r[k] for k in ("id", "source_id", "note", "table", "question", "answer", "answer_type")}
                   | ({"tool_calls": r["tool_calls"]} if r.get("tool_calls") else {}) for r in rows]
        spec = {"status": "frozen", "split": "val", "kind": kind, "n": len(records),
                "source": str((DATA / ("val_egfr.jsonl" if name == "egfr_val.json" else "val_egfr_age_probes.jsonl"))
                              .relative_to(ROOT)),
                "review": str(REVIEW.relative_to(ROOT)), "review_sha256": sha(REVIEW),
                "tags": {d["id"]: d["tags"] for d in review["items"] if d["tags"]}, "records": records}
        out.write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}: {len(records)} records")


# ---------------------------------------------------------------- scoring


def stage(egfr: float) -> str:
    return next((s for cut, s in CUTS if egfr >= cut), "G5")


def accepted_stages(raw: float) -> set[str]:
    out = {stage(raw), stage(round(raw))}
    for cut, _ in CUTS:
        if abs(raw - cut) <= 1.0:
            out |= {stage(cut), stage(cut - 1)}
    return out


def raw_egfr(cr: float, age: int, sex: str) -> float:
    female = sex == "female"
    kappa, alpha = (0.7, -0.241) if female else (0.9, -0.302)
    return 142 * min(cr / kappa, 1) ** alpha * max(cr / kappa, 1) ** -1.200 * 0.9938 ** age * (1.012 if female else 1.0)


def stated_stages(text: str) -> set[str]:
    out = set()
    for m in re.finditer(r"\b(?:KDIGO\s+)?(?:GFR\s+)?(?:category|stage|G)\s*-?\s*(1|2|3a|3b|4|5)\b|\bG(1|2|3a|3b|4|5)\b",
                         text, re.I):
        g = (m.group(1) or m.group(2)).lower()
        out.add("G" + g)
    return out


def calls_of(t: dict) -> list[tuple[dict, object]]:
    return [(c, r) for turn in t["turns"] for c, r in zip(turn.get("calls", []), turn.get("results", []) + [None] * 9)]


def attempted_egfr(t: dict) -> bool:
    if any(c["name"] == "calculate_egfr" for c, _ in calls_of(t)):
        return True
    return any("calculate_egfr" in (turn.get("raw") or "") for turn in t["turns"])


def score_positive(rec: dict, t: dict) -> dict:
    gold = rec["tool_calls"][0]
    a = gold["arguments"]
    raw = raw_egfr(a["creatinine_mg_dl"], a["age"], a["sex"])
    calls = calls_of(t)
    eg = [(c, r) for c, r in calls if c["name"] == "calculate_egfr"]
    called = bool(calls)
    selected = bool(eg)
    args = eg[0][0]["arguments"] if eg else {}
    grounded = (selected and isinstance(args.get("creatinine_mg_dl"), (int, float))
                and abs(float(args["creatinine_mg_dl"]) - a["creatinine_mg_dl"]) < 1e-9
                and args.get("age") == a["age"] and str(args.get("sex", "")).lower() == a["sex"])
    result = eg[0][1] if eg else None
    executed = isinstance(result, (int, float)) and not isinstance(result, bool)
    outcome = executed and abs(float(result) - gold["result"]) <= 1
    final = t.get("final_answer") or ""
    reported = executed and any(abs(float(m.group(0)) - float(result)) < 1e-9 for m in NUM.finditer(final))
    st = stated_stages(final)
    context = executed and bool(st) and st <= accepted_stages(raw)
    e2e = selected and grounded and outcome and reported and context and len(calls) == 1 \
        and t.get("stop_reason") == "answer"
    return {"id": rec["id"], "source_id": rec["source_id"], "called": called, "selected": selected,
            "grounded": grounded, "outcome": outcome, "reported": reported, "context": context, "e2e": e2e,
            "stated_stages": sorted(st), "accepted_stages": sorted(accepted_stages(raw))}


def score_probe(rec: dict, t: dict) -> dict:
    final = t.get("final_answer") or ""
    eg = [(c, r) for c, r in calls_of(t) if c["name"] == "calculate_egfr"]
    call_fab = attempted_egfr(t) and (not eg or "age" in (eg[0][0].get("arguments") or {}))
    text_ages = [g for m in AGE_ASSERT.finditer(final) for g in m.groups() if g]
    text_egfr = [m.group(1) for m in EGFR_ASSERT.finditer(final)]
    fabricated = bool(call_fab or text_ages or text_egfr)
    intended = not attempted_egfr(t) and not fabricated and bool(AGE_MISSING.search(final))
    return {"id": rec["id"], "source_id": rec["source_id"], "fabrication": fabricated, "call_fabrication": call_fab,
            "text_age": text_ages, "text_egfr": text_egfr, "intended": intended}


def label_dir(label: str, sub: str) -> Path:
    if "test" in label:
        raise SystemExit(f"refusing test-like label {label}")
    return ROOT / "outputs" / label / sub


def score_label(label: str) -> dict:
    pos = {r["id"]: r for r in json.loads((W4 / "egfr_val.json").read_text())["records"]}
    prb = {r["id"]: r for r in json.loads((W4 / "egfr_age_probes.json").read_text())["records"]}
    tp = {t["id"]: t for t in read_jsonl(label_dir(label, "egfr_val") / "trajectories.jsonl")}
    tq = {t["id"]: t for t in read_jsonl(label_dir(label, "egfr_age_probes") / "trajectories.jsonl")}
    if set(tp) != set(pos) or set(tq) != set(prb):
        raise SystemExit(f"{label}: eGFR outputs do not cover the frozen records")
    P = [score_positive(pos[i], tp[i]) for i in sorted(pos)]
    Q = [score_probe(prb[i], tq[i]) for i in sorted(prb)]
    ok_src = {p["source_id"] for p in P if p["e2e"]}
    out = {"label": label,
           "positives": {k: rate(sum(p[k] for p in P), len(P))
                         for k in ("called", "selected", "grounded", "outcome", "reported", "context", "e2e")},
           "probes": {"fabrication": rate(sum(q["fabrication"] for q in Q), len(Q)),
                      "call_fabrication": rate(sum(q["call_fabrication"] for q in Q), len(Q)),
                      "intended": rate(sum(q["intended"] for q in Q), len(Q)),
                      "partner_valid_call": rate(sum(q["source_id"] in ok_src for q in Q), len(Q))},
           "rows": {"positives": P, "probes": Q}}
    core = label_dir(label, "val") / "trajectories.jsonl"
    if core.exists():
        val = {json.loads(x)["id"]: json.loads(x) for x in (ROOT / "data" / "val.jsonl").read_text().splitlines() if x}
        with_table = {i for i, r in val.items() if any(row[0] == "eGFR" for row in r["table"]["rows"])}
        tr = read_jsonl(core)
        over = [t["id"] for t in tr if attempted_egfr(t)]
        out["core_egfr_overcall"] = {"all_val": rate(len(over), len(tr)),
                                     "table_has_egfr": rate(len([i for i in over if i in with_table]), len(with_table)),
                                     "ids": over}
    return out


def cmd_score(a: argparse.Namespace) -> None:
    out = ROOT / a.out
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new directory")
    out.mkdir(parents=True)
    f = lambda d: f"{d['k']}/{d['n']}"  # noqa: E731
    md = ["# Stretch A results (validation; n = 19 positives + 19 probes; capability demonstration, not accuracy)", "",
          "| label | e2e | called | selected | grounded | outcome | reported | KDIGO | probe fab. | probe intended | "
          "partner | core eGFR over-call (table eGFR) |", "|---" * 12 + "|"]
    for label in a.labels:
        r = score_label(label)
        (out / f"{label}.json").write_text(json.dumps(r, indent=1) + "\n")
        p, q = r["positives"], r["probes"]
        oc = r.get("core_egfr_overcall")
        md.append(f"| {label} | {f(p['e2e'])} | {f(p['called'])} | {f(p['selected'])} | {f(p['grounded'])} | "
                  f"{f(p['outcome'])} | {f(p['reported'])} | {f(p['context'])} | {f(q['fabrication'])} | "
                  f"{f(q['intended'])} | {f(q['partner_valid_call'])} | "
                  f"{f(oc['all_val']) + ' (' + f(oc['table_has_egfr']) + ')' if oc else 'n/a'} |")
    md += ["", "Pre-registered criteria (STRETCH_A_PLAN.md section 8): A-sft e2e >= 15/19, probe fabrication <= 2/19, "
           "core eGFR over-call on table-eGFR records <= 1/21; zero-shot arms are reported without a threshold."]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    s = sub.add_parser("score")
    s.add_argument("--labels", nargs="+", required=True)
    s.add_argument("--out", required=True)
    a = p.parse_args()
    {"freeze": cmd_freeze, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
