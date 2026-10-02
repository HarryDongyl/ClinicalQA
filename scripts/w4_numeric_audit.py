"""Claim-level numeric audit and the pre-registered H2 decision (docs/EXPERIMENTS_WAVE4.md, Wave4-Q4).

    uv run python scripts/w4_numeric_audit.py packet --labels w3_relabel_lr1e4_s42_step000250 w4_q35_4b_relabel_lr1e4_step000250
    uv run python scripts/w4_numeric_audit.py merge  --raters reports/w4/numeric_audit/rater_A.jsonl reports/w4/numeric_audit/rater_B.jsonl \
                                                    [--adjudication reports/w4/numeric_audit/adjudication.jsonl]
    uv run python scripts/w4_numeric_audit.py h2     --a w3_relabel_lr1e4_s42_step000250 --b w4_q35_4b_relabel_lr1e4_step000250

packet  Blinded packet: the 50 val numeric answers of each label, shuffled together, with the frozen rubric. Items
        carry no model identity and no scorer verdict; the identity key is written separately (key.json, not for
        raters). Refuses to overwrite.
merge   Combines two independent rater files (item, label correct|wrong|disputed, claims, reason). Agreement and
        Cohen's kappa are reported. Disagreements need an adjudication row (the user, Wave4-Q4); "disputed" from
        either rater stays disputed unless adjudicated. Writes final_labels.jsonl.
h2      Paired exact two-sided sign test over items where both labels are resolved (disputed excluded) and differ.
        Pre-registered rule: "Qwen3.5 stronger on numeric" iff b-only correct > a-only correct and p < 0.05.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "w4" / "numeric_audit"
RUBRIC = """# Numeric claim-level audit rubric (frozen before labelling; docs/EXPERIMENTS_WAVE4.md Wave4-Q4)

Judge against the NOTE, TABLE and QUESTION; the reference answer may be wrong. For each answer, list every claim about
the patient's values that answers the question or is asserted along the way, and check each:

1. reference range: taken from the table/note/question when the input supplies one (a remembered external range is
   wrong when the input has one; the standard range is acceptable when the input has none);
2. analyte: the value named is the one the question asks about (or the correct one for "which/most/closest/other");
3. direction: above/below/within is correct for that value and range;
4. arithmetic: differences, ratios and percentages are right (reasonable rounding accepted, e.g. 2.8 vs 2.83);
5. units: correct or consistent;
6. ranking criterion: "most abnormal" etc. uses the scale the question names; if the question does not name one and
   absolute vs relative disagree for same-unit analytes, either defensible choice is accepted;
7. extra claims: an additional patient-value claim that is false makes the answer wrong (e.g. a wrong range status
   for another analyte), even if the requested part is right.

Label: correct (all requested parts right, no false patient-value claim), wrong (any requested part wrong or any false
patient-value claim), disputed (the question is under-specified or the input supports two readings; explain).
Self-contradiction: judge the final stated answer, but a contradicted requested claim is wrong.
Output one JSON line per item: {"item", "label", "claims": [{"claim", "check", "ok"}], "reason"}.
"""


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def cmd_packet(a: argparse.Namespace) -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; the audit packet is built once")
    val = {r["id"]: r for r in read_jsonl(ROOT / "data" / "val.jsonl")}
    ids = sorted(i for i, r in val.items() if r["answer_type"] == "numeric_reasoning")
    items = []
    for label in a.labels:
        if "test" in label:
            raise SystemExit("validation only")
        traj = {t["id"]: t for t in read_jsonl(ROOT / "outputs" / label / "val" / "trajectories.jsonl")}
        items += [(label, i, traj[i].get("final_answer")) for i in ids]
    random.Random(20261002).shuffle(items)
    OUT.mkdir(parents=True)
    (OUT / "rubric.md").write_text(RUBRIC, encoding="utf-8")
    packet, key = [], {}
    for n, (label, i, answer) in enumerate(items):
        r = val[i]
        item = f"n_{n:03d}"
        packet.append({"item": item, "question": r["question"], "note": r["note"], "table": r["table"],
                       "reference_answer_may_be_wrong": r["answer"], "model_final_answer": answer})
        key[item] = {"label": label, "id": i}
    (OUT / "packet.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in packet), encoding="utf-8")
    (OUT / "key.json").write_text(json.dumps({"labels": a.labels, "items": key}, indent=1) + "\n")
    print(json.dumps({"items": len(packet), "rubric_sha256": hashlib.sha256(RUBRIC.encode()).hexdigest(),
                      "packet": str((OUT / "packet.jsonl").relative_to(ROOT))}))


def kappa(x: list[str], y: list[str]) -> float | None:
    cats = sorted(set(x) | set(y))
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def cmd_merge(a: argparse.Namespace) -> None:
    r1, r2 = ({x["item"]: x for x in read_jsonl(ROOT / p)} for p in a.raters)
    if set(r1) != set(r2):
        raise SystemExit("rater files must label the same items")
    adj = {x["item"]: x for x in read_jsonl(ROOT / a.adjudication)} if a.adjudication else {}
    final, pending = [], []
    for item in sorted(r1):
        l1, l2 = r1[item]["label"], r2[item]["label"]
        if l1 == l2:
            label, source = l1, "agreed"
        elif item in adj:
            label, source = adj[item]["label"], "adjudicated"
        else:
            pending.append(item)
            continue
        final.append({"item": item, "label": label, "source": source})
    items = sorted(r1)
    report = {"n": len(items), "agreement": round(sum(r1[i]["label"] == r2[i]["label"] for i in items) / len(items), 3),
              "kappa": kappa([r1[i]["label"] for i in items], [r2[i]["label"] for i in items]),
              "disagreements": [i for i in items if r1[i]["label"] != r2[i]["label"]], "pending_adjudication": pending}
    (OUT / "agreement.json").write_text(json.dumps(report, indent=1) + "\n")
    if pending:
        print(json.dumps(report, indent=1))
        raise SystemExit(f"{len(pending)} disagreements need adjudication rows before final labels exist")
    (OUT / "final_labels.jsonl").write_text("".join(json.dumps(f) + "\n" for f in final))
    print(json.dumps({k: report[k] for k in ("n", "agreement", "kappa")}))


def sign_test(b_only: int, a_only: int) -> float:
    n = b_only + a_only
    if not n:
        return 1.0
    k = min(b_only, a_only)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def cmd_h2(a: argparse.Namespace) -> None:
    key = json.loads((OUT / "key.json").read_text())["items"]
    lab = {f["item"]: f["label"] for f in read_jsonl(OUT / "final_labels.jsonl")}
    by = {}
    for item, k in key.items():
        by.setdefault(k["id"], {})[k["label"]] = lab[item]
    resolved = {i: v for i, v in by.items() if v.get(a.a) in ("correct", "wrong") and v.get(a.b) in ("correct", "wrong")}
    b_only = [i for i, v in resolved.items() if v[a.b] == "correct" and v[a.a] == "wrong"]
    a_only = [i for i, v in resolved.items() if v[a.a] == "correct" and v[a.b] == "wrong"]
    p = sign_test(len(b_only), len(a_only))
    count = lambda name, want: sum(v.get(name) == want for v in by.values())  # noqa: E731
    out = {"a": a.a, "b": a.b, "n_items": len(by), "n_resolved_pairs": len(resolved),
           "disputed_items": sorted(i for i in by if i not in resolved),
           "correct": {a.a: count(a.a, "correct"), a.b: count(a.b, "correct")},
           "b_only_correct": sorted(b_only), "a_only_correct": sorted(a_only), "sign_test_p_two_sided": round(p, 4),
           "primary_criterion_met": len(b_only) > len(a_only) and p < 0.05,
           "rule": "b stronger on numeric iff b-only > a-only and exact two-sided sign test p < 0.05 (Wave4-Q4); "
                   "the non-inferiority checks (P1, grounded tool, partner calls, other assignment metrics) are "
                   "evaluated from the scored outputs and must also hold"}
    (OUT / "h2.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("correct", "n_resolved_pairs", "sign_test_p_two_sided",
                                          "primary_criterion_met")}))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("packet")
    k.add_argument("--labels", nargs=2, required=True)
    m = sub.add_parser("merge")
    m.add_argument("--raters", nargs=2, required=True)
    m.add_argument("--adjudication", default=None)
    h = sub.add_parser("h2")
    h.add_argument("--a", required=True)
    h.add_argument("--b", required=True)
    a = p.parse_args()
    {"packet": cmd_packet, "merge": cmd_merge, "h2": cmd_h2}[a.cmd](a)


if __name__ == "__main__":
    main()
