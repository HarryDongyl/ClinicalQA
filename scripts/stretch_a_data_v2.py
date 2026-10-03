"""Stretch A revision (D-100): stage-balanced, negative-rich calculate_egfr training rows.

    uv run python scripts/stretch_a_data_v2.py         # writes data/stretch_a_v2/ (refuses to overwrite)

Why (round-two diagnosis, docs/DECISIONS.md D-100): the first A-sft fabricated an age on 16/19 age-removed probes
(call probability pushed to about 0.5 while positives-vs-probes AUROC stayed 0.98: a threshold shift from 40 positives
against 6 age negatives) and mapped eGFR to KDIGO categories badly (G3a and G5 had 3 and 1 training examples).

Construction, train notes only (the 164 eligible sources of stretch_a_data.eligible, seed 42):
  positives  120: every G3a/G3b/G4/G5 source (62) plus 29 G1 and 29 G2. The answer states the KDIGO range before the
             category ("22 mL/min/1.73m², which is in the 15–29 range (KDIGO GFR category G4, ...)").
  age        60: 40 paired with positives (same note, age removed; stratified by category) + 20 unpaired sources.
  sex        20: unpaired sources, sex removed.
Questions: the four original templates (they presuppose age and sex; the frozen evaluation uses them) plus four
neutral ones, assigned round-robin within each kind so wording does not predict calling. Negative answers state what
is documented, with values, and what is missing. The evaluation sets (configs/w4/egfr_*.json) are unchanged.
Every positive is recomputed with an independent CKD-EPI implementation. Sources whose edited note keeps an age or sex
cue (v1 residue checks) are skipped and the next source is used: paired age negatives stay within their category.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stretch_a_data as v1  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "stretch_a_v2"
SEED = 42
STAGES = ("G1", "G2", "G3a", "G3b", "G4", "G5")
CAP_G1_G2 = 29
N_PAIRED_AGE, N_UNPAIRED_AGE, N_SEX = 40, 20, 20
QUESTIONS = v1.QUESTIONS + [
    "What is this patient's estimated GFR, and which KDIGO category does it fall in?",
    "Estimate this patient's kidney function with the CKD-EPI equation.",
    "What eGFR does this patient's renal function correspond to, and what is the KDIGO GFR category?",
    "Assess this patient's estimated glomerular filtration rate and its KDIGO category.",
]
BAND = {"G1": "which is at or above 90", "G2": "which is in the 60–89 range", "G3a": "which is in the 45–59 range",
        "G3b": "which is in the 30–44 range", "G4": "which is in the 15–29 range", "G5": "which is below 15"}


def independent_egfr(cr: float, age: int, sex: str) -> int:
    """Second CKD-EPI 2021 implementation (piecewise form) used only to check the generator."""
    if sex == "female":
        k, a, f = 0.7, -0.241, 1.012
    else:
        k, a, f = 0.9, -0.302, 1.0
    x = cr / k
    raw = 142 * (x ** a if x < 1 else x ** -1.200) * 0.9938 ** age * f
    return int(Decimal(repr(raw)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def positive(e: dict, rid: str, q: str) -> dict:
    egfr = v1.ckd_epi_2021(e["cr"], e["age"], e["sex"])
    stage = v1.g_stage(egfr)
    answer = (f"Using the patient's serum creatinine of {e['cr']} mg/dL, age {e['age']} and sex ({e['sex']}), the "
              f"estimated GFR (CKD-EPI 2021) is {egfr} mL/min/1.73m², {BAND[stage]} (KDIGO GFR category {stage}, "
              f"assuming stable kidney function).")
    return {"id": rid, "source_id": e["src"]["id"], "note": e["src"]["note"], "table": e["src"]["table"], "question": q,
            "answer": answer, "answer_type": "tool_call",
            "tool_calls": [{"tool": "calculate_egfr",
                            "arguments": {"creatinine_mg_dl": e["cr"], "age": e["age"], "sex": e["sex"]},
                            "result": egfr}],
            "stretch_a": {"kind": "positive", "egfr": egfr, "stage": stage}}


def negative(e: dict, rid: str, q: str, missing: str, paired: bool) -> dict:
    r = v1.negative(e, rid, q, missing)
    documented = f"sex ({e['sex']})" if missing == "age" else f"age ({e['age']})"
    r["answer"] = (f"The serum creatinine ({e['cr']} mg/dL) and {documented} are documented, but the patient's {missing} "
                   f"is not recorded, so eGFR cannot be calculated with CKD-EPI.")
    egfr = v1.ckd_epi_2021(e["cr"], e["age"], e["sex"])
    r["stretch_a"].update({"paired": paired, "source_stage": v1.g_stage(egfr)})
    return r


def build() -> dict[str, list[dict]]:
    rng = random.Random(SEED)
    pool = v1.eligible("train")
    by = {s: [] for s in STAGES}
    for e in pool:
        by[v1.g_stage(v1.ckd_epi_2021(e["cr"], e["age"], e["sex"]))].append(e)
    for s in STAGES:
        rng.shuffle(by[s])
    chosen = {s: by[s] if s not in ("G1", "G2") else by[s][:CAP_G1_G2] for s in STAGES}
    rest = [e for s in ("G1", "G2") for e in by[s][CAP_G1_G2:]]
    rng.shuffle(rest)
    pos_src = [e for s in STAGES for e in chosen[s]]
    # Paired age negatives: stratified by category, proportional to the positives (largest remainder).
    quota = {s: len(chosen[s]) * N_PAIRED_AGE / len(pos_src) for s in STAGES}
    take = {s: int(quota[s]) for s in STAGES}
    for s in sorted(STAGES, key=lambda s: quota[s] - take[s], reverse=True)[:N_PAIRED_AGE - sum(take.values())]:
        take[s] += 1
    def clean(e: dict, missing: str) -> bool:  # skip sources whose edited note keeps a cue (v1 residue checks)
        r = v1.negative(e, "x", QUESTIONS[0], missing)["stretch_a"]
        return not (r["residue"] or r["extractor_still_finds"])

    paired_src = [e for s in STAGES for e in [x for x in chosen[s] if clean(x, "age")][:take[s]]]
    age_src = [e for e in rest if clean(e, "age")][:N_UNPAIRED_AGE]
    sex_src = [e for e in rest if e not in age_src and clean(e, "sex")][:N_SEX]
    if len(paired_src) < N_PAIRED_AGE or len(age_src) < N_UNPAIRED_AGE or len(sex_src) < N_SEX:
        raise SystemExit("not enough residue-free sources")
    q = lambda i: QUESTIONS[i % len(QUESTIONS)]  # noqa: E731  round-robin within each kind
    rng.shuffle(pos_src)
    pos = [positive(e, f"sa2_train_{i:03d}", q(i)) for i, e in enumerate(pos_src)]
    n = len(pos)
    age = [negative(e, f"sa2_train_{n + i:03d}", q(i), "age", True) for i, e in enumerate(paired_src)]
    age += [negative(e, f"sa2_train_{n + len(paired_src) + i:03d}", q(len(paired_src) + i), "age", False)
            for i, e in enumerate(age_src)]
    sex = [negative(e, f"sa2_train_{n + len(age) + i:03d}", q(i), "sex", False) for i, e in enumerate(sex_src)]
    return {"positive": pos, "negative_age": age, "negative_sex": sex}


def checks(parts: dict[str, list[dict]]) -> dict:
    rows = [r for v in parts.values() for r in v]
    mism = [r["id"] for r in parts["positive"]
            if independent_egfr(*(r["tool_calls"][0]["arguments"][k] for k in ("creatinine_mg_dl", "age", "sex")))
            != r["tool_calls"][0]["result"]]
    flagged = [r["id"] for r in rows if r["stretch_a"].get("residue") or r["stretch_a"].get("extractor_still_finds")]
    ids = [r["id"] for r in rows]
    return {
        "independent_egfr_mismatches": mism, "residue_flagged": flagged, "duplicate_ids": len(ids) - len(set(ids)),
        "distinct_sources": len({r["source_id"] for r in rows}),
        "positive_stages": dict(Counter(r["stretch_a"]["stage"] for r in parts["positive"])),
        "age_negative_source_stages": dict(Counter(r["stretch_a"]["source_stage"] for r in parts["negative_age"])),
        "paired_age_negatives": sum(r["stretch_a"]["paired"] for r in parts["negative_age"]),
        "questions_by_kind": {k: dict(Counter(QUESTIONS.index(r["question"]) for r in v)) for k, v in parts.items()},
        "soft_age_mentions": sum(bool(r["stretch_a"].get("soft_age_mentions")) for r in parts["negative_age"]),
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; generated data is never overwritten")
    parts = build()
    report = checks(parts)
    if report["independent_egfr_mismatches"] or report["residue_flagged"] or report["duplicate_ids"]:
        raise SystemExit(f"generator checks failed: {json.dumps(report)}")
    OUT.mkdir(parents=True)
    rows = parts["positive"] + parts["negative_age"] + parts["negative_sex"]
    path = OUT / "train_additions.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    manifest = {"seed": SEED, "formula": "CKD-EPI 2021 race-free, integer half-up", "decision": "D-100",
                "counts": {k: len(v) for k, v in parts.items()}, "eligible_pool": {"train": len(v1.eligible("train"))},
                "checks": report, "questions": QUESTIONS,
                "sha256": {"train_additions.jsonl": hashlib.sha256(path.read_bytes()).hexdigest()}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("counts", "checks")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
