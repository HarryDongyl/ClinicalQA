"""Stratify the validation numeric items before the claim-level audit (INTERVIEW_PREP.md sections 5.9 and 5.13).

    uv run python scripts/numeric_audit_prep.py --keys reports/w3/results_2026-10-02/v21/keys.jsonl \
        --scored reports/w3/results_2026-10-02/v21 --labels w3_c_filtered_s42 w3_relabel_lr1e4_s42_step000250 \
        --out reports/w3/numeric_audit_prep

Strata (from the input-derived v2.1 keys; computed without reading any prediction):
  key_agrees_gold   every structured check agrees with gold        -> high-confidence reference; audit still checks
                                                                       extra false claims
  key_disputes_gold at least one structured check disagrees        -> pre-marked "disputed" for the audit
  open_or_partial   text fallback, unscored clauses or no agreement   -> reference must be built by the auditors
                    signal
The per-model v2.1 verdicts are attached for audit bookkeeping only; auditors must not see them (blinding).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def stratum(key: dict) -> str:
    agrees = [c["gold_agrees"] for c in key["checks"] if c["kind"] != "text" and c.get("gold_agrees") is not None]
    if any(a is False for a in agrees):
        return "key_disputes_gold"
    if key["coverage"] == "structured" and agrees and all(agrees):
        return "key_agrees_gold"
    return "open_or_partial"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--keys", required=True)
    p.add_argument("--scored", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = ROOT / a.out
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new directory")
    out.mkdir(parents=True)
    keys = [k for k in read_jsonl(ROOT / a.keys) if k["answer_type"] == "numeric_reasoning"]
    verdicts = {lab: {s["id"]: s["correct"] for s in read_jsonl(ROOT / a.scored / f"{lab}.scored.jsonl")}
                for lab in a.labels}
    rows = [{"id": k["id"], "stratum": stratum(k), "coverage": k["coverage"],
             "checks": [c["kind"] for c in k["checks"]],
             "v21_verdict_not_for_auditors": {lab: verdicts[lab].get(k["id"]) for lab in a.labels}} for k in keys]
    (out / "strata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    counts = Counter(r["stratum"] for r in rows)
    md = ["# Numeric audit stratification (validation, 50 items)", "",
          "Computed from input-derived v2.1 keys only. The v2.1 verdicts are bookkeeping and must be hidden from auditors.",
          "", "| stratum | n | " + " | ".join(f"v2.1 pass {lab}" for lab in a.labels) + " |",
          "|---|---|" + "---|" * len(a.labels)]
    for s in ("key_agrees_gold", "key_disputes_gold", "open_or_partial"):
        sub = [r for r in rows if r["stratum"] == s]
        md.append(f"| {s} | {len(sub)} | " + " | ".join(
            f"{sum(bool(r['v21_verdict_not_for_auditors'][lab]) for r in sub)}/{len(sub)}" for lab in a.labels) + " |")
    md += ["", "- " + ", ".join(r["id"] for r in rows if r["stratum"] == "key_disputes_gold") + " (disputed)"]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
