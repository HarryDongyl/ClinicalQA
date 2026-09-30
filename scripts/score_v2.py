"""Rescore saved trajectories with scorer v2 (CPU only; no generation).

    uv run python scripts/score_v2.py --runs-dir outputs --split val --out reports/scorer_v2/val \
        --labels base raw_lr1e4_step000125 ...

Writes keys.jsonl (answer keys, built without predictions), <label>.scored.jsonl, summary.json and
summary.md with v1 vs v2 accuracy per answer type (Wilson 95% CI), per-check-kind accuracy and the
tool / abstention funnels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from clinqa import scorer_v2 as s2
from clinqa.config import PROJECT_ROOT
from clinqa.evaluate import wilson

TYPES = ("extractive", "numeric_reasoning", "tool_call", "uncertain")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rate(rows: list[dict], key: str = "correct") -> dict:
    k = sum(bool(r.get(key)) for r in rows)
    return {"k": k, "n": len(rows), "rate": round(k / len(rows), 4) if rows else None, "ci95": wilson(k, len(rows))}


def aggregate(scores: list[dict], v1: dict[str, dict]) -> dict:
    by = defaultdict(list)
    for s in scores:
        by[s["answer_type"]].append(s)
    out = {t: {"v2": _rate(by[t]), "v1": _rate([v1[s["id"]] for s in by[t]])} for t in TYPES}
    kinds: dict[str, list[bool]] = defaultdict(list)
    for s in scores:
        for c in s.get("checks", []):
            kinds[c["kind"]].append(c["ok"])
    out["check_kinds"] = {k: {"k": sum(v), "n": len(v), "rate": round(sum(v) / len(v), 4)} for k, v in kinds.items()}
    tool = by["tool_call"]
    call = [s for s in tool if s["expected"] == "call"]
    out["tool"] = {
        "expected_call": _rate(call), "expected_abstain_q5": _rate([s for s in tool if s["expected"] == "abstain"]),
        "relevance": _rate(tool, "relevance_ok"),
        "name_ok|called": _rate([s for s in call if s.get("name_ok") is not None], "name_ok"),
        "outcome_ok|called": _rate([s for s in call if s.get("outcome_ok") is not None], "outcome_ok"),
        "args_exact_v1|called": _rate([s for s in call if s.get("args_exact_v1") is not None], "args_exact_v1"),
        "parse_error": _rate(tool, "parse_error"),
    }
    answer = [s for s in scores if s["answer_type"] in ("extractive", "numeric_reasoning")]
    out["over_refusal"] = _rate(answer, "over_refusal") if answer else None
    out["over_call"] = _rate([s for s in scores if s["answer_type"] != "tool_call"], "called")
    out["macro"] = round(sum(out[t]["v2"]["rate"] for t in TYPES) / 4, 4)
    out["macro_v1"] = round(sum(out[t]["v1"]["rate"] for t in TYPES) / 4, 4)
    out["errors"] = {t: dict(Counter(s["error"] for s in by[t] if not s["correct"]).most_common()) for t in TYPES}
    out["coverage"] = dict(Counter(s["coverage"] for s in scores))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    records = {r["id"]: r for r in _jsonl(PROJECT_ROOT / "data" / f"{a.split}.jsonl")}
    keys = {i: s2.build_key(r) for i, r in records.items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "keys.jsonl").write_text("".join(json.dumps(k.as_dict(), ensure_ascii=False) + "\n" for k in keys.values()),
                                    encoding="utf-8")
    summary = {"scorer": s2.VERSION, "split": a.split, "runs": {}}
    for label in a.labels:
        d = Path(a.runs_dir) / label / a.split
        trajs = {t["id"]: t for t in _jsonl(d / "trajectories.jsonl")}
        v1 = {s["id"]: s for s in _jsonl(d / "scored.jsonl")}
        scores = [s2.score(records[i], trajs[i], keys[i]) for i in records if i in trajs]
        for s in scores:
            s["v1_correct"] = v1[s["id"]]["correct"]
        (out / f"{label}.scored.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in scores),
                                                   encoding="utf-8")
        summary["runs"][label] = aggregate(scores, v1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    L = [f"# Scorer v2 rescore ({a.split})", "", "| run | extr v1→v2 | num v1→v2 | tool v1→v2 | unc v1→v2 | macro v1→v2 |",
         "|---|---|---|---|---|---|"]
    for label, m in summary["runs"].items():
        cells = [f"{m[t]['v1']['rate']:.3f}→**{m[t]['v2']['rate']:.3f}**" for t in TYPES]
        L.append(f"| {label} | " + " | ".join(cells) + f" | {m['macro_v1']:.3f}→**{m['macro']:.3f}** |")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
