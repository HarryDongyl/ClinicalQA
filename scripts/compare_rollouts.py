"""Item-level comparison of two saved validation rollouts (no scoring, no GPU).

Uses:
  determinism  re-generated v1 arm vs the wave-one output of the same weights and prompt
               (e.g. w2p_v1_raw_lr1e4 vs raw_lr1e4_step000125)
  ablation     which items change between prompt arms (e.g. w2p_v1_base vs w2p_v2_base)

Compares raw assistant turns, parsed calls, final answers and stop reasons per record id, and writes
a JSON summary plus the differing ids. Refuses test labels, like rescore_validation_v2.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load(label: str, split: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if "test" in label or split != "val":
        raise SystemExit(f"refusing non-validation input: {label}/{split}")
    d = ROOT / "outputs" / label / split
    rows = [json.loads(line) for line in (d / "trajectories.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return {r["id"]: r for r in rows}, json.loads((d / "run.json").read_text(encoding="utf-8"))


def view(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw": [t["raw"] for t in r["turns"]],
        "calls": [c for t in r["turns"] for c in t.get("calls", [])],
        "final_answer": r.get("final_answer"),
        "stop_reason": r.get("stop_reason"),
        "pre_call_text": [t["content"] for t in r["turns"] if t.get("calls") and t.get("content")],
    }


def protocol_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    pa, pb = a.get("protocol", {}), b.get("protocol", {})
    out = {k: [pa.get(k), pb.get(k)] for k in sorted(set(pa) | set(pb))
           if k != "source_sha256" and pa.get(k) != pb.get(k)}
    sa, sb = pa.get("source_sha256", {}), pb.get("source_sha256", {})
    changed = sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))
    if changed:
        out["source_sha256_changed"] = changed
    if a.get("adapter_sha256") != b.get("adapter_sha256"):
        out["adapter_sha256"] = [a.get("adapter_sha256"), b.get("adapter_sha256")]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", help="write JSON summary here (default: print only)")
    args = ap.parse_args(argv)
    ra, ma = load(args.a, args.split)
    rb, mb = load(args.b, args.split)
    if set(ra) != set(rb):
        raise SystemExit(f"record ids differ: {len(set(ra) ^ set(rb))} not shared")
    fields = ["raw", "calls", "final_answer", "stop_reason"]
    diff: dict[str, list[str]] = {f: [] for f in fields}
    for i in sorted(ra):
        va, vb = view(ra[i]), view(rb[i])
        for f in fields:
            if va[f] != vb[f]:
                diff[f].append(i)
    n = len(ra)
    summary = {
        "a": args.a, "b": args.b, "split": args.split, "n": n,
        "protocol_differences": protocol_diff(ma, mb),
        "identical_items": n - len(set().union(*diff.values())),
        "differing": {f: {"count": len(ids), "ids": ids} for f, ids in diff.items()},
        "calls_made": [sum(bool(view(r)["calls"]) for r in ra.values()),
                       sum(bool(view(r)["calls"]) for r in rb.values())],
        "pre_call_text_turns": [sum(bool(view(r)["pre_call_text"]) for r in ra.values()),
                                sum(bool(view(r)["pre_call_text"]) for r in rb.values())],
    }
    text = json.dumps(summary, indent=2)
    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(f"{args.a} vs {args.b}: {summary['identical_items']}/{n} identical; "
          + ", ".join(f"{f} differs {len(ids)}" for f, ids in diff.items()))
    print("protocol differences:", json.dumps(summary["protocol_differences"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
