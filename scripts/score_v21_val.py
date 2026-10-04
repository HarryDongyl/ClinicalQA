"""Guarded wrapper around scripts/score_v2.py (candidate scorer v2.1; DECISIONS D-040, D-047).

    uv run python scripts/score_v21_val.py --out reports/<new dir>/v21 --labels w2p_v1_base ...

score_v2.py accepts --split test, overwrites its output directory and silently scores the intersection
of IDs. This wrapper refuses all three: validation only, a new output directory, and every label must
hold exactly the 250 validation IDs. Output is diagnostic; v2.1 is not approved for selection.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ids(path: Path) -> set[str]:
    return {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scorer", default="2.1", choices=["2.1", "2.2"])
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; scorer reports are never overwritten, choose a new directory")
    expected = ids(ROOT / "data" / "val.jsonl")
    for label in a.labels:
        if "test" in label:
            raise SystemExit(f"refusing test-like label: {label}")
        d = ROOT / "outputs" / label / "val"
        for name in ("trajectories.jsonl", "scored.jsonl"):
            if not (d / name).exists() or ids(d / name) != expected:
                raise SystemExit(f"{label}: {name} missing or does not cover the {len(expected)} validation IDs")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "score_v2.py"), "--runs-dir", str(ROOT / "outputs"),
                    "--split", "val", "--out", str(out), "--scorer", a.scorer, "--labels", *a.labels], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
