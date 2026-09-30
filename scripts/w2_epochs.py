"""Wave-two epoch-checkpoint validation without the legacy selector (DECISIONS D-045, D-049).

    uv run python scripts/w2_epochs.py generate --run w2_filtered_lr1e4_mb1 --prompt v1
    uv run python scripts/w2_epochs.py stable --run w2_filtered_lr1p5e4_mb1

generate  evaluates every epoch-end checkpoint in outputs/<run>/manifest.json on val with
          configs/eval_w2_prompt_<prompt>.yaml. Labels: <run>_step<N> (v1) or w2p_<prompt>_<run>_step<N>.
          Each checkpoint's adapter hash must match the manifest. Existing labels are reused only when
          the protocol and adapter match (clinqa.evaluate generate). No selection is written: checkpoint
          choice waits for the scorer gate (D-049), so `make epochs` (legacy select) is not used here.
stable    exits 1 if any logged loss/grad_norm is non-finite or the run did not finish; the only
          criterion for skipping the 2e-4 arm (D-045).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinqa.run_info import adapter_sha256  # noqa: E402

PROMPTS = ("v1", "v2")


def epoch_checkpoints(run: str) -> list[tuple[int, str, str]]:
    manifest = json.loads((ROOT / "outputs" / run / "manifest.json").read_text(encoding="utf-8"))
    out = []
    for ck in manifest["checkpoints"]:
        if not ck.get("epoch_end", abs(ck["epoch"] - round(ck["epoch"])) < 1e-6):
            continue
        parts = Path(ck["path"]).parts
        # Manifests record the pod's absolute path; resolve it inside this checkout.
        rel = Path(*parts[parts.index("checkpoints"):]) if "checkpoints" in parts else Path(ck["path"])
        out.append((ck["step"], str(rel), ck["adapter_sha256"]))
    if len(out) != 2:
        raise SystemExit(f"{run}: expected two epoch-end checkpoints, found {len(out)}")
    return out


def label(run: str, prompt: str, step: int) -> str:
    base = f"{run}_step{step:06d}"
    return base if prompt == "v1" else f"w2p_{prompt}_{base}"


def cmd_generate(run: str, prompt: str) -> None:
    config = f"configs/eval_w2_prompt_{prompt}.yaml"
    for step, path, sha in epoch_checkpoints(run):
        if adapter_sha256(ROOT / path) != sha:
            raise SystemExit(f"{path}: adapter hash differs from outputs/{run}/manifest.json")
        cmd = [sys.executable, "-m", "clinqa.evaluate", "--config", config, "generate", "--run", run,
               "--split", "val", "--label", label(run, prompt, step), "--adapter", path]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


def cmd_stable(run: str) -> None:
    d = ROOT / "outputs" / run
    if not (d / "manifest.json").exists() or not (d / "train_log.jsonl").exists():
        raise SystemExit(f"{run}: no manifest or train log; training did not finish")
    rows = [json.loads(line) for line in (d / "train_log.jsonl").read_text().splitlines() if line]
    bad = [r.get("step") for r in rows for k in ("loss", "grad_norm", "train_loss")
           if k in r and not math.isfinite(float(r[k]))]
    if bad:
        raise SystemExit(f"{run}: non-finite loss/grad_norm at steps {sorted(set(bad))}")
    print(f"{run}: all logged loss/grad_norm finite ({len(rows)} log rows)")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--run", required=True)
    g.add_argument("--prompt", choices=PROMPTS, default="v1")
    s = sub.add_parser("stable")
    s.add_argument("--run", required=True)
    a = p.parse_args()
    cmd_generate(a.run, a.prompt) if a.cmd == "generate" else cmd_stable(a.run)


if __name__ == "__main__":
    main()
