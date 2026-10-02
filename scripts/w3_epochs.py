"""Wave-three epoch checkpoints (D-076): resolve and hash-check, or generate on val/P1/train-fit.

    uv run python scripts/w3_epochs.py ckpt --run w3_relabel_lr1e4_s42 --epoch 2      prints checkpoint path
    uv run python scripts/w3_epochs.py generate --run w3_relabel_lr1e4_s42 --config configs/eval_w3_v1.yaml

generate: both epoch checkpoints on val (labels <run>_step<N>); epoch two only (the prospective primary
endpoint) also on the frozen P1 probes and, with --trainfit, on the D-TRAINFIT IDs. Epoch one is diagnostic.
--p1-all-epochs also runs P1 on epoch one (wave four, Wave4-Q1: diagnostic only, never used for selection).
--records-files adds further frozen probe files on epoch two (e.g. Stretch A eGFR sets). No selection is written.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from clinqa.run_info import adapter_sha256  # noqa: E402
from w2_epochs import epoch_checkpoints  # noqa: E402

P1 = "configs/w3/p1_probes.json"
TRAINFIT = "configs/w3/trainfit_ids.json"


def checkpoint(run: str, epoch: int) -> tuple[int, str]:
    step, path, sha = epoch_checkpoints(run)[epoch - 1]
    if adapter_sha256(ROOT / path) != sha:
        raise SystemExit(f"{path}: adapter hash differs from outputs/{run}/manifest.json")
    return step, path


def evaluate(config: str, run: str, label: str, adapter: str, *extra: str) -> None:
    cmd = [sys.executable, "-m", "clinqa.evaluate", "--config", config, "generate", "--run", run,
           "--label", label, "--adapter", adapter, *extra]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("ckpt")
    c.add_argument("--run", required=True)
    c.add_argument("--epoch", type=int, choices=(1, 2), required=True)
    g = sub.add_parser("generate")
    g.add_argument("--run", required=True)
    g.add_argument("--config", required=True)
    g.add_argument("--trainfit", action="store_true")
    g.add_argument("--p1-all-epochs", action="store_true")
    g.add_argument("--records-files", nargs="*", default=[])
    a = p.parse_args()
    if a.cmd == "ckpt":
        print(checkpoint(a.run, a.epoch)[1])
        return
    for epoch in (1, 2):
        step, path = checkpoint(a.run, epoch)
        label = f"{a.run}_step{step:06d}"
        evaluate(a.config, a.run, label, path, "--split", "val")
        if epoch == 2 or a.p1_all_epochs:
            evaluate(a.config, a.run, label, path, "--split", "val", "--records-file", P1)
        if epoch == 2:
            for extra in a.records_files:
                evaluate(a.config, a.run, label, path, "--split", "val", "--records-file", extra)
            if a.trainfit:
                evaluate(a.config, a.run, label, path, "--split", "train", "--ids-file", TRAINFIT)


if __name__ == "__main__":
    main()
