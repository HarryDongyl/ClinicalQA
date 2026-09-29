#!/usr/bin/env python3
"""Export source, metrics and selected adapters; use --partial for pre-final backups."""
import argparse
import json
from pathlib import Path
import subprocess
import zipfile

from package_runpod import ROOT, DIRECTORIES, FILES, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    final_path = ROOT / "outputs/final_selection.json"
    selected = []
    if not args.partial:
        if not final_path.exists():
            parser.error("no frozen selection; use --partial for an intermediate backup")
        final = json.loads(final_path.read_text())
        for entry in final["runs"]:
            if not (ROOT / "outputs" / entry["label"] / "test/metrics.json").exists():
                parser.error("test incomplete; use --partial for an intermediate backup")
            if entry["adapter"]:
                selected.append(ROOT / entry["adapter"])
    for p in (ROOT / "outputs").glob("*/selection.json"):
        selection = json.loads(p.read_text())
        selected.append(ROOT / selection["checkpoints"][selection["selected"]])
    files = {ROOT / name for name in FILES}
    for name in (*DIRECTORIES, "outputs"):
        files.update((ROOT / name).rglob("*"))
    for checkpoint in selected:
        if not checkpoint.resolve().is_relative_to(ROOT / "checkpoints"):
            parser.error("selected checkpoint is outside project; copy it into checkpoints first")
        # Adapters/tokenizer only; optimizer recovery remains on the persistent volume.
        files.update(p for p in checkpoint.iterdir() if p.suffix in {".json", ".safetensors", ".txt", ".jinja"})
    files = sorted(p for p in files if p.is_file() and not p.is_symlink()
                   and not {"__pycache__", ".DS_Store", "tb"}.intersection(p.parts) and p.suffix != ".pyc")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, Path("Clinical") / p.relative_to(ROOT))
        manifest = {str(p.relative_to(ROOT)): digest(p) for p in files}
        z.writestr("Clinical/EXPORT_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        z.writestr("Clinical/EXPORT_GIT_REVISION.txt", git.stdout)
    print(f"Saved {args.output}; SHA256 {digest(args.output)}")
    print("Includes inference adapters, not optimizer recovery. Keep the volume or separately archive checkpoints/ to resume training.")


if __name__ == "__main__":
    main()
