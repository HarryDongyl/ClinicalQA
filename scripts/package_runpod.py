#!/usr/bin/env python3
"""Build a source/data upload ZIP or verify its extracted manifest. No ML dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("src", "tests", "configs", "docs", "reports", "data", "_data", "scripts")
FILES = ("README.md", "Makefile", "pyproject.toml", "requirements.txt", "uv.lock", ".python-version", ".gitignore")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        manifest = json.loads(args.verify.read_text())
        for name, expected in manifest.items():
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or digest(path) != expected:
                raise SystemExit(f"Upload verification failed: {name}")
        print(f"Verified {len(manifest)} uploaded files")
        return
    if not args.output:
        parser.error("provide --output or --verify")
    paths = [ROOT / name for name in FILES]
    for name in DIRECTORIES:
        paths.extend((ROOT / name).rglob("*"))
    paths = sorted(p for p in paths if p.is_file() and not p.is_symlink()
                   and not {"__pycache__", ".DS_Store", ".pytest_cache"}.intersection(p.parts)
                   and p.suffix not in {".pyc", ".key"} and p.name != ".env")
    manifest = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, Path("Clinical") / p.relative_to(ROOT))
        z.writestr("Clinical/UPLOAD_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output}: {len(paths)} files, {args.output.stat().st_size:,} bytes")
    print("SHA256", digest(args.output))


if __name__ == "__main__":
    main()
