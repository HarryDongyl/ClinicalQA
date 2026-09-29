"""Raw data preparation and loading.

`prepare_data` copies the provided files from `_data/` into `data/` byte-for-byte
(originals are never modified), validates every line as a JSON object, checks split
sizes, and writes a deterministic MANIFEST.json with SHA-256 checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

from clinqa.config import load_yaml, resolve
from clinqa.validation import validate_records

SPLITS = ("train", "val", "test")
DEFAULT_CONFIG = "configs/data.yaml"


class DataPreparationError(RuntimeError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key {key!r}")
        obj[key] = value
    return obj


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number {value}")
    return number


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _validate(records: list[dict[str, Any]], split: str) -> None:
    try:
        validate_records(records, split)
    except ValueError as error:
        raise DataPreparationError(str(error)) from error


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with resolve(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line, object_pairs_hook=_unique_object,
                                 parse_float=_finite_float, parse_constant=_reject_constant)
            except ValueError as e:
                raise DataPreparationError(f"{path}:{lineno}: invalid JSON ({e})") from e
            if not isinstance(obj, dict):
                raise DataPreparationError(f"{path}:{lineno}: expected a JSON object")
            records.append(obj)
    return records


def _copy_one(src: Path, dst: Path, force: bool) -> str:
    """Copy src to dst unless an identical file already exists. Returns the action taken."""
    src_hash = sha256_file(src)
    if dst.exists():
        if sha256_file(dst) == src_hash:
            return "unchanged"
        if not force:
            raise DataPreparationError(
                f"{dst} exists with different content; re-run with --force to overwrite"
            )
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    if sha256_file(dst) != src_hash:
        raise DataPreparationError(f"checksum mismatch after copying {src} -> {dst}")
    return "copied"


def _display(path: Path) -> str:
    """Project-relative path when possible, so the manifest is machine-independent."""
    try:
        return str(path.relative_to(resolve(".")))
    except ValueError:
        return str(path)


def prepare_data(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    source_dir = resolve(config["source_dir"])
    output_dir = resolve(config["output_dir"])

    entries: dict[str, dict[str, Any]] = {}
    items = [(name, spec) for name, spec in config["splits"].items()]
    items.append(("reference", config["reference"]))

    # Validate every source and destination before writing any split. A bad val
    # file must not leave a partially refreshed train/test snapshot.
    prepared = []
    for name, spec in items:
        src = source_dir / spec["source"]
        dst = output_dir / spec["target"]
        if not src.exists():
            raise DataPreparationError(f"missing source file: {src}")
        records = read_jsonl(src)
        _validate(records, name)
        n_records = len(records)
        expected = spec.get("expected_count")
        if expected is not None and n_records != expected:
            raise DataPreparationError(f"{name}: expected {expected} records, found {n_records}")
        if dst.exists() and not force and sha256_file(dst) != sha256_file(src):
            raise DataPreparationError(f"{dst} exists with different content; re-run with --force to overwrite")
        prepared.append((name, src, dst, n_records))

    for name, src, dst, n_records in prepared:
        action = _copy_one(src, dst, force)
        entries[name] = {
            "source": _display(src),
            "target": _display(dst),
            "sha256": sha256_file(dst),
            "bytes": dst.stat().st_size,
            "n_records": n_records,
            "action": action,
        }

    manifest = {
        "description": "Byte-identical copies of the provided data files; originals in source_dir are never modified.",
        "files": {k: {kk: vv for kk, vv in v.items() if kk != "action"} for k, v in entries.items()},
    }
    manifest_path = resolve(config["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"manifest": manifest, "actions": {k: v["action"] for k, v in entries.items()}}


def split_path(split: str, config_path: str = DEFAULT_CONFIG) -> Path:
    cfg = load_yaml(config_path)
    return resolve(cfg["output_dir"]) / cfg["splits"][split]["target"]


def load_split(split: str, config_path: str = DEFAULT_CONFIG) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    records = read_jsonl(split_path(split, config_path))
    _validate(records, split)
    return records


def load_reference(config_path: str = DEFAULT_CONFIG) -> list[dict[str, Any]]:
    cfg = load_yaml(config_path)
    records = read_jsonl(resolve(cfg["output_dir"]) / cfg["reference"]["target"])
    _validate(records, "reference")
    return records


def verify_manifest(config_path: str = DEFAULT_CONFIG) -> dict[str, Any]:
    """Verify live source/canonical bytes before trusting audit provenance."""
    cfg = load_yaml(config_path)
    manifest = json.loads(resolve(cfg["manifest"]).read_text(encoding="utf-8"))
    specs = {**cfg["splits"], "reference": cfg["reference"]}
    for name, spec in specs.items():
        info = manifest["files"][name]
        target = resolve(cfg["output_dir"]) / spec["target"]
        source = resolve(cfg["source_dir"]) / spec["source"]
        if any(sha256_file(path) != info["sha256"] for path in (source, target)):
            raise DataPreparationError(f"{name}: manifest checksum mismatch; input snapshot changed")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copy and validate the provided data files into data/.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true", help="overwrite targets whose content differs")
    args = parser.parse_args(argv)
    try:
        result = prepare_data(load_yaml(args.config), force=args.force)
    except DataPreparationError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for name, info in result["manifest"]["files"].items():
        print(f"{name:9s} {info['n_records']:5d} records  {result['actions'][name]:9s}  {info['target']}  sha256={info['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
