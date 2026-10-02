"""Wave-four confirmatory test run (docs/EXPERIMENTS_WAVE4.md step 6; D-030/D-036 safeguards, option B).

    uv run python scripts/w4_test.py freeze --entry LABEL EVAL_CONFIG RUN ADAPTER [--entry ...]   # writes configs/w4/final_test.json
    uv run python scripts/w4_test.py run [--rerun-reason "..."]                                     # GPU; clean committed tree only

freeze records, per model: eval config path + sha256 + protocol() digest, run, adapter path + adapter sha256, and the
pre-specified report (natural Q5 fabrication on test, grounded tool tasks, the five assignment metrics under v1 and
v2.1). It refuses if the file exists: the list is frozen and committed before any test output.
run refuses a dirty or uncommitted tree, any changed eval config/protocol/adapter, and existing test outputs unless
--rerun-reason is given (written to RERUN.txt and disclosed). Each model uses its own family's eval config.
Scoring and the report run on CPU afterwards: scripts/score_v2.py --split test, then the pre-specified summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinqa.config import load_yaml, resolve  # noqa: E402
from clinqa.run_info import adapter_sha256, git_state, sha256_file  # noqa: E402

FROZEN = ROOT / "configs" / "w4" / "final_test.json"
REPORT = ["natural Q5 fabrication on test (n = 10; underpowered, disclosed)", "grounded tool tasks on test",
          "five assignment metrics on test under legacy v1 and diagnostic v2.1"]


def protocol_digest(eval_config: str) -> str:
    from clinqa.evaluate import protocol

    return hashlib.sha256(json.dumps(protocol(load_yaml(eval_config)), sort_keys=True, default=str).encode()).hexdigest()


def cmd_freeze(a: argparse.Namespace) -> None:
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} exists; the test list is frozen once")
    entries = []
    for label, config, run, adapter in a.entry:
        if not label.endswith("_test"):
            raise SystemExit(f"{label}: test labels must end with _test")
        cfg = load_yaml(config)
        if run not in cfg["runs"]:
            raise SystemExit(f"{run} is not a run of {config}")
        ad = None if adapter in ("none", "null", "-") else adapter
        entries.append({"label": label, "eval_config": config, "eval_config_sha256": sha256_file(resolve(config)),
                        "protocol_sha256": protocol_digest(config), "run": run, "adapter": ad,
                        "adapter_sha256": adapter_sha256(resolve(ad)) if ad else None})
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps({"status": "frozen", "runs": entries, "pre_specified_report": REPORT}, indent=1) + "\n")
    print(f"wrote {FROZEN.relative_to(ROOT)} with {len(entries)} models; commit it before `run`")


def cmd_run(a: argparse.Namespace) -> int:
    from clinqa.evaluate import generate, run_dir, score

    spec = json.loads(FROZEN.read_text())
    git = git_state()
    if git["dirty"] or not git["commit"]:
        print("refusing: the test run requires a clean, committed tree (D-030)")
        return 1
    for e in spec["runs"]:
        if sha256_file(resolve(e["eval_config"])) != e["eval_config_sha256"] or protocol_digest(e["eval_config"]) != \
                e["protocol_sha256"]:
            raise SystemExit(f"{e['label']}: eval config or protocol changed after freezing")
        if e["adapter"] and adapter_sha256(resolve(e["adapter"])) != e["adapter_sha256"]:
            raise SystemExit(f"{e['label']}: adapter hash changed after freezing")
        out = run_dir(load_yaml(e["eval_config"]), e["label"], "test")
        if (out / "trajectories.jsonl").exists() and not a.rerun_reason:
            print(f"refusing to overwrite {out}: test runs once; pass --rerun-reason to disclose a rerun")
            return 1
    frozen = {"final_config": str(FROZEN.relative_to(ROOT)), "final_config_sha256": sha256_file(FROZEN),
              "final_git_commit": git["commit"]}
    for e in spec["runs"]:
        cfg = load_yaml(e["eval_config"])
        generate(cfg, e["run"], "test", e["adapter"], e["label"], extra_info=frozen)
        if a.rerun_reason:
            (run_dir(cfg, e["label"], "test") / "RERUN.txt").write_text(a.rerun_reason + "\n")
        score(cfg, e["label"], "test")  # legacy v1 metrics; v2.1 on CPU afterwards
    print("Test generation done. Score on CPU: scripts/score_v2.py --split test --labels "
          + " ".join(e["label"] for e in spec["runs"]))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--entry", nargs=4, action="append", required=True, metavar=("LABEL", "EVAL_CONFIG", "RUN", "ADAPTER"))
    r = sub.add_parser("run")
    r.add_argument("--rerun-reason", default=None)
    a = p.parse_args()
    if a.cmd == "freeze":
        cmd_freeze(a)
        return 0
    return cmd_run(a)


if __name__ == "__main__":
    raise SystemExit(main())
