"""Agreement of scorer v2 with the blind adjudication sets (reports/scorer_v2/adjudication/).

    uv run python scripts/score_v2.py ... --out reports/scorer_v2/{val,test}   # rescore first
    uv run python scripts/scorer_v2_agreement.py

Reference labels: dev_val = A and C agree (the single A/C tie, item_015, is resolved as incorrect by the rubric);
holdout1_test = HC (identical to HA on all 120 items); holdout2_test = items where HA and HC agree (108/110). Rater B is excluded (one templated reason for 71 items).
"""

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "reports" / "scorer_v2"


def labels(files):
    return {x["item"]: x["label"] == "correct" for f in files for x in map(json.loads, open(f))}


def agree(key_file, ref, scored_dir):
    key = json.load(open(key_file))
    scored = {}
    for f in glob.glob(str(scored_dir / "*.scored.jsonl")):
        run = Path(f).name.replace(".scored.jsonl", "")
        for line in open(f):
            s = json.loads(line)
            scored[(run, s["id"])] = s
    items = sorted(key)
    ref_ = [ref[i] for i in items]
    pred = [scored[(key[i]["run"], key[i]["id"])]["correct"] for i in items]
    n = len(items)
    po = sum(a == b for a, b in zip(pred, ref_)) / n
    px, py = sum(pred) / n, sum(ref_) / n
    pe = px * py + (1 - px) * (1 - py)
    wrong = [(i, key[i]["id"], scored[(key[i]["run"], key[i]["id"])]["error"]) for i, a, b in zip(items, pred, ref_)
             if a != b]
    return {"n": n, "agreement": round(po, 3), "kappa": round((po - pe) / (1 - pe), 3),
            "false_pass": sum(a and not b for a, b in zip(pred, ref_)),
            "false_fail": sum(b and not a for a, b in zip(pred, ref_)), "disagreements": wrong}


def main():
    dev = ROOT / "adjudication" / "dev_val"
    a = labels([dev / "rater_A.jsonl"])
    c = labels(sorted(dev.glob("rater_C*.jsonl")))
    print("dev_val (val, 120):", agree(dev / "key.json", {i: a[i] and c[i] for i in a}, ROOT / "val"))
    h1 = ROOT / "adjudication" / "holdout1_test"
    hc = labels(sorted(h1.glob("rater_HC*.jsonl")))
    print("holdout1_test (test, 120):", agree(h1 / "hold_key.json", hc, ROOT / "test"))
    h2 = ROOT / "adjudication" / "holdout2_test"
    ha2, hc2 = labels([h2 / "rater_HA.jsonl"]), labels(sorted(h2.glob("rater_HC*.jsonl")))
    both = {i for i in ha2 if ha2[i] == hc2[i]}  # 2 items differ only on the "manual BMI, no tool call" policy
    key2 = json.load(open(h2 / "hold2_key.json"))
    sub = h2 / "_agreed_key.json"
    sub.write_text(json.dumps({i: key2[i] for i in both}))
    print("holdout2_test (test, frozen v2.1, HA=HC items):", agree(sub, hc2, ROOT / "test"))
    sub.unlink()


if __name__ == "__main__":
    main()
