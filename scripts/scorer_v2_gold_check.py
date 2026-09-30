"""Gold-as-prediction self-check for scorer v2 (MedCalc-style keys). Never run on test.

    uv run python scripts/scorer_v2_gold_check.py train|val
"""

import collections
import json
import sys

from clinqa import scorer_v2 as s2
from clinqa.config import PROJECT_ROOT

split = sys.argv[1] if len(sys.argv) > 1 else "train"
assert split in ("train", "val"), "gold self-check is train/val only"
recs = [json.loads(line) for line in open(PROJECT_ROOT / "data" / f"{split}.jsonl")]
recs = [r for r in recs if r["answer_type"] in ("extractive", "numeric_reasoning")]
fails, examples, coverage = collections.Counter(), collections.defaultdict(list), collections.Counter()
for r in recs:
    k = s2.build_key(r)
    coverage[(r["answer_type"], k.coverage)] += 1
    res = s2.score(r, {"final_answer": r["answer"], "turns": [], "stop_reason": "answer"}, k)
    if not res["correct"]:
        fails[res["error"]] += 1
        examples[res["error"]].append(r["id"])
print("coverage:", dict(coverage))
print(f"gold pass {len(recs) - sum(fails.values())}/{len(recs)}")
for e, c in fails.most_common():
    print(c, e, examples[e][:6])
