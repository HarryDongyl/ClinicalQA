"""Build TRAIN-ONLY Q5 annotation proposals; never alter or register a training view."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from clinqa.parsing import extract_inline_bmi, record_body_measurements
from clinqa.training_data import grounding_flags


def main():
    out = ROOT / "reports/history/q5_uncertain_proposals"
    if out.exists():
        raise FileExistsError("Proposal directory exists; preserve it and version subsequent audits explicitly.")
    source = ROOT / "data/train.jsonl"
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    assert len(records) == 2000 and all(r["id"].startswith("train_") for r in records)
    ids = {f.id for f in grounding_flags(records)}
    proposals = []
    for r in records:
        if r["id"] not in ids:
            continue
        ms = record_body_measurements(r)
        missing = [kind for kind in ("weight", "height") if not any(m.kind == kind for m in ms)]
        documented_bmi = extract_inline_bmi(r["note"])
        answer = None
        if documented_bmi:
            category = "review_documented_bmi_before_relabeling"
        elif not missing:
            category = "review_argument_mismatch_not_missing_input"
        else:
            category = "candidate_missing_input_uncertain"
            available = []
            for kind in ("weight", "height"):
                values = {(m.value, m.unit) for m in ms if m.kind == kind}
                if len(values) == 1:
                    value, unit = next(iter(values))
                    available.append(f"{kind} {value:g} {unit}")
                elif len(values) > 1:
                    category = "review_multiple_documented_measurements"
            prefix = "The provided information documents " + " and ".join(available) + ". " if available else ""
            field = " and ".join(missing).capitalize()
            answer = prefix + field + (" are" if len(missing)>1 else " is") + " not documented, so BMI cannot be calculated from the available information."
        proposals.append({"id": r["id"], "status": "unreviewed_do_not_train", "category": category,
                          "input": {k: r[k] for k in ("note", "table", "question")},
                          "original_answer": r["answer"], "original_tool_calls": r.get("tool_calls", []),
                          "missing_fields": missing, "documented_bmi": documented_bmi,
                          "proposed_answer_type": "uncertain" if answer else None,
                          "proposed_answer": answer, "proposed_tool_calls": [] if answer else None,
                          "reviewer": None, "accepted": None, "rationale": None})
    out.mkdir(parents=True)
    (out / "proposals.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False)+"\n" for p in proposals))
    manifest = {"source": "data/train.jsonl", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "split": "train", "validation_or_test_read": False, "canonical_data_modified": False,
                "training_view_registered": False, "n": len(proposals),
                "categories": dict(Counter(p["category"] for p in proposals))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
