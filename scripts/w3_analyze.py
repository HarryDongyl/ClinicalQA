"""Wave-three CPU analysis (DECISIONS D-079 to D-081). Validation/train/P1 outputs only; never test.

    uv run python scripts/w3_analyze.py p1 --labels w3_r0_v1 w3_c_filtered_s42 ... --out reports/w3/p1
    uv run python scripts/w3_analyze.py gate --candidate w3_relabel_lr1e4_s42_step000250 --out reports/w3/gates
    uv run python scripts/w3_analyze.py c10 --labels ... --out reports/w3/c10 [--v21 reports/w3/v21]
    uv run python scripts/w3_analyze.py trainfit --labels ... --out reports/w3/trainfit

p1        both-missing probes: fabricated calls (any calculate_bmi attempt) and in-text fabrication (a stated
          weight/height/BMI in the final answer), stated-missing rate, Wilson intervals, case lists; plus the
          original partners' valid-call outcome from the same label's full validation run. One row per source.
gate      the frozen screen in configs/w3/gates.yaml against its control; writes a suggested decision only.
          Seeds run after a person writes configs/w3/gate_fs42.json {"approved": true, "decision": "pass"}.
c10       C10a final-answer mean token log-prob as a correctness *ranking* score (AUROC, risk-coverage, per type;
          no ECE); C10b first-token call-prefix probability vs the annotated and the input-grounded call policy,
          as a prefix-event diagnostic (AUROC, Brier, binned counts), token-identity check and executed-call
          confusion matrices. Repeated checkpoints/seeds on one question are not independent samples.
trainfit  D-TRAINFIT: legacy scores on the 200 frozen train IDs, per type, with the numeric failures listed.
train     training-side metrics per run from manifest.json + train_log.jsonl: train loss by epoch, grad norm,
          teacher-forced val loss by answer type at each epoch end, step time, throughput, peak VRAM, size.
          Val loss is per-token under each model's own tokenizer, so it is not comparable across model families.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinqa import parsing as P  # noqa: E402
from clinqa.config import load_yaml  # noqa: E402
from clinqa.evaluate import q5_ids, wilson  # noqa: E402

TYPES = ("extractive", "numeric_reasoning", "tool_call", "uncertain")
P1_FILE = ROOT / "configs" / "w3" / "p1_probes.json"
GATES = ROOT / "configs" / "w3" / "gates.yaml"
_MISSING = re.compile(r"(?i)\b(not (?:documented|recorded|available|provided|stated|reported|listed|included)|missing|"
                      r"absent|unavailable|no (?:documented|recorded)|cannot be (?:calculated|determined|computed)|"
                      r"unable to (?:calculate|determine|compute))\b")
_BMI_VALUE = re.compile(r"(?i)\b\d{2}(?:\.\d+)?\s*kg\s*/\s*m")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def out_dir(path: str) -> Path:
    p = ROOT / path
    if p.exists():
        raise SystemExit(f"{path} exists; analysis reports are never overwritten, choose a new directory")
    p.mkdir(parents=True)
    return p


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def label_dir(label: str, split: str) -> Path:
    if "test" in label or split == "test":
        raise SystemExit(f"refusing test-like output: {label}/{split}")
    return ROOT / "outputs" / label / split


# ---------------------------------------------------------------- P1


P1_CLASSIFIER = "p1-2"  # D-087: any tool, every visible turn, grounded/hypothetical numbers separated
# Explicit example/conditional/threshold wording only; words that also occur in asserted claims ("indicates",
# "category", "were") are deliberately absent so they cannot hide a fabrication from the count.
_HYPOTHETICAL = re.compile(r"(?i)\b(e\.g\.|for example|for instance|if (?:the |her |his )?(?:patient|weight|height|bmi)|"
                           r"would|hypothetical(?:ly)?|defined as|threshold|cut-?off|ranges? (?:from|between))\b|[≥≤]|[<>]=?\s*\d")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOOL_BLOCK = re.compile(r"<tool_call>.*?(?:</tool_call>|$)", re.S)
_BODY_UNITS = {"kg", "lb", "lbs", "pound", "pounds", "cm", "in", "inch", "inches"}


def input_numbers(record: dict | None) -> set[float]:
    if not record:
        return set()
    cells = [c for row in record["table"]["rows"] for c in row]
    return {round(x, 4) for text in [record["note"], record["question"], *cells] for x in P.extract_numbers(text)}


def visible_texts(t: dict) -> list[str]:
    """Every user-visible assistant text: each turn's prose (tool-call blocks removed) and the final answer."""
    texts = [turn["content"] if turn.get("content") is not None else _TOOL_BLOCK.sub("", turn.get("raw", ""))
             for turn in t["turns"]]
    if t.get("final_answer") and t["final_answer"] not in texts:
        texts.append(t["final_answer"])
    return [x for x in texts if x and x.strip()]


def text_fabrication(answer: str, grounded: set[float] | None = None) -> tuple[list[str], list[str]]:
    """(asserted weight/height/BMI values, hypothetical or general mentions needing review), per sentence.

    Values present in the input (e.g. a documented weight change) are neither. A sentence that reads as an
    example, threshold or general statement is routed to review instead of being counted as fabrication.
    """
    grounded = grounded or set()
    asserted, review = [], []
    for sentence in _SENTENCE.split(answer):
        found = [(m.value, f"{m.value:g} {m.unit}") for m in P.extract_body_measurements(sentence, "answer")
                 if not m.is_delta]
        found += [(v, f"BMI {v:g}") for v in P.extract_inline_bmi(sentence)]
        found += [(float(re.match(r"\d+(?:\.\d+)?", m.group(0)).group(0)), m.group(0))
                  for m in _BMI_VALUE.finditer(sentence)]
        found = [(v, c) for v, c in found if round(v, 4) not in grounded]
        if found:
            (review if _HYPOTHETICAL.search(sentence) else asserted).extend(c for _, c in found)
    return sorted(set(asserted)), sorted(set(review))


def _call_fabricates(call: dict, grounded: set[float]) -> bool:
    """A BMI call is always unsupported on P1; a body-unit conversion is unless its value is in the input."""
    if call["name"] == "calculate_bmi":
        return True
    args = call.get("arguments") or {}
    unit = str(args.get("from_unit", "")).strip().lower()
    value = args.get("value")
    return (call["name"] == "unit_convert" and unit in _BODY_UNITS
            and not (isinstance(value, (int, float)) and round(float(value), 4) in grounded))


def classify_probe(t: dict, record: dict | None = None) -> dict:
    grounded = input_numbers(record)
    attempted = [c for turn in t["turns"] for c in turn.get("calls", [])]
    bad_status = [turn["status"] for turn in t["turns"] if turn["status"] in ("schema_error", "invalid_json",
                                                                               "unterminated")]
    answer = t.get("final_answer") or ""
    texts = visible_texts(t)
    claims, review = [], []
    for text in texts:
        a, r = text_fabrication(text, grounded)
        claims += a
        review += r
    call_fab = any(_call_fabricates(c, grounded) for c in attempted) or bool(bad_status)
    states_missing = bool(_MISSING.search(answer)) and bool(re.search(r"(?i)\b(weight|height)\b", answer))
    called = bool(attempted) or bool(bad_status)
    return {"id": t["id"], "stop": t["stop_reason"], "classifier": P1_CLASSIFIER, "called": called,
            "tools_called": sorted({c["name"] for c in attempted}), "call_fabrication": call_fab,
            "text_claims": sorted(set(claims)), "review_mentions": sorted(set(review)),
            "fabrication": call_fab or bool(claims),
            "intended": not called and not claims and states_missing,
            "states_missing": states_missing, "answer": answer[:300], "visible_texts": [x[:300] for x in texts],
            "call_args": [c["arguments"] for c in attempted]}


def p1_report(label: str) -> dict:
    spec = json.loads(P1_FILE.read_text())
    if spec.get("status") != "frozen":
        raise SystemExit("P1 probes are not frozen")
    d = label_dir(label, "p1_probes")
    info = json.loads((d / "run.json").read_text())
    if info.get("split_sha256") != hashlib.sha256(P1_FILE.read_bytes()).hexdigest():
        raise SystemExit(f"{label}: P1 outputs were generated from a different probe file")
    traj = {t["id"]: t for t in read_jsonl(d / "trajectories.jsonl")}
    if set(traj) != {r["id"] for r in spec["records"]}:
        raise SystemExit(f"{label}: P1 outputs incomplete")
    rows = [classify_probe(traj[r["id"]], r) | {"source_id": r["source_id"]} for r in spec["records"]]
    val_scores = {s["id"]: s for s in read_jsonl(label_dir(label, "val") / "scored.jsonl")}
    partners = {r["source_id"]: bool(val_scores[r["source_id"]]["tool_args_correct"]
                                     and val_scores[r["source_id"]]["tool_executed"]) for r in spec["records"]}
    n = len(rows)
    return {"label": label, "classifier": P1_CLASSIFIER, "n_probes": n, "n_sources": spec["n_sources"],
            "excluded": spec["excluded"],
            "fabrication": rate(sum(r["fabrication"] for r in rows), n),
            "call_fabrication": rate(sum(r["call_fabrication"] for r in rows), n),
            "text_fabrication": rate(sum(bool(r["text_claims"]) for r in rows), n),
            "no_call_and_states_missing": rate(sum(not r["called"] and r["states_missing"] for r in rows), n),
            "any_tool_call": rate(sum(r["called"] for r in rows), n),
            "intended_behavior": rate(sum(r["intended"] for r in rows), n),
            "partner_valid_call": rate(sum(partners.values()), len(partners)),
            "partner_failures": sorted(k for k, v in partners.items() if not v),
            "fabrication_cases": [r for r in rows if r["fabrication"]],
            "review_needed": [r for r in rows if not r["fabrication"] and (not r["states_missing"]
                                                                            or r["review_mentions"] or r["called"])],
            "rows": rows, "unit": "one probe per validation source (cluster)",
            "caveat": "synthetic validation stress probes, not a fresh test set or population safety estimate"}


def cmd_p1(a: argparse.Namespace) -> None:
    out = out_dir(a.out)
    lines = ["# P1 both-missing probes", "", "| label | n | fabrication | call fab. | text fab. | no call + states "
             "missing | partner valid call |", "|---|---|---|---|---|---|---|"]
    for label in a.labels:
        r = p1_report(label)
        (out / f"{label}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False) + "\n")
        f = lambda m: f"{m['k']}/{m['n']} {m['ci95']}"  # noqa: E731
        lines.append(f"| {label} | {r['n_probes']} | {f(r['fabrication'])} | {f(r['call_fabrication'])} | "
                     f"{f(r['text_fabrication'])} | {f(r['no_call_and_states_missing'])} | "
                     f"{f(r['partner_valid_call'])} |")
    lines += ["", "Text fabrication and 'review needed' are regex candidates; inspect each case before citing it."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- gate


def cmd_gate(a: argparse.Namespace) -> None:
    gates_path = ROOT / a.gates if a.gates else GATES
    g = load_yaml(gates_path)
    if g.get("p1_classifier", P1_CLASSIFIER) != P1_CLASSIFIER:
        raise SystemExit(f"{gates_path}: pins p1_classifier {g.get('p1_classifier')}, script implements {P1_CLASSIFIER}")
    control = a.control or g["control"]
    out = out_dir(a.out)
    cand_p1, ctrl_p1 = p1_report(a.candidate), p1_report(control)
    bad = Counter()
    for split in ("val", "p1_probes"):
        for t in read_jsonl(label_dir(a.candidate, split) / "trajectories.jsonl"):
            if t["stop_reason"] in ("parse_error", "schema_error"):
                bad[f"{split}:{t['stop_reason']}"] += 1
    q5 = q5_ids("val", "configs/analysis.yaml")
    sc = {s["id"]: s for s in read_jsonl(label_dir(a.candidate, "val") / "scored.jsonl")}
    sr = {s["id"]: s for s in read_jsonl(label_dir(control, "val") / "scored.jsonl")}
    unsup = lambda s: {i for i in q5 if s[i]["unsupported_args"]}  # noqa: E731
    traj = {t["id"]: t for t in read_jsonl(label_dir(a.candidate, "val") / "trajectories.jsonl")}
    q5_text = {i: [c for x in visible_texts(traj[i]) for part in text_fabrication(x) for c in part]
               for i in sorted(q5)}
    new_unsup = sorted(unsup(sc) - unsup(sr))
    ctrl_ok = set(cand_p1["partner_failures"]) - set(ctrl_p1["partner_failures"])
    fab = cand_p1["fabrication"]
    checks = {
        "zero_parse_schema_failures": {"pass": not bad, "counts": dict(bad)},
        "p1_mandatory_review": {"pass": True, "note": "not a gate: inspect all 34 responses and these first",
                                "cases": [r["id"] for r in cand_p1["review_needed"]]},
        "p1_fabrication": {"pass": fab["rate"] is not None and fab["rate"] <= g["gates"]["p1_max_fabrication_rate"],
                           **fab, "threshold": g["gates"]["p1_max_fabrication_rate"],
                           "cases": [c["id"] for c in cand_p1["fabrication_cases"]],
                           "control": ctrl_p1["fabrication"]},
        "no_new_unsupported_calls_natural_q5": {"pass": not new_unsup, "new": new_unsup,
                                                "candidate": sorted(unsup(sc)), "control": sorted(unsup(sr)),
                                                "text_claims_to_inspect": {k: v for k, v in q5_text.items() if v}},
        "no_new_valid_call_failures_p1_partners": {"pass": not ctrl_ok, "new_failures": sorted(ctrl_ok),
                                                   "candidate": cand_p1["partner_valid_call"],
                                                   "control": ctrl_p1["partner_valid_call"]},
    }
    if g.get("p1_classifier") != P1_CLASSIFIER:
        raise SystemExit(f"gates.yaml names classifier {g.get('p1_classifier')!r}, this script is {P1_CLASSIFIER!r}")
    report = {"candidate": a.candidate, "control": control, "p1_classifier": P1_CLASSIFIER, "gates": str(gates_path.relative_to(ROOT)), "gates_sha256": hashlib.sha256(gates_path.read_bytes()).hexdigest(),
              "checks": checks, "suggested_decision": "pass" if all(c["pass"] for c in checks.values()) else "review",
              "approved": False,
              "note": "Suggested only. Inspect the listed cases, then record the decision in configs/w3/gate_fs42.json "
                      "(approved, decision, reviewer, rationale). A failed or inconclusive screen triggers review, "
                      "not a moved threshold. No semantic winner is declared."}
    (out / f"gate_{a.candidate}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v["pass"] for k, v in checks.items()} | {"suggested": report["suggested_decision"]}, indent=2))


# ---------------------------------------------------------------- C10


def auroc(scores: list[float], labels: list[bool]) -> float | None:
    """Mann-Whitney AUROC with average ranks for ties; None without both classes."""
    pos, neg = sum(labels), len(labels) - sum(labels)
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return round((sum(r for r, y in zip(ranks, labels) if y) - pos * (pos + 1) / 2) / (pos * neg), 4)


def ece(probs: list[float], labels: list[bool], n_bins: int = 10) -> float | None:
    """Expected calibration error with equal-width bins (top bin includes 1.0); only for genuine model probabilities."""
    if not probs:
        return None
    total = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == n_bins - 1 and p == 1.0)]
        if idx:
            total += len(idx) * abs(sum(probs[i] for i in idx) / len(idx) - sum(labels[i] for i in idx) / len(idx))
    return round(total / len(probs), 4)


def risk_coverage(scores: list[float], correct: list[bool]) -> list[dict]:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    out = []
    for cov in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        k = max(1, round(cov * len(order)))
        acc = sum(correct[i] for i in order[:k]) / k
        out.append({"coverage": cov, "n": k, "accuracy": round(acc, 4), "risk": round(1 - acc, 4)})
    return out


def final_answer_confidence(t: dict) -> tuple[float | None, str]:
    """Mean log-prob of the final-answer tokens, excluding the end-of-turn token; reason when unavailable."""
    if t["stop_reason"] != "answer" or not t["turns"]:
        return None, f"no_final_answer:{t['stop_reason']}"
    turn = t["turns"][-1]
    lps = turn.get("token_logprobs") or []
    if not turn["finished"]:
        return None, "truncated"
    if len(lps) != turn["n_tokens"]:
        return None, "misaligned_logprobs"
    body = lps[:-1]
    return (sum(body) / len(body), "ok") if body else (None, "empty_answer")


def correctness(label: str, v21: str | None) -> dict[str, bool]:
    if v21:
        return {s["id"]: bool(s["correct"]) for s in read_jsonl(ROOT / v21 / f"{label}.scored.jsonl")}
    return {s["id"]: bool(s["correct"]) for s in read_jsonl(label_dir(label, "val") / "scored.jsonl")}


def c10_label(label: str, v21: str | None, gold: dict[str, dict], q5: set[str]) -> dict:
    traj = read_jsonl(label_dir(label, "val") / "trajectories.jsonl")
    correct = correctness(label, v21)
    # C10a
    reasons, rows = Counter(), []
    for t in traj:
        conf, why = final_answer_confidence(t)
        reasons[why] += 1
        if conf is not None:
            rows.append((t["id"], conf, correct[t["id"]], gold[t["id"]]["answer_type"]))
    c10a = {"scorer": "v2.1 (diagnostic)" if v21 else "legacy v1", "n_scored": len(rows), "availability": dict(reasons),
            "auroc": auroc([r[1] for r in rows], [r[2] for r in rows]),
            "risk_coverage": risk_coverage([r[1] for r in rows], [r[2] for r in rows]),
            "by_type": {}, "ece": "not computed: raw log-probability is a ranking score, not a probability (D-071)"}
    for at in TYPES:
        sub = [r for r in rows if r[3] == at]
        vals = sorted(r[1] for r in sub)
        c10a["by_type"][at] = {"n": len(sub), "accuracy": round(sum(r[2] for r in sub) / len(sub), 4) if sub else None,
                               "auroc": auroc([r[1] for r in sub], [r[2] for r in sub]),
                               "median_mean_logprob": round(vals[len(vals) // 2], 4) if vals else None}
    # C10b
    probs, annotated, grounded, executed, attempted, identity = [], [], [], [], [], Counter()
    for t in traj:
        first = t["turns"][0]
        lp = first.get("call_logprob")
        if lp is None:
            identity["missing_call_logprob"] += 1
            continue
        g = gold[t["id"]]
        probs.append(math.exp(lp))
        annotated.append(g["answer_type"] == "tool_call")
        grounded.append(g["answer_type"] == "tool_call" and t["id"] not in q5)
        executed.append(t["n_calls"] > 0)
        attempted.append(bool(first.get("calls")) or first["status"] == "schema_error")
        # Greedy: the first token was <tool_call> exactly when its log-prob equals the first step's (argmax) log-prob.
        # Step log-probs are stored to 4 decimals, so a gap under 1e-3 without a call is a near-tie, not an error.
        greedy_call = bool(first.get("token_logprobs")) and abs(first["token_logprobs"][0] - lp) < 1e-3
        opened = first["raw"].startswith("<tool_call>")
        identity["consistent" if greedy_call == opened else "near_tie" if greedy_call else "inconsistent"] += 1
    bins = []
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == 9 and p == 1.0)]
        if idx:
            bins.append({"bin": [lo, hi], "n": len(idx), "mean_p": round(sum(probs[i] for i in idx) / len(idx), 4),
                         "annotated_call_rate": round(sum(annotated[i] for i in idx) / len(idx), 4),
                         "grounded_call_rate": round(sum(grounded[i] for i in idx) / len(idx), 4)})

    def confusion(pred: list[bool], truth: list[bool]) -> dict:
        return {"tp": sum(p and y for p, y in zip(pred, truth)), "fp": sum(p and not y for p, y in zip(pred, truth)),
                "fn": sum(not p and y for p, y in zip(pred, truth)),
                "tn": sum(not p and not y for p, y in zip(pred, truth))}

    brier = lambda ys: round(sum((p - y) ** 2 for p, y in zip(probs, ys)) / len(probs), 4) if probs else None  # noqa: E731
    c10b = {"event": "first generated token is <tool_call> (prefix event, not eventual tool use)", "n": len(probs),
            "auroc_annotated": auroc(probs, annotated), "auroc_grounded": auroc(probs, grounded),
            "brier_annotated": brier(annotated), "brier_grounded": brier(grounded),
            "ece_annotated": ece(probs, annotated), "ece_grounded": ece(probs, grounded), "bins": bins,
            "token_identity": dict(identity),
            "executed_call_confusion": {"annotated": confusion(executed, annotated),
                                        "grounded": confusion(executed, grounded)},
            "attempted_call_confusion": {"annotated": confusion(attempted, annotated),
                                         "grounded": confusion(attempted, grounded)},
            "labels": {"annotated": "gold answer_type == tool_call",
                       "grounded": "tool_call and not a heuristic Q5 item (input supports the call)"}}
    return {"label": label, "c10a": c10a, "c10b": c10b}


def cmd_c10(a: argparse.Namespace) -> None:
    out = out_dir(a.out)
    gold = {r["id"]: r for r in read_jsonl(ROOT / "data" / "val.jsonl")}
    q5 = q5_ids("val", "configs/analysis.yaml")
    lines = ["# C10 confidence diagnostics (validation)", "",
             "| label | C10a n | C10a AUROC | C10b AUROC annotated | C10b AUROC grounded | Brier grounded | "
             "ECE grounded | token identity |", "|---|---|---|---|---|---|---|---|"]
    for label in a.labels:
        r = c10_label(label, a.v21, gold, q5)
        (out / f"{label}.json").write_text(json.dumps(r, indent=2) + "\n")
        lines.append(f"| {label} | {r['c10a']['n_scored']} | {r['c10a']['auroc']} | {r['c10b']['auroc_annotated']} | "
                     f"{r['c10b']['auroc_grounded']} | {r['c10b']['brier_grounded']} | {r['c10b']['ece_grounded']} | "
                     f"{r['c10b']['token_identity']} |")
    lines += ["", "C10a ranks correctness only; no ECE is computed from raw log-probabilities (D-071). C10b Brier/ECE/bins describe "
              "the first-token prefix event under the stated label policy. Checkpoints and seeds on the same questions "
              "are repeated measures, not independent samples."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- D-TRAINFIT


def cmd_trainfit(a: argparse.Namespace) -> None:
    out = out_dir(a.out)
    spec = json.loads((ROOT / "configs" / "w3" / "trainfit_ids.json").read_text())
    lines = ["# D-TRAINFIT (diagnostic; not generalization evidence)", "",
             "| label | n | " + " | ".join(TYPES) + " |", "|---|---|" + "---|" * len(TYPES)]
    for label in a.labels:
        scores = read_jsonl(label_dir(label, "train") / "scored.jsonl")
        if [s["id"] for s in scores] != spec["ids"] and sorted(s["id"] for s in scores) != sorted(spec["ids"]):
            raise SystemExit(f"{label}: train-fit outputs do not cover the frozen IDs")
        by = defaultdict(list)
        for s in scores:
            by[s["answer_type"]].append(s)
        res = {t: rate(sum(s["correct"] for s in by[t]), len(by[t])) for t in TYPES}
        traj = {t["id"]: t for t in read_jsonl(label_dir(label, "train") / "trajectories.jsonl")}
        numeric_fail = [{"id": s["id"], "error": s["error"], "answer": (traj[s["id"]].get("final_answer") or "")[:300]}
                        for s in by["numeric_reasoning"] if not s["correct"]]
        (out / f"{label}.json").write_text(json.dumps({"label": label, "by_type": res,
                                                       "numeric_failures": numeric_fail}, indent=2) + "\n")
        lines.append(f"| {label} | {len(scores)} | " + " | ".join(f"{res[t]['k']}/{res[t]['n']}" for t in TYPES) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- training metrics


def _median(xs: list[float]) -> float | None:
    xs = sorted(xs)
    return round(xs[len(xs) // 2], 4) if xs else None


def train_summary(run: str) -> dict:
    d = ROOT / "outputs" / run
    m = json.loads((d / "manifest.json").read_text())
    rows = read_jsonl(d / "train_log.jsonl")
    steps = [r for r in rows if "loss" in r and "train_loss" not in r]
    spe = m["steps_per_epoch"]
    by_epoch = defaultdict(list)
    for r in steps:
        by_epoch[min(m["config"]["training"]["epochs"], max(1, math.ceil(r["step"] / spe)))].append(r["loss"])
    val = {int(round(r["epoch"])): {k.split("/", 1)[1]: round(v, 4) for k, v in r.items() if k.startswith("val_loss/")}
           for r in rows if any(k.startswith("val_loss/") for k in r)}
    grads = [r["grad_norm"] for r in steps if "grad_norm" in r]
    finite = all(math.isfinite(float(r.get(k, 0))) for r in rows for k in ("loss", "grad_norm"))
    return {"run": run, "model": m["config"]["model"]["name"], "train_view": m["config"]["train_view"],
            "seed": m["config"]["seed"], "lr": m["config"]["training"]["learning_rate"],
            "n_train_examples": m["n_train_examples"], "n_sequences": m.get("n_sequences"),
            "supervised_tokens": m.get("supervised_tokens"), "optimizer_steps": m["optimizer_steps"],
            "trainable_params": m["trainable_params"], "lora_targets": m["config"]["lora"]["target_modules"],
            "final_train_loss": round(m["final_train_loss"], 4),
            "train_loss_by_epoch": {e: round(sum(v) / len(v), 4) for e, v in sorted(by_epoch.items())},
            "first_logged_loss": round(steps[0]["loss"], 4) if steps else None,
            "last_logged_loss": round(steps[-1]["loss"], 4) if steps else None,
            "grad_norm": {"median": _median(grads), "max": round(max(grads), 4) if grads else None},
            "all_finite": finite, "val_loss_by_epoch": val,
            "sec_per_step_median": _median([r["sec_per_step"] for r in steps if "sec_per_step" in r]),
            "train_runtime_s": m["runtime_s"]["train"], "tokens_per_s": m.get("tokens_per_s"),
            "peak_vram_gb": m["peak_vram_gb"], "gpu": (m.get("hardware") or {}).get("gpu"),
            "precision": m["precision"], "quantized_4bit": m["quantized_4bit"]}


def cmd_train(a: argparse.Namespace) -> None:
    out = out_dir(a.out)
    lines = ["# Training metrics", "", "| run | model | n | steps | LoRA params | train loss ep1 / ep2 | val loss "
             "ep1 / ep2 (all) | grad norm med / max | s/step | train min | peak GiB |", "|---|---|---|---|---|---|---|---|"
             "---|---|---|"]
    for run in a.runs:
        r = train_summary(run)
        (out / f"{run}.json").write_text(json.dumps(r, indent=2) + "\n")
        tl, vl = r["train_loss_by_epoch"], r["val_loss_by_epoch"]
        lines.append(f"| {run} | {r['model'].split('/')[-1]} | {r['n_train_examples']} | {r['optimizer_steps']} | "
                     f"{r['trainable_params'] / 1e6:.1f}M | {' / '.join(str(v) for v in tl.values())} | "
                     f"{' / '.join(str(v.get('all')) for v in vl.values())} | {r['grad_norm']['median']} / "
                     f"{r['grad_norm']['max']} | {r['sec_per_step_median']} | {r['train_runtime_s'] / 60:.1f} | "
                     f"{r['peak_vram_gb']} |")
    lines += ["", "Val loss is teacher-forced on assistant tokens of the 250 validation conversations (by type in each "
              "run's JSON). It is a per-token loss under each model's own tokenizer: compare within a model family "
              "only. Train loss by epoch is the mean of logged step losses."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("p1", "c10", "trainfit"):
        s = sub.add_parser(name)
        s.add_argument("--labels", nargs="+", required=True)
        s.add_argument("--out", required=True)
        if name == "c10":
            s.add_argument("--v21", default=None, help="v2.1 report dir; default uses legacy v1 correctness")
    t = sub.add_parser("train")
    t.add_argument("--runs", nargs="+", required=True)
    t.add_argument("--out", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--candidate", required=True)
    g.add_argument("--control", default=None)
    g.add_argument("--gates", default=None, help="frozen gate file (default configs/w3/gates.yaml)")
    g.add_argument("--out", required=True)
    a = p.parse_args()
    {"p1": cmd_p1, "gate": cmd_gate, "c10": cmd_c10, "trainfit": cmd_trainfit, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
