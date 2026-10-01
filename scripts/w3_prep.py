"""Wave-three CPU preparation and approval gates (EXPERIMENTS_WAVE3; DECISIONS D-075 to D-083). No GPU.

    uv run python scripts/w3_prep.py relabel-template   Q5 review sheet configs/w3/q5_relabel_review.jsonl + packet
    uv run python scripts/w3_prep.py fewshot            draft demos configs/w3/fewshot_v3.json (one per type, train)
    uv run python scripts/w3_prep.py trainfit           frozen D-TRAINFIT IDs configs/w3/trainfit_ids.json
    uv run python scripts/w3_prep.py p1                 draft P1 probes configs/w3/p1_probes.json + packet
    uv run python scripts/w3_prep.py approve KIND --reviewer NAME   KIND = prompt_v3 | fewshot | p1
    uv run python scripts/w3_prep.py check [--require KIND ...]     readiness; exit 1 if a required item is missing
    uv run python scripts/w3_prep.py audit-8b           Qwen3-8B mask/stop-token/call-prefix audit -> reports/w3/

Every artifact is built from train records, except P1, which edits the 40 grounded validation BMI records
(D-070). Builders refuse to overwrite; approval re-runs every automatic check and records the reviewer and
hashes. Drafts are never used by the GPU round (clinqa.evaluate refuses unapproved demos/unfrozen probes).
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinqa import parsing as P  # noqa: E402
from clinqa.data_io import load_split, read_jsonl  # noqa: E402
from clinqa.formatting import (load_tokenizer, prompt_messages, render, target_messages,  # noqa: E402
                               user_content)
from clinqa.config import load_yaml  # noqa: E402
from clinqa.training_data import grounding_flags  # noqa: E402

W3 = ROOT / "configs" / "w3"
REPORTS = ROOT / "reports" / "w3"
TYPES = ("extractive", "numeric_reasoning", "tool_call", "uncertain")
PROMPT_V3 = ROOT / "configs" / "prompts" / "system_v3.txt"
FEWSHOT = W3 / "fewshot_v3.json"
TRAINFIT = W3 / "trainfit_ids.json"
P1 = W3 / "p1_probes.json"
REVIEW = W3 / "q5_relabel_review.jsonl"
APPROVALS = W3 / "approvals.json"
TRAINFIT_PER_TYPE = 50


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def refuse_existing(*paths: Path) -> None:
    for p in paths:
        if p.exists():
            raise SystemExit(f"{p.relative_to(ROOT)} exists; frozen/drafted artifacts are never overwritten")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def filtered_train() -> list[dict]:
    return read_jsonl(ROOT / "data" / "processed" / "q5_filtered" / "train.jsonl")


# ---------------------------------------------------------------- Q5 relabel review sheet


def cmd_relabel_template(_: argparse.Namespace) -> None:
    refuse_existing(REVIEW)
    proposals = read_jsonl(ROOT / "reports" / "q5_uncertain_proposals" / "proposals.jsonl")
    flagged = {f.id for f in grounding_flags(load_split("train"))}
    if {p["id"] for p in proposals} != flagged:
        raise SystemExit("proposal IDs differ from the current train Q5 candidates; rebuild proposals first")
    rows = [{"id": p["id"], "accepted": None, "reviewer": None, "answer": p["proposed_answer"],
             "rationale": None, "proposal_category": p["category"], "missing_fields": p["missing_fields"]}
            for p in proposals]
    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    REVIEW.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    L = ["# Q5 relabel review packet (train only)", "",
         f"{len(rows)} candidates. Edit `configs/w3/q5_relabel_review.jsonl`: set `accepted` true/false, `reviewer`, "
         "`rationale`; edit `answer` if the proposal does not answer every part of the question from the input. "
         "Accept only when the input genuinely lacks the measurements needed for the gold calculation (D-075).", ""]
    for p in proposals:
        q = p["input"]["question"]
        ms = P.record_body_measurements({"note": p["input"]["note"], "table": p["input"]["table"], "question": q})
        L += [f"## {p['id']}", "", f"**Question:** {q}", "",
              f"**Parser measurements in input:** {[(m.kind, m.value, m.unit, m.source) for m in ms] or 'none'}",
              f"**Documented BMI:** {p['documented_bmi'] or 'none'}", "",
              f"**Original gold call:** `{json.dumps(p['original_tool_calls'])}`", "",
              f"**Original answer:** {p['original_answer']}", "", f"**Proposed answer:** {p['proposed_answer']}", ""]
    (REPORTS / "q5_review_packet.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {REVIEW.relative_to(ROOT)} ({len(rows)} rows, all undecided) and reports/w3/q5_review_packet.md")


# ---------------------------------------------------------------- few-shot demonstrations


def fewshot_candidates(records: list[dict], tok) -> dict[str, dict]:
    """Per type, the median-length record with no quality flag; the tool demo is a metric calculate_bmi call."""
    flagged = {json.loads(line)["id"] for line in (ROOT / "reports" / "quality_flags.jsonl").read_text().splitlines()
               if line.strip() and json.loads(line)["split"] == "train"}
    chosen = {}
    for t in TYPES:
        pool = [r for r in records if r["answer_type"] == t and r["id"] not in flagged]
        if t == "tool_call":
            pool = [r for r in pool if r["tool_calls"][0]["tool"] == "calculate_bmi"
                    and not P.extract_body_measurements(r["question"], "question")
                    and all(m.unit in ("kg", "cm") for m in P.record_body_measurements(r))]
        sized = sorted(((len(tok(render(tok, [{"role": "user", "content": user_content(r)}] + target_messages(r)),
                                 add_special_tokens=False)["input_ids"]), r["id"]) for r in pool))
        chosen[t] = next(r for r in pool if r["id"] == sized[len(sized) // 2][1])
    return chosen


def demo_messages(chosen: dict[str, dict]) -> list[dict]:
    msgs = []
    for t in TYPES:
        r = chosen[t]
        msgs += [{"role": "user", "content": user_content(r)}] + target_messages(r)
    return msgs


def cmd_fewshot(_: argparse.Namespace) -> None:
    refuse_existing(FEWSHOT)
    fmt = load_yaml(ROOT / "configs" / "format_core.yaml")
    tok = load_tokenizer(fmt)
    chosen = fewshot_candidates(filtered_train(), tok)
    msgs = demo_messages(chosen)
    system = PROMPT_V3.read_text(encoding="utf-8").strip()
    val = load_split("val")
    lengths = sorted(len(tok(render(tok, prompt_messages(r, system, msgs), add_generation_prompt=True),
                             add_special_tokens=False)["input_ids"]) for r in val)
    spec = {"status": "draft", "split": "train", "selection_rule": (
                "per answer type in TYPES order: records in the q5_filtered train view with no quality flag of any "
                "severity; tool_call restricted to calculate_bmi with metric-only input measurements and none in the "
                "question; pick the median rendered-length record (D-078)"),
            "ids": [chosen[t]["id"] for t in TYPES], "order": list(TYPES),
            "messages": msgs, "messages_sha256": sha256_text(canon(msgs)),
            "system_prompt": str(PROMPT_V3.relative_to(ROOT)),
            "system_prompt_sha256": sha256_text(system),
            "val_first_turn_prompt_tokens": {"min": lengths[0], "p50": lengths[len(lengths) // 2], "max": lengths[-1]},
            "tokenizer": fmt["tokenizer"], "tokenizer_revision": fmt["tokenizer_revision"],
            "self_check": "tool demo result recomputed by the executor and equal to gold (formatting.target_messages)",
            "reviewer": None}
    write_json(FEWSHOT, spec)
    print(f"wrote draft {FEWSHOT.relative_to(ROOT)}: ids {spec['ids']}, val prompt tokens "
          f"{spec['val_first_turn_prompt_tokens']}")


# ---------------------------------------------------------------- D-TRAINFIT IDs


def cmd_trainfit(_: argparse.Namespace) -> None:
    refuse_existing(TRAINFIT)
    if not FEWSHOT.exists():
        raise SystemExit("build the few-shot draft first: demonstrations are excluded from train-fit")
    demos = set(json.loads(FEWSHOT.read_text())["ids"])
    records = filtered_train()  # rows unchanged in both the filtered and relabeled views
    rng = random.Random(42)
    ids = []
    for t in TYPES:
        pool = sorted(r["id"] for r in records if r["answer_type"] == t and r["id"] not in demos)
        ids += rng.sample(pool, TRAINFIT_PER_TYPE)
    order = {r["id"]: i for i, r in enumerate(records)}
    ids.sort(key=order.__getitem__)
    write_json(TRAINFIT, {"status": "frozen", "split": "train", "ids": ids, "n": len(ids),
                          "per_type": TRAINFIT_PER_TYPE, "seed": 42, "excluded_demo_ids": sorted(demos),
                          "rule": "50 per answer type, random.Random(42).sample over sorted unchanged q5_filtered "
                                  "IDs, few-shot demonstrations excluded; diagnostic only, not generalization (D-080)",
                          "ids_sha256": sha256_text("\n".join(ids))})
    print(f"wrote {TRAINFIT.relative_to(ROOT)} ({len(ids)} IDs)")


# ---------------------------------------------------------------- P1 both-missing probes

_LABEL_TOKENS = re.compile(r"(?i)\b(weight|wt|height|ht|bmi|body mass index|kg/m2|kg/m²|kg|kgs|kilograms|lbs?|pounds|"
                           r"cm|inches|inch|in|ft|feet|m)\b|[\d.,:;()\-/²'\"~=]+|\s+|and")
_UNIT_VALUE = re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*(kg|kgs|kilograms|lbs?|pounds|cm|inches|inch)\b")
_RESIDUAL = [
    (re.compile(r"(?i)\b\d+(?:\.\d+)?\s*(?:ft|feet)\b"), "feet value"),
    (re.compile(r"\d+\s*'\s*\d+\s*(?:\"|''|in)?"), "feet-inches"),
    (re.compile(r"(?i)\bkg\s*/\s*m"), "BMI unit"),
    (re.compile(r"(?i)\b(?:weight|height|wt|ht)\b\s*[:=]?\s*\d"), "labelled number"),
]


def _mentions(text: str) -> list[tuple[int, int]]:
    spans = [(m.start, m.end) for m in P.extract_body_measurements(text) if not m.is_delta]
    spans += [(m.start(), m.end()) for m in P._BMI_RE.finditer(text)]
    return spans


def _edit_line(line: str) -> tuple[str | None, bool]:
    """(edited line or None to drop it, whether a mid-line clause edit was needed)."""
    if not _mentions(line):
        return line, False
    if not _LABEL_TOKENS.sub("", line).strip(" -*•#"):
        return None, False
    clauses = re.split(r"(?<=[,;.])\s+", line)
    kept = [c for c in clauses if not _mentions(c) and not re.search(r"(?i)\b(weight|height|bmi)\b\s*[:=]", c)]
    text = " ".join(kept).rstrip(",; ")
    if text and not text.endswith((".", ":")) and line.rstrip().endswith("."):
        text += "."
    return (text if _LABEL_TOKENS.sub("", text).strip(" -*•#") else None), True


def edit_record(r: dict) -> tuple[dict | None, dict]:
    info: dict = {"source_id": r["id"], "mid_line_edits": 0, "removed_table_rows": []}
    if _mentions(r["question"]) or any(p.search(r["question"]) for p, _ in _RESIDUAL):
        info["excluded"] = "question states a measurement or BMI"
        return None, info
    rows = []
    for row in r["table"]["rows"]:
        if re.match(r"(?i)^\s*(weight|height|bmi|body mass index)\b", row[0]):
            info["removed_table_rows"].append(row)
        else:
            rows.append(row)
    if not rows:
        info["excluded"] = "table would be empty"
        return None, info
    lines = []
    for line in r["note"].split("\n"):
        edited, mid = _edit_line(line)
        info["mid_line_edits"] += mid
        if edited is not None:
            lines.append(edited)
    note = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    probe = {"id": f"p1_{r['id']}", "source_id": r["id"], "note": note,
             "table": {**r["table"], "rows": rows}, "question": r["question"]}
    problems = probe_problems(probe, r)
    if problems:
        info["excluded"] = "residual measurement after editing: " + "; ".join(problems)
        return None, info
    return probe, info


def probe_problems(probe: dict, source: dict) -> list[str]:
    """Automatic checks: nothing from which weight, height or BMI can be read remains."""
    out = []
    if P.record_body_measurements(probe):
        out.append("parser still finds a weight/height")
    for field in ("note", "question"):
        if P.extract_inline_bmi(probe[field]):
            out.append(f"BMI value in {field}")
        # A unit value in the plausible body range that the parser dropped (it only drops implausible
        # sizes such as a 2 cm lesion, or changes such as "lost 8 lbs") still needs a human look.
        deltas = [d for d in P.extract_body_measurements(probe[field]) if d.is_delta]
        for m in _UNIT_VALUE.finditer(probe[field]):
            unit = P._BODY_UNIT[m.group(2).lower()][1]
            lo, hi = P.PLAUSIBLE_RANGE[unit]
            if lo <= float(m.group(1)) <= hi and not any(d.start <= m.start() < d.end for d in deltas):
                out.append(f"unit value '{m.group(0)}' in {field}")
        for pat, name in _RESIDUAL:
            for m in pat.finditer(probe[field]):
                if not any(d.is_delta and d.start <= m.start() < d.end
                           for d in P.extract_body_measurements(probe[field])):
                    out.append(f"{name} '{m.group(0)}' in {field}")
    if any(re.match(r"(?i)^\s*(weight|height|bmi)", row[0]) for row in probe["table"]["rows"]):
        out.append("table still has a weight/height/BMI row")
    gold = {k: source[k] for k in ("answer", "answer_type", "tool_calls")}
    if not grounding_flags([{**probe, **gold, "id": source["id"]}]):
        out.append("gold BMI arguments are still grounded in the edited input")
    return out


def cmd_p1(_: argparse.Namespace) -> None:
    refuse_existing(P1)
    from clinqa.evaluate import q5_ids

    q5 = q5_ids("val", "configs/analysis.yaml")
    sources = [r for r in load_split("val") if r["answer_type"] == "tool_call" and r["id"] not in q5
               and any(c["tool"] == "calculate_bmi" for c in r["tool_calls"])]
    probes, infos = [], []
    for r in sources:
        probe, info = edit_record(r)
        infos.append(info)
        if probe:
            probes.append(probe)
    spec = {"status": "draft", "split": "val", "kind": "P1 both-missing BMI counterfactual (D-070, D-079)",
            "n_sources": len(sources), "n_eligible": len(probes),
            "excluded": [{"source_id": i["source_id"], "reason": i["excluded"]} for i in infos if "excluded" in i],
            "intended_behavior": {"call": False, "states_missing": ["weight", "height"],
                                  "rule": "input-derived: neither weight nor height (nor a BMI) remains, so a BMI "
                                          "cannot be computed; no calculate_bmi call, say what is missing"},
            "original_partners": "the unedited source IDs, scored from the same label's full validation run",
            "edits": [{k: i[k] for k in ("source_id", "mid_line_edits", "removed_table_rows")}
                      for i in infos if "excluded" not in i],
            "records": probes, "records_sha256": sha256_text(canon(probes)), "reviewer": None}
    write_json(P1, spec)
    by_id = {r["id"]: r for r in sources}
    L = ["# P1 probe review packet (validation-derived synthetic probes)", "",
         f"{len(probes)} eligible of {len(sources)} grounded BMI sources; {len(spec['excluded'])} excluded. "
         "Check that every weight/height/BMI mention is gone, nothing else changed meaning, and the note still reads "
         "naturally. Then `uv run python scripts/w3_prep.py approve p1 --reviewer <name>` freezes the set.", ""]
    for e in spec["excluded"]:
        L.append(f"- excluded `{e['source_id']}`: {e['reason']}")
    for probe, info in zip(probes, [i for i in infos if "excluded" not in i]):
        src = by_id[probe["source_id"]]
        diff = difflib.unified_diff(src["note"].splitlines(), probe["note"].splitlines(), lineterm="", n=0)
        L += ["", f"## {probe['id']} (mid-line edits: {info['mid_line_edits']})", "",
              f"Question: {probe['question']}", "", f"Removed table rows: {info['removed_table_rows'] or 'none'}",
              "", "```diff", *list(diff)[2:], "```"]
    (REPORTS / "p1_review_packet.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote draft {P1.relative_to(ROOT)}: {len(probes)}/{len(sources)} eligible; "
          f"{sum(i['mid_line_edits'] > 0 for i in infos if 'excluded' not in i)} need mid-line review")


# ---------------------------------------------------------------- 8B template audit


def cmd_audit_8b(_: argparse.Namespace) -> None:
    """Every assistant turn supervised once, with exactly the inference prefix; stop/call tokens unchanged."""
    from clinqa.formatting import IGNORE_INDEX, build_conversation, encode_segments

    fmt = load_yaml(ROOT / "configs" / "format_w3_8b.yaml")
    tok, tok4 = load_tokenizer(fmt), load_tokenizer(load_yaml(ROOT / "configs" / "format_core.yaml"))
    system = (ROOT / fmt["system_prompt"]).read_text(encoding="utf-8").strip()
    problems, n_segments, longest, supervised = [], Counter(), 0, 0
    for r in filtered_train():
        msgs = build_conversation(r, system)
        segs = encode_segments(tok, msgs, max_length=fmt["max_length"], segmented=True)
        turns = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        prefixes = {render(tok, msgs[:i], add_generation_prompt=True) for i in turns}
        spans = [(s, a) for s in segs for a, _ in s.char_spans]
        if len(spans) != len(turns) or any(s.text[:a] not in prefixes for s, a in spans):
            problems.append(r["id"])
        for s in segs:
            text = tok.decode([t for t in s.labels if t != IGNORE_INDEX])
            if "<think>" in text or not text.endswith("<|im_end|>"):
                problems.append(r["id"])
            supervised += s.n_supervised
        n_segments[(r["answer_type"], len(segs))] += 1
        longest = max(longest, *(len(s.input_ids) for s in segs))
    gen_prompt = render(tok, prompt_messages(filtered_train()[0], system), add_generation_prompt=True)
    tokens = {t: (tok.convert_tokens_to_ids(t), tok4.convert_tokens_to_ids(t))
              for t in ("<tool_call>", "<|im_end|>", "<|endoftext|>")}
    report = {"tokenizer": fmt["tokenizer"], "tokenizer_revision": fmt["tokenizer_revision"],
              "chat_template_kwargs": fmt["chat_template_kwargs"], "n_records": len(filtered_train()),
              "problems": sorted(set(problems)), "segments_by_type": {f"{k[0]}:{k[1]}": v for k, v in
                                                                      sorted(n_segments.items())},
              "max_sequence_tokens": longest, "max_length": fmt["max_length"], "supervised_tokens": supervised,
              "generation_prompt_ends_with_empty_think": gen_prompt.endswith("<think>\n\n</think>\n\n"),
              "token_ids_8b_vs_4b": tokens,
              "call_prefix_is_single_token": tok("<tool_call>", add_special_tokens=False)["input_ids"]
              == [tokens["<tool_call>"][0]],
              "passed": False}
    report["passed"] = (not report["problems"] and longest <= fmt["max_length"]
                        and report["generation_prompt_ends_with_empty_think"] and report["call_prefix_is_single_token"]
                        and all(a == b for a, b in tokens.values()))
    write_json(REPORTS / "mask_audit_8b.json", report)
    print(json.dumps({k: report[k] for k in ("passed", "segments_by_type", "max_sequence_tokens", "problems")}))
    if not report["passed"]:
        raise SystemExit(1)


# ---------------------------------------------------------------- approvals and readiness


def _approvals() -> dict:
    return json.loads(APPROVALS.read_text()) if APPROVALS.exists() else {}


def cmd_approve(a: argparse.Namespace) -> None:
    approvals = _approvals()
    if a.kind in approvals:
        raise SystemExit(f"{a.kind} already approved; a changed artifact needs a new version name")
    if a.kind == "prompt_v3":
        approvals[a.kind] = {"path": str(PROMPT_V3.relative_to(ROOT)), "sha256": sha256_file(PROMPT_V3)}
    elif a.kind == "fewshot":
        spec = json.loads(FEWSHOT.read_text())
        if approvals.get("prompt_v3", {}).get("sha256") != sha256_file(PROMPT_V3):
            raise SystemExit("approve prompt_v3 first: the demonstrations are measured under the v3 prompt")
        if spec["messages_sha256"] != sha256_text(canon(spec["messages"])):
            raise SystemExit("demonstration messages were edited after drafting; rebuild them")
        spec.update(status="approved", reviewer=a.reviewer)
        write_json(FEWSHOT, spec)
        approvals[a.kind] = {"path": str(FEWSHOT.relative_to(ROOT)), "sha256": sha256_file(FEWSHOT)}
    elif a.kind == "p1":
        spec = json.loads(P1.read_text())
        by_id = {r["id"]: r for r in load_split("val")}
        problems = {p["id"]: probe_problems(p, by_id[p["source_id"]]) for p in spec["records"]}
        if any(problems.values()) or spec["records_sha256"] != sha256_text(canon(spec["records"])):
            raise SystemExit(f"P1 checks fail or records changed: {[k for k, v in problems.items() if v]}")
        spec.update(status="frozen", reviewer=a.reviewer)
        write_json(P1, spec)
        approvals[a.kind] = {"path": str(P1.relative_to(ROOT)), "sha256": sha256_file(P1)}
    approvals[a.kind]["reviewer"] = a.reviewer
    write_json(APPROVALS, approvals)
    print(f"approved {a.kind}: {approvals[a.kind]}")


def status() -> dict[str, str]:
    approvals = _approvals()
    out = {}
    for kind, path in (("prompt_v3", PROMPT_V3), ("fewshot", FEWSHOT), ("p1", P1)):
        entry = approvals.get(kind)
        out[kind] = ("missing" if not path.exists() else "draft" if not entry else
                     "approved" if entry["sha256"] == sha256_file(path) else "CHANGED_AFTER_APPROVAL")
    out["trainfit"] = "approved" if TRAINFIT.exists() else "missing"
    if not REVIEW.exists():
        out["relabel"] = "missing"
    else:
        rows = read_jsonl(REVIEW)
        undecided = sum(r.get("accepted") not in (True, False) or not r.get("reviewer") for r in rows)
        out["relabel"] = "approved" if not undecided else f"draft ({undecided}/{len(rows)} undecided)"
    return out


def cmd_check(a: argparse.Namespace) -> None:
    s = status()
    for k, v in s.items():
        print(f"{k:10s} {v}")
    if REVIEW.exists():
        rows = read_jsonl(REVIEW)
        print(f"relabel    accepted={sum(r.get('accepted') is True for r in rows)} "
              f"rejected={sum(r.get('accepted') is False for r in rows)}")
    missing = [k for k in a.require or [] if s.get(k) != "approved"]
    if missing:
        raise SystemExit(f"not approved: {', '.join(missing)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("relabel-template", "fewshot", "trainfit", "p1", "audit-8b"):
        sub.add_parser(name)
    ap = sub.add_parser("approve")
    ap.add_argument("kind", choices=("prompt_v3", "fewshot", "p1"))
    ap.add_argument("--reviewer", required=True)
    ck = sub.add_parser("check")
    ck.add_argument("--require", nargs="*", choices=("prompt_v3", "fewshot", "p1", "trainfit", "relabel"))
    a = p.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    {"relabel-template": cmd_relabel_template, "fewshot": cmd_fewshot, "trainfit": cmd_trainfit, "p1": cmd_p1,
     "audit-8b": cmd_audit_8b, "approve": cmd_approve, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
