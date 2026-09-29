"""Data-quality checks Q1-Q19. Each check returns a CheckResult with per-record flags.

Raw data is never modified; flags are written to reports/quality_flags.jsonl so later
phases can decide (per docs/DECISIONS.md) whether to filter, relabel, or keep records.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from clinqa import parsing as P
from clinqa import tools
from clinqa.analysis.common import CheckResult, Flag, mask_numbers, percentile, truncate
from clinqa.analysis.features import Features
from clinqa.analysis.stats import ANSWER_TYPES


@dataclass
class Context:
    data: dict[str, list[dict[str, Any]]]
    feats: dict[str, list[Features]]
    reference: list[dict[str, Any]]
    cfg: dict[str, Any]

    @property
    def splits(self) -> list[str]:
        return list(self.data)

    def records(self):
        """Yield (split, record, features) over all splits in order."""
        for s in self.splits:
            for r, f in zip(self.data[s], self.feats[s]):
                yield s, r, f

    def tol(self, key: str) -> Any:
        return self.cfg["tolerances"][key]


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol + 1e-9


def _decimals(num_text: str) -> int:
    return len(num_text.split(".")[1]) if "." in num_text else 0


# ---------------------------------------------------------------------------
# Q1 schema
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = {"id": str, "note": str, "table": dict, "question": str, "answer": str, "answer_type": str}


def q1_schema(ctx: Context) -> CheckResult:
    res = CheckResult(
        "Q1", "Schema validity", "error",
        "Required fields and types; answer_type enum; tool_calls present iff answer_type == tool_call; "
        "tool call shape; table type, headers and row lengths.",
        "Hard failure: the formatter asserts these invariants, so any violation must be fixed before Phase 2.",
    )
    empty_cells = Counter()
    for s, r, _ in ctx.records():
        rid = r.get("id", "?")
        problems = []
        for k, t in _REQUIRED_FIELDS.items():
            if k not in r:
                problems.append(f"missing field {k}")
            elif not isinstance(r[k], t):
                problems.append(f"{k} is {type(r[k]).__name__}, expected {t.__name__}")
            elif t is str and not r[k].strip():
                problems.append(f"empty {k}")
        extra = set(r) - set(_REQUIRED_FIELDS) - {"tool_calls"}
        if extra:
            problems.append(f"unexpected fields {sorted(extra)}")
        at = r.get("answer_type")
        if at not in ANSWER_TYPES:
            problems.append(f"invalid answer_type {at!r}")
        has_calls = "tool_calls" in r
        if has_calls != (at == "tool_call"):
            problems.append(f"tool_calls present={has_calls} but answer_type={at}")
        if has_calls:
            tcs = r["tool_calls"]
            if not isinstance(tcs, list) or not tcs:
                problems.append("tool_calls must be a non-empty list")
            else:
                for tc in tcs:
                    if not (isinstance(tc, dict) and isinstance(tc.get("tool"), str)
                            and isinstance(tc.get("arguments"), dict) and "result" in tc):
                        problems.append(f"malformed tool call {tc!r}")
        table = r.get("table", {})
        if isinstance(table, dict):
            tt = table.get("type")
            if tt not in P.EXPECTED_HEADERS:
                problems.append(f"invalid table type {tt!r}")
            elif table.get("headers") != P.EXPECTED_HEADERS[tt]:
                problems.append(f"headers {table.get('headers')} do not match type {tt}")
            rows = table.get("rows")
            if not isinstance(rows, list) or not rows:
                problems.append("table has no rows")
            else:
                for row in rows:
                    if not isinstance(row, list) or len(row) != len(table.get("headers", [])):
                        problems.append(f"row length mismatch {row!r}")
                    elif not all(isinstance(c, str) for c in row):
                        problems.append(f"non-string cell in {row!r}")
                    else:
                        for c_i, c in enumerate(row):
                            if not c.strip():
                                empty_cells[(row[0], table["headers"][c_i])] += 1
        for p in problems:
            res.flags.append(Flag("Q1", s, rid, p))
    res.metrics["empty_cells"] = {f"{k[0]} / {k[1]}": v for k, v in sorted(empty_cells.items())}
    if empty_cells:
        res.notes.append(
            "Empty table cells are schema-valid but must be rendered explicitly by the table serializer: "
            + ", ".join(f"{k} ({v})" for k, v in res.metrics["empty_cells"].items())
        )
    return res


# ---------------------------------------------------------------------------
# Q2 ids
# ---------------------------------------------------------------------------

def q2_ids(ctx: Context) -> CheckResult:
    res = CheckResult(
        "Q2", "ID uniqueness and naming", "error",
        "IDs are unique across all splits, prefixed with their split name, and contiguous.",
        "IDs are the join key for predictions and flags; violations must be fixed before Phase 2.",
    )
    seen: dict[str, str] = {}
    for s in ctx.splits:
        nums = []
        for r in ctx.data[s]:
            rid = r["id"]
            if rid in seen:
                res.flags.append(Flag("Q2", s, rid, f"duplicate id (also in {seen[rid]})"))
            seen[rid] = s
            m = re.fullmatch(rf"{s}_(\d+)", rid)
            if not m:
                res.flags.append(Flag("Q2", s, rid, f"id does not match '{s}_<n>'"))
            else:
                nums.append(int(m.group(1)))
        if nums and sorted(nums) != list(range(len(nums))):
            res.flags.append(Flag("Q2", s, "*", "ids are not contiguous from 0"))
    return res


# ---------------------------------------------------------------------------
# Q3 duplicates / near-duplicates, Q18 contamination
# ---------------------------------------------------------------------------

def _shingles(text: str, k: int) -> set[tuple[str, ...]]:
    toks = re.findall(r"\w+", text.lower())
    return {tuple(toks[i: i + k]) for i in range(max(0, len(toks) - k + 1))}


def _max_jaccard_against(
    query: list[set], index_sets: list[set], max_df: int
) -> list[tuple[float, int]]:
    inv: dict[tuple, list[int]] = defaultdict(list)
    for i, sh in enumerate(index_sets):
        for g in sh:
            inv[g].append(i)
    out = []
    for q in query:
        hits = Counter()
        for g in q:
            lst = inv.get(g)
            if lst and len(lst) <= max_df:
                hits.update(lst)
        best = (0.0, -1)
        for i, _ in sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))[:10]:
            j = len(q & index_sets[i]) / len(q | index_sets[i])
            if j > best[0]:
                best = (j, i)
        out.append(best)
    return out


def q3_duplicates(ctx: Context) -> CheckResult:
    res = CheckResult(
        "Q3", "Duplicates and near-duplicates", "warn",
        "Exact duplicate (note, question) pairs within a split; note reuse; number-masked note templates; "
        "near-duplicate notes across splits (word 5-gram Jaccard vs. train).",
        "Splits are used as-is (assignment requirement). Near-duplicate val/test notes are reported so that test "
        "scores can be read as in-distribution generalisation rather than novel-note generalisation.",
    )
    nd = ctx.cfg["near_duplicate"]
    for s in ctx.splits:
        seen_pair: dict[tuple[str, str], str] = {}
        note_uses = Counter(r["note"] for r in ctx.data[s])
        for r in ctx.data[s]:
            key = (r["note"], r["question"])
            if key in seen_pair:
                res.flags.append(Flag("Q3", s, r["id"], f"exact duplicate of {seen_pair[key]} (note + question)"))
            else:
                seen_pair[key] = r["id"]
        res.metrics.setdefault("notes_reused_within_split", {})[s] = sum(1 for v in note_uses.values() if v > 1)
        masked = Counter(mask_numbers(r["note"]) for r in ctx.data[s])
        res.metrics.setdefault("distinct_note_templates_numbers_masked", {})[s] = {
            "distinct": len(masked), "records": len(ctx.data[s]),
        }
    if "train" in ctx.data:
        train_sh = [_shingles(r["note"], nd["shingle_size"]) for r in ctx.data["train"]]
        train_masked = {mask_numbers(r["note"]) for r in ctx.data["train"]}
        for s in ctx.splits:
            if s == "train":
                continue
            q_sh = [_shingles(r["note"], nd["shingle_size"]) for r in ctx.data[s]]
            best = _max_jaccard_against(q_sh, train_sh, nd["max_document_frequency"])
            js = [b[0] for b in best]
            res.metrics.setdefault("max_jaccard_vs_train", {})[s] = {
                "p50": round(percentile(js, 50), 3), "p90": round(percentile(js, 90), 3),
                "p99": round(percentile(js, 99), 3), "max": round(max(js), 3),
                f">= {nd['jaccard_threshold']}": sum(1 for j in js if j >= nd["jaccard_threshold"]),
            }
            res.metrics.setdefault("note_template_shared_with_train", {})[s] = sum(
                1 for r in ctx.data[s] if mask_numbers(r["note"]) in train_masked
            )
            for r, (j, i) in zip(ctx.data[s], best):
                if j >= nd["jaccard_threshold"]:
                    res.flags.append(Flag("Q3", s, r["id"], f"near-duplicate of {ctx.data['train'][i]['id']} (Jaccard {j:.3f})"))
    return res


def q18_contamination(ctx: Context) -> CheckResult:
    res = CheckResult(
        "Q18", "Cross-split contamination", "warn",
        "Exact note, (note, question) and (question, answer) overlap between train and val/test.",
        "Reported, not removed (splits are used as-is). Any overlap is excluded from headline test metrics "
        "or reported separately.",
    )
    if "train" not in ctx.data:
        return res
    train_notes = {r["note"]: r["id"] for r in ctx.data["train"]}
    train_pairs = {(r["note"], r["question"]): r["id"] for r in ctx.data["train"]}
    train_qa = {(r["question"], r["answer"]): r["id"] for r in ctx.data["train"]}
    for s in ctx.splits:
        if s == "train":
            continue
        counts = Counter()
        for r in ctx.data[s]:
            if (r["note"], r["question"]) in train_pairs:
                counts["note+question"] += 1
                res.flags.append(Flag("Q18", s, r["id"], f"same note+question as {train_pairs[(r['note'], r['question'])]}"))
            elif r["note"] in train_notes:
                counts["note"] += 1
                res.flags.append(Flag("Q18", s, r["id"], f"same note as {train_notes[r['note']]}"))
            if (r["question"], r["answer"]) in train_qa:
                counts["question+answer"] += 1
        res.metrics[s] = {k: counts.get(k, 0) for k in ("note+question", "note", "question+answer")}
    return res


# ---------------------------------------------------------------------------
# Q4 tool results, Q5 argument grounding, Q6 rounding, Q9 answer uses result
# ---------------------------------------------------------------------------

def _half_up(x: float, nd: int) -> float:
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def q4_tool_results(ctx: Context) -> CheckResult:
    res = CheckResult(
        "Q4", "Gold tool results reproduce with our tools", "error",
        "Every gold tool call is executed with clinqa.tools and compared with the gold `result`.",
        "Any mismatch would mean the gold answer and the executable tool disagree; mismatches would be listed in "
        "tests/test_tools.py and excluded from tool-result metrics.",
    )
    n = 0
    for s, r, _ in ctx.records():
        for tc in r.get("tool_calls", []):
            n += 1
            got = tools.execute_tool(tc["tool"], tc["arguments"])
            gold = tc["result"]
            if isinstance(got, str) or not isinstance(gold, (int, float)) or abs(got - gold) > 1e-9:
                detail = f"{tc['tool']}({json.dumps(tc['arguments'], ensure_ascii=False)}) -> {got!r}, gold {gold!r}"
                if tc["tool"] == "unit_convert" and not isinstance(got, str):
                    raw = tools._CONVERSIONS[(tools.normalize_unit(tc["arguments"]["from_unit"]),
                                              tools.normalize_unit(tc["arguments"]["to_unit"]),
                                              tools.normalize_substance(tc["arguments"].get("substance")))](tc["arguments"]["value"])
                    detail += f" (unrounded {raw!r}, half-up {_half_up(raw, tools.UNIT_CONVERT_DECIMALS)!r})"
                res.flags.append(Flag("Q4", s, r["id"], detail))
    res.metrics["tool_calls_checked"] = n
    return res


_SUBSTANCE_ROWS = {"glucose": {"Glucose (fasting)", "Fasting Glucose"}, "creatinine": {"Creatinine"}, "cholesterol": {"Total Cholesterol"}}
_SUBSTANCE_ANALYTE = {"glucose": "Glucose (fasting)", "creatinine": "Creatinine", "cholesterol": "Total Cholesterol"}
_SOURCE_PRIORITY = {"note": 0, "table": 1, "question": 2}
_MATCH_PRIORITY = {"exact": 0, "converted": 1, "approx": 2}


def _ground_body_arg(target: float, kind: str, ms: list[P.Measurement], exact: float, approx: float) -> dict[str, Any] | None:
    metric_unit = "kg" if kind == "weight" else "cm"
    best = None
    for m in ms:
        if m.kind != kind:
            continue
        if m.unit == metric_unit:
            match = "exact" if _close(m.value, target, exact) else None
        else:
            conv = m.metric_value
            match = "converted" if _close(round(conv, 1), target, exact) else ("approx" if _close(conv, target, approx) else None)
        if match is None:
            continue
        cand = {"source": m.source, "unit": m.unit, "match": match, "value": m.value, "labeled": m.labeled,
                "diff": round(m.metric_value - target, 3)}
        key = (_SOURCE_PRIORITY[m.source], _MATCH_PRIORITY[match], not m.labeled)
        if best is None or key < best[0]:
            best = (key, cand)
    return best[1] if best else None


def bmi_grounding(r: dict[str, Any], f: Features, ctx: Context) -> dict[str, Any]:
    """Locate each calculate_bmi argument in note/table/question (shared by Q5, Q6, Q8)."""
    tc = next(tc for tc in r["tool_calls"] if tc["tool"] == "calculate_bmi")
    a = tc["arguments"]
    w = _ground_body_arg(a["weight_kg"], "weight", f.measurements, ctx.tol("exact"), ctx.tol("conversion_approx"))
    h = _ground_body_arg(a["height_cm"], "height", f.measurements, ctx.tol("exact"), ctx.tol("conversion_approx"))
    return {"weight": w, "height": h, "args": a, "result": tc["result"]}


def _grounding_label(g: dict[str, Any]) -> str:
    parts = []
    for k in ("weight", "height"):
        x = g[k]
        parts.append(f"{k}:missing" if x is None else f"{k}:{x['source']}/{x['unit']}")
    return " ".join(parts)


def q5_arg_grounding(ctx: Context) -> CheckResult:
    res = CheckResult(
        "Q5", "Tool arguments are grounded in the input", "error",
        "Gold-anchored reverse matching: each gold argument must be recoverable from a value in the note, table or "
        "question (directly, or after lb->kg / in->cm conversion). Ungrounded arguments cannot be produced by a "
        "model that reads the input and therefore teach argument hallucination.",
        "Raw Core keeps all rows. The optional q5_filtered view excludes train candidates only and writes "
        "review packets; val/test remain unchanged. Report full-set and Q5-grounded diagnostic metrics.",
    )
    patterns: dict[str, Counter] = defaultdict(Counter)
    question_only = Counter()
    for s, r, f in ctx.records():
        for tc in r.get("tool_calls", []):
            if tc["tool"] == "calculate_bmi":
                g = bmi_grounding(r, f, ctx)
                label = _grounding_label(g)
                patterns[s][label] += 1
                missing = [k for k in ("weight", "height") if g[k] is None]
                if missing:
                    documented = sorted({f"{m.value} {m.unit}" for m in f.measurements if m.kind in missing and not m.equivalent})
                    res.flags.append(Flag(
                        "Q5", s, r["id"],
                        f"calculate_bmi {json.dumps(g['args'])}: {', '.join(missing)} not found in input"
                        + (f"; input documents {documented}" if documented else "; no such measurement in input"),
                    ))
                elif all(g[k]["source"] == "question" for k in ("weight", "height")):
                    question_only[s] += 1
            elif tc["tool"] == "unit_convert":
                a = tc["arguments"]
                sub = tools.normalize_substance(a.get("substance"))
                v = a.get("value")
                where = None
                for row in r["table"]["rows"]:
                    if row[0] in _SUBSTANCE_ROWS.get(sub, set()) and P.parse_float(row[1]) is not None \
                            and _close(P.parse_float(row[1]), v, ctx.tol("exact")):
                        where = "table"
                        break
                if where is None and sub in _SUBSTANCE_ANALYTE:
                    if any(_close(P.parse_float(x), v, ctx.tol("exact")) for x in P.note_analyte_values(r["note"], _SUBSTANCE_ANALYTE[sub])):
                        where = "note"
                if where is None and any(_close(x, v, ctx.tol("exact")) for x in P.extract_numbers(r["question"])):
                    where = "question"
                patterns[s][f"unit_convert value:{where or 'missing'}"] += 1
                if where is None:
                    res.flags.append(Flag("Q5", s, r["id"], f"unit_convert value {v} ({sub}) not found in input"))
    res.metrics["grounding_patterns"] = {s: dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))) for s, c in patterns.items()}
    res.metrics["bmi_args_only_in_question"] = {s: question_only.get(s, 0) for s in ctx.splits}
    return res


def q6_rounding(ctx: Context) -> CheckResult:
    res = CheckResult(
        "Q6", "Imperial -> metric rounding convention", "info",
        "For BMI arguments recovered via lb->kg or in->cm, compare round(converted, 1) with the gold metric value, "
        "and measure how often a model that converts exactly (e.g. via unit_convert, 2 dp) would get a different BMI "
        "than the gold result.",
        "Evaluation accepts an imperial-derived argument within half a unit of the last digit of either the gold value "
        "or round(exact conversion, 1) (D-027); no universal tolerance is applied (P-009).",
    )
    stats = Counter()
    diffs = []
    bmi_diff = Counter()
    for s, r, f in ctx.records():
        if not any(tc["tool"] == "calculate_bmi" for tc in r.get("tool_calls", [])):
            continue
        g = bmi_grounding(r, f, ctx)
        chain_args = {}
        for k in ("weight", "height"):
            x = g[k]
            if x is None:
                continue
            if x["unit"] in ("lb", "in"):
                stats[f"{k}:{x['match']}"] += 1
                diffs.append(abs(x["diff"]))
                if x["match"] == "approx":
                    res.flags.append(Flag("Q6", s, r["id"], f"{k} {x['value']} {x['unit']} -> {x['diff']:+.3f} vs gold after conversion"))
                chain_args[k] = tools.unit_convert(x["value"], x["unit"], "kg" if k == "weight" else "cm")
            else:
                chain_args[k] = x["value"]
        if len(chain_args) == 2 and any(g[k]["unit"] in ("lb", "in") for k in ("weight", "height")):
            chained = tools.calculate_bmi(chain_args["weight"], chain_args["height"])
            d = round(chained - g["result"], 1)
            bmi_diff[f"{d:+.1f}"] += 1
    res.metrics["converted_args"] = dict(sorted(stats.items()))
    res.metrics["max_abs_metric_diff"] = round(max(diffs), 3) if diffs else 0.0
    res.metrics["chained_bmi_minus_gold"] = dict(sorted(bmi_diff.items()))
    return res


_ANSWER_NUM_RE = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?")


def q9_answer_uses_result(ctx: Context) -> CheckResult:
    res = CheckResult(
        "Q9", "Final answer incorporates the tool result", "warn",
        "The gold answer of every tool_call example should state the tool result.",
        "Examples failing this would teach the model to ignore tool output; they are flagged for exclusion from "
        "'answer incorporates result' metrics.",
    )
    tol = ctx.tol("tool_result_in_answer")
    for s, r, _ in ctx.records():
        for tc in r.get("tool_calls", []):
            nums = [P.parse_float(x) for x in _ANSWER_NUM_RE.findall(r["answer"])]
            if not any(n is not None and _close(n, tc["result"], tol) for n in nums):
                res.flags.append(Flag("Q9", s, r["id"], f"result {tc['result']} not stated in answer: {truncate(r['answer'], 120)}"))
    return res


# ---------------------------------------------------------------------------
# Q7 inline BMI, Q8 label consistency
# ---------------------------------------------------------------------------

def _primary(ms: list[P.Measurement], kind: str) -> P.Measurement | None:
    cands = [m for m in ms if m.kind == kind and not m.equivalent]
    if not cands:
        return None
    labeled = [m for m in cands if m.labeled]
    return (labeled or cands)[0]


def q7_inline_bmi(ctx: Context) -> CheckResult:
    res = CheckResult(
        "Q7", "BMI stated in the note vs. computed BMI", "warn",
        "Notes sometimes state a BMI. It is compared with the BMI computed from the note's own weight and height, "
        "and with the gold result for calculate_bmi examples.",
        "Stated BMIs are distractors: training targets always come from the tool, never from a stated BMI. "
        "Inconsistent stated BMIs are documented; notes are not edited.",
    )
    tol = ctx.tol("inline_bmi_mismatch")
    compared = agree = 0
    gold_cmp = Counter()
    buckets = Counter()
    uncertain_with_stated = Counter()
    for s, r, f in ctx.records():
        if not f.inline_bmi:
            continue
        w, h = _primary(f.note_measurements, "weight"), _primary(f.note_measurements, "height")
        if w and h:
            computed = tools.calculate_bmi(w.metric_value, h.metric_value)
            compared += 1
            best = min(f.inline_bmi, key=lambda b: abs(b - computed))
            gap = abs(best - computed)
            if gap > tol:
                buckets["0.5-1" if gap < 1 else "1-2" if gap < 2 else "2-5" if gap < 5 else ">=5"] += 1
                res.flags.append(Flag("Q7", s, r["id"], f"note states BMI {f.inline_bmi} but {w.value} {w.unit} / {h.value} {h.unit} gives {computed}"))
            else:
                agree += 1
        for tc in r.get("tool_calls", []):
            if tc["tool"] == "calculate_bmi":
                gold_cmp["agrees" if any(abs(b - tc["result"]) <= tol for b in f.inline_bmi) else "disagrees"] += 1
        if r["answer_type"] == "uncertain" and f.is_bmi_question:
            uncertain_with_stated[s] += 1
            res.flags.append(Flag("Q7", s, r["id"], f"uncertain BMI question but note states BMI {f.inline_bmi}"))
    res.metrics["notes_with_stated_bmi_and_measurements"] = compared
    res.metrics["stated_bmi_consistent"] = agree
    res.metrics["inconsistency_size_bmi_units"] = {k: buckets.get(k, 0) for k in ("0.5-1", "1-2", "2-5", ">=5")}
    res.metrics["tool_call_stated_bmi_vs_gold_result"] = dict(gold_cmp)
    res.metrics["uncertain_bmi_questions_with_stated_bmi"] = {s: uncertain_with_stated.get(s, 0) for s in ctx.splits}
    return res


def _question_drugs(question: str, meds: list[P.Medication]) -> list[P.Medication]:
    """Medications the question asks the dose of ('dose of X' / 'X dose'), else any medication it names."""
    q = question.lower()
    names = set(re.findall(r"dos(?:e|age|ing) of (?:the )?([a-z][a-z-]+)", q)) | set(re.findall(r"([a-z][a-z-]+) (?:dose|dosage)\b", q))
    if names & {m.name.split()[0] for m in meds}:
        return [m for m in meds if m.name.split()[0] in names]
    return [m for m in meds if re.search(rf"\b{re.escape(m.name.split()[0])}\b", q)]


def q8_label_consistency(ctx: Context) -> CheckResult:
    res = CheckResult(
        "Q8", "answer_type label consistency", "warn",
        "Uncertain examples must really lack the information they say is missing (height/weight, dose, allergy "
        "documentation); BMI questions should be tool_call iff weight and height are both available.",
        "Contradictions are candidates for exclusion from training (P-012). The BMI decision table quantifies how "
        "cleanly the data separates 'call the tool' from 'state what is missing'.",
    )
    decision = Counter()
    verified = Counter()
    for s, r, f in ctx.records():
        input_ms = [m for m in f.measurements if m.source in ("note", "table")]
        has_w = any(m.kind == "weight" for m in f.measurements)
        has_h = any(m.kind == "height" for m in f.measurements)
        is_bmi_call = any(tc["tool"] == "calculate_bmi" for tc in r.get("tool_calls", []))
        if f.is_bmi_question and (is_bmi_call or r["answer_type"] == "uncertain"):
            decision[(r["answer_type"], "both present" if has_w and has_h else "missing")] += 1
        if r["answer_type"] != "uncertain":
            continue
        cat = f.uncertain_category
        if cat in ("height", "weight", "height_and_weight"):
            present = [k for k in ("height", "weight") if k in cat and any(m.kind == k for m in input_ms)]
            if present:
                vals = sorted({f"{m.value} {m.unit}" for m in input_ms if m.kind in present})
                res.flags.append(Flag("Q8", s, r["id"], f"answer says {cat} missing but note documents {vals}"))
            else:
                verified[cat] += 1
        elif cat == "allergy":
            if f.allergy in ("nkda", "documented"):
                res.flags.append(Flag("Q8", s, r["id"], f"answer says allergy info missing but note allergy status is {f.allergy}"))
            else:
                verified[cat] += 1
        elif cat == "dose":
            qd = _question_drugs(r["question"], f.medications)
            if not qd:
                verified["dose (drug not matched; unverified)"] += 1
            elif any(m.has_dose for m in qd):
                res.flags.append(Flag("Q8", s, r["id"], f"answer says dose missing but medication list has {[m.raw for m in qd if m.has_dose]}"))
            else:
                verified[cat] += 1
        else:
            verified[f"{cat} (not machine-verifiable)"] += 1
    res.metrics["bmi_decision_table"] = {f"{k[0]} | weight+height {k[1]}": v for k, v in sorted(decision.items())}
    res.metrics["uncertain_verified"] = dict(sorted(verified.items()))
    return res


# ---------------------------------------------------------------------------
# Q10 numeric arithmetic, Q11 extractive grounding, Q12 note vs table
# ---------------------------------------------------------------------------

# A numeric token must never backtrack to a shorter decimal when a suffix
# assertion fails (e.g. 4.97x must not become 4.9). Permit sentence-final dots.
_N = r"(?<![\w.])([+\-\u2212]?\d+(?:\.\d+)?)(?!\d|\.\d)"
_EQ_PATTERNS = [
    ("pct_change", re.compile(r"\(\s*" + _N + r"\s*[-\u2212\u2013]\s*" + _N + r"\s*\)\s*/\s*" + _N + r"\s*[\u00d7x*]\s*100\s*[\u2248=~]\s*" + _N + r"\s*%")),
    ("pct", re.compile(_N + r"\s*/\s*" + _N + r"\s*[\u00d7x*]\s*100\s*[\u2248=~]\s*" + _N + r"\s*%")),
    ("diff", re.compile(_N + r"\s*[-\u2212\u2013]\s*" + _N + r"\s*=\s*" + _N + r"(?!\s*[/\u00d7x*]\s*[+\-\u2212]?\d)")),
    ("ratio", re.compile(_N + r"\s*[\u00f7/]\s*" + _N + r"\s*[\u2248=~]\s*" + _N + r"(?!\s*%)(?!\s*[\u00d7x*]\s*[+\-\u2212]?\d)")),
]
_BY_RE = re.compile(
    r"\bby (?:approximately |about |roughly |~)?" + _N + r"(?!\.?\d)(?!\s*%)(?!\s*percent)(?!\s*(?:times|fold|-fold|x\b))"
)
_DIVIDE_BEFORE_RE = re.compile(r"(?:divid\w*|multipl\w*)(?:(?!\.\s)[^;]){0,30}$", re.I)


def _eq_ok(kind: str, nums: list[str]) -> tuple[bool, float]:
    v = [float(x.replace("\u2212", "-")) for x in nums]
    if kind == "pct_change":
        computed = (v[0] - v[1]) / v[2] * 100 if v[2] else float("nan")
    elif kind == "pct":
        computed = v[0] / v[1] * 100 if v[1] else float("nan")
    elif kind == "diff":
        computed = v[0] - v[1]
    else:
        computed = v[0] / v[1] if v[1] else float("nan")
    stated = v[-1]
    tol = max(0.5 * 10 ** -_decimals(nums[-1]), 0.01 * abs(stated)) + 1e-9
    return abs(computed - stated) <= tol, computed


def q10_numeric_arithmetic(ctx: Context) -> CheckResult:
    res = CheckResult(
        "Q10", "Arithmetic in numeric_reasoning gold answers (heuristic)", "warn",
        "(a) explicit equations in gold answers (a - b = c, a / b x 100 = c%, (a - b) / b x 100 = c%, a / b = c) are "
        "recomputed; (b) every 'by D' amount must equal |x - y| for some pair of numbers stated in the answer, "
        "question or table (value or reference bound).",
        "Heuristic: only checks what the answer spells out, not clinical correctness. Equation mismatches are "
        "candidates for exclusion from training (P-012); unverified 'by D' claims are listed for manual review.",
    )
    eq_checked = by_checked = by_verified = 0
    tol_abs = ctx.tol("numeric_abs")
    for s, r, _ in ctx.records():
        if r["answer_type"] != "numeric_reasoning":
            continue
        a = r["answer"]
        spans = []
        for kind, pat in _EQ_PATTERNS:
            for m in pat.finditer(a):
                if any(m.start() < e and m.end() > b for b, e in spans):
                    continue
                spans.append((m.start(), m.end()))
                eq_checked += 1
                ok, computed = _eq_ok(kind, list(m.groups()))
                if not ok:
                    res.flags.append(Flag("Q10", s, r["id"], f"equation '{m.group(0)}' recomputes to {computed:.3f}"))
        pool = sorted(set(P.extract_numbers(a.replace(",", "")) + P.extract_numbers(r["question"])
                          + [x for row in r["table"]["rows"] for cell in row[1:] for x in P.extract_numbers(cell)]))
        for m in _BY_RE.finditer(a):
            if _DIVIDE_BEFORE_RE.search(a[max(0, m.start() - 40): m.start()]):
                continue
            d = float(m.group(1).replace("\u2212", "-"))
            by_checked += 1
            tol = max(tol_abs, 0.5 * 10 ** -_decimals(m.group(1)))
            if any(_close(abs(x - y), d, tol) for x, y in itertools.combinations(pool, 2)):
                by_verified += 1
            else:
                sent = next((x for x in re.split(r"(?<=[.;])\s+", a) if m.group(0) in x), a)
                res.flags.append(Flag("Q10", s, r["id"], f"'by {m.group(1)}' is not a difference of any two input/answer numbers: {truncate(sent, 140)}"))
    res.metrics["equations_checked"] = eq_checked
    res.metrics["by_amounts_checked"] = by_checked
    res.metrics["by_amounts_verified"] = by_verified
    return res


def _input_numbers(r: dict[str, Any]) -> list[float]:
    nums = P.extract_numbers(r["note"]) + P.extract_numbers(r["question"])
    for row in r["table"]["rows"]:
        for cell in row:
            nums.extend(P.extract_numbers(cell))
    return nums


def q11_extractive_grounding(ctx: Context) -> CheckResult:
    res = CheckResult(
        "Q11", "Numbers in extractive answers appear in the input", "warn",
        "Every number in an extractive gold answer should be copied from the note, table or question. Missing numbers "
        "are split into (a) reference-range bounds that appear in other records' tables, i.e. the answer cites a "
        "standard range that this input does not show, and (b) other unsupported numbers.",
        "(a) is a mild form of external knowledge (standard lab ranges) and is kept; the system prompt may allow "
        "standard reference ranges (P-003). (b) are candidates for exclusion (P-012).",
    )
    known_bounds: set[float] = set()
    for _, r, _ in ctx.records():
        if r["table"]["type"] == "labs":
            for row in r["table"]["rows"]:
                known_bounds.update(P.extract_numbers(row[3]))
    checked = 0
    kinds = Counter()
    for s, r, _ in ctx.records():
        if r["answer_type"] != "extractive":
            continue
        checked += 1
        allowed = _input_numbers(r)
        missing = []
        for x in _ANSWER_NUM_RE.findall(r["answer"]):
            v = P.parse_float(x)
            if v is not None and not any(_close(v, y, 1e-9) for y in allowed):
                missing.append(x)
        if missing:
            kind = "reference range not in input" if all(float(x) in known_bounds for x in missing) else "unsupported number"
            kinds[kind] += 1
            res.flags.append(Flag("Q11", s, r["id"], f"{kind} {missing}: {truncate(r['answer'], 140)}"))
    res.metrics["extractive_answers_checked"] = checked
    res.metrics["flag_kinds"] = dict(sorted(kinds.items()))
    return res


def q12_note_table_conflicts(ctx: Context) -> CheckResult:
    res = CheckResult(
        "Q12", "Note text vs. table value conflicts", "warn",
        "When the note text states a value for an analyte that is also in the table (e.g. 'creatinine 3.6 mg/dL'), "
        "the two should agree.",
        "Conflicts make 'the patient's X' ambiguous. The table is treated as the source of truth in the system "
        "prompt (pending P-003); conflicting records are candidates for exclusion (P-012).",
    )
    compared = Counter()
    conflicts = Counter()
    rounded: list[str] = []
    for s, r, _ in ctx.records():
        for row in r["table"]["rows"]:
            name, value = row[0], row[1]
            if name not in P.NOTE_ANALYTE_ALIASES:
                continue
            note_vals = P.note_analyte_values(r["note"], name)
            if not note_vals:
                continue
            compared[name] += 1
            if name == "Blood Pressure":
                bad = [v for v in note_vals if P.parse_bp(v) != P.parse_bp(value)]
            else:
                tv = P.parse_float(value)
                bad = [v for v in note_vals if tv is None or not _close(float(v), tv, 1e-9)]
                # A note that restates the table value at lower precision ('ALT 133' for 133.3) is not a conflict.
                restated = [v for v in bad if tv is not None and _close(float(v), round(tv, _decimals(v)), 1e-9)]
                if restated:
                    rounded.append(f"{s}/{r['id']} {name}: note {restated} vs table {value}")
                bad = [v for v in bad if v not in restated]
            if bad:
                conflicts[name] += 1
                res.flags.append(Flag("Q12", s, r["id"], f"{name}: note says {bad} but table says {value}"))
    res.metrics["analytes_compared"] = dict(sorted(compared.items()))
    res.metrics["analytes_conflicting"] = dict(sorted(conflicts.items()))
    res.metrics["rounded_restatements"] = rounded
    return res


# ---------------------------------------------------------------------------
# Q13 unicode, Q14 table parseability, Q15 label ambiguity
# ---------------------------------------------------------------------------

_NOTABLE_CHARS = {
    "\u00b5": "MICRO SIGN", "\u03bc": "GREEK SMALL LETTER MU", "\u2013": "EN DASH", "\u2014": "EM DASH",
    "\u2212": "MINUS SIGN", "\u00d7": "MULTIPLICATION SIGN", "\u00f7": "DIVISION SIGN", "\u2248": "ALMOST EQUAL TO",
    "\u2264": "LESS-THAN OR EQUAL TO", "\u2265": "GREATER-THAN OR EQUAL TO", "\u00b0": "DEGREE SIGN",
    "\u00b2": "SUPERSCRIPT TWO", "\u00b3": "SUPERSCRIPT THREE",
}


def q13_unicode(ctx: Context) -> CheckResult:
    res = CheckResult(
        "Q13", "Unicode and unit-spelling variants", "info",
        "Inventory of look-alike characters per field. In particular the micro sign (U+00B5) and Greek mu (U+03BC) "
        "are both used for 'micro'.",
        "Raw text is kept as-is (the model should read what clinicians write). clinqa.tools.normalize_unit and the "
        "evaluator treat U+00B5, U+03BC and 'u' as equivalent.",
    )
    inv: dict[str, Counter] = defaultdict(Counter)
    for s, r, _ in ctx.records():
        fields = {
            "note": r["note"], "question": r["question"], "answer": r["answer"],
            "table": " ".join(" ".join(row) for row in r["table"]["rows"]),
            "tool_args": json.dumps([tc["arguments"] for tc in r.get("tool_calls", [])], ensure_ascii=False),
        }
        for fld, text in fields.items():
            for ch, name in _NOTABLE_CHARS.items():
                c = text.count(ch)
                if c:
                    inv[f"U+{ord(ch):04X} {name}"][fld] += c
        if r.get("tool_calls") and "\u03bc" in fields["tool_args"] and ("\u00b5" in r["question"] or "\u00b5" in r["answer"]):
            res.flags.append(Flag("Q13", s, r["id"], "question/answer use U+00B5 while tool arguments use U+03BC"))
    res.metrics["char_inventory"] = {k: dict(sorted(v.items())) for k, v in sorted(inv.items())}
    return res


def q14_table_parse(ctx: Context) -> CheckResult:
    res = CheckResult(
        "Q14", "Table values and reference ranges are parseable", "warn",
        "Values must be numeric (or systolic/diastolic for blood pressure); lab reference ranges must be 'a-b', "
        "'<x' or '>x'.",
        "Parseable values let the evaluator verify numeric answers against the table; unparseable cells are rendered "
        "verbatim.",
    )
    atypical = Counter()
    for s, r, f in ctx.records():
        if f.panel in ("mixed", "vitals+body"):
            atypical[f.panel] += 1
            res.flags.append(Flag("Q14", s, r["id"], f"atypical {r['table']['type']} table rows {[row[0] for row in r['table']['rows']]}"))
        for row in r["table"]["rows"]:
            name, value = row[0], row[1]
            if name == "Blood Pressure":
                if P.parse_bp(value) is None:
                    res.flags.append(Flag("Q14", s, r["id"], f"unparseable BP {value!r}"))
            elif P.parse_float(value) is None:
                res.flags.append(Flag("Q14", s, r["id"], f"unparseable value {name}={value!r}"))
            if r["table"]["type"] == "labs" and len(row) > 3 and P.parse_ref_range(row[3]) is None:
                res.flags.append(Flag("Q14", s, r["id"], f"unparseable reference range {name}={row[3]!r}"))
    res.metrics["atypical_tables"] = dict(atypical)
    return res


_RANGE_Q_RE = re.compile(r"within (?:the |its |their )?(?:normal )?(?:reference )?range|normal (?:reference )?range|reference range|above|below|elevated|abnormal", re.I)


def q15_label_ambiguity(ctx: Context) -> CheckResult:
    res = CheckResult(
        "Q15", "extractive vs. numeric_reasoning boundary", "info",
        "Many extractive questions ask whether a value is within its reference range, which is a threshold check "
        "(the stated definition of numeric_reasoning). Also lists number-masked question templates used under both labels.",
        "Labels are kept. The evaluator scores each type with its own metric, and the report discusses that "
        "'extractive' accuracy includes simple threshold comparisons.",
    )
    by_type = Counter()
    totals = Counter()
    tmpl_types: dict[str, set[str]] = defaultdict(set)
    for _, r, f in ctx.records():
        if r["answer_type"] in ("extractive", "numeric_reasoning"):
            totals[r["answer_type"]] += 1
            if _RANGE_Q_RE.search(r["question"]):
                by_type[r["answer_type"]] += 1
            tmpl_types[f.question_template].add(r["answer_type"])
    shared = sorted(t for t, ts in tmpl_types.items() if len(ts) > 1)
    for s, r, f in ctx.records():
        if f.question_template in shared and r["answer_type"] in ("extractive", "numeric_reasoning"):
            res.flags.append(Flag("Q15", s, r["id"], f"template used as both types ({r['answer_type']} here): {truncate(r['question'], 110)}"))
    res.metrics["range_wording_pct"] = {t: round(100 * by_type[t] / totals[t], 1) for t in sorted(totals)}
    res.metrics["templates_shared_between_types"] = shared
    return res


# ---------------------------------------------------------------------------
# Q16 plausibility, Q17 answer length
# ---------------------------------------------------------------------------

_DRUG_CLASSES = {
    "beta-blocker": {"metoprolol", "carvedilol", "bisoprolol"},
    "anticoagulant": {"warfarin", "apixaban", "enoxaparin"},
    "DPP-4 inhibitor": {"sitagliptin", "linagliptin"},
    "ACE inhibitor / ARB": {"lisinopril", "ramipril", "losartan"},
    "SSRI": {"sertraline", "citalopram"},
    "gabapentinoid": {"gabapentin", "pregabalin"},
    "statin": {"atorvastatin", "rosuvastatin"},
    "PPI": {"omeprazole", "pantoprazole"},
    "SGLT2 inhibitor": {"dapagliflozin", "empagliflozin"},
}
_SELF_ANNOTATED_RE = re.compile(r"coding error|possibly erroneous|likely erroneous|dual diagnosis|noted in (?:chart|records)|per (?:chart|records)|discrepan", re.I)
_FEMALE_ONLY_RE = re.compile(
    r"\bpregnan\w*|\bmenstrua\w*|\bovar(?:y|ies|ian)\b|\buter(?:us|ine)\b|\bendometri\w*|\bvagin\w*|\bcervix\b"
    r"|\bcervical (?:cancer|dysplasia|screening|cytology)",
    re.I,
)
_MALE_ONLY_RE = re.compile(r"\bprostat\w*|\bBPH\b|\btesticular\b|\bscrot\w*", re.I)


def q16_plausibility(ctx: Context) -> CheckResult:
    res = CheckResult(
        "Q16", "Clinical plausibility (synthetic-data artefacts)", "info",
        "Hematocrit/hemoglobin ratio, sex-specific conditions, self-annotated chart discrepancies, co-prescribed "
        "drugs of the same class, age extremes.",
        "Kept: these look like intentional noise that a grounded QA model must tolerate. They are not answer-type "
        "errors. Same-class drug combinations are relevant for the Stretch B reference lookup.",
    )
    lo, hi = ctx.tol("hct_hgb_ratio")
    counts = Counter()
    combos = Counter()
    for s, r, f in ctx.records():
        rows = {row[0]: P.parse_float(row[1]) for row in r["table"]["rows"]}
        if rows.get("Hemoglobin") and rows.get("Hematocrit"):
            ratio = rows["Hematocrit"] / rows["Hemoglobin"]
            if not lo <= ratio <= hi:
                counts["hct/hgb ratio out of range"] += 1
                res.flags.append(Flag("Q16", s, r["id"], f"Hct {rows['Hematocrit']} / Hgb {rows['Hemoglobin']} = {ratio:.2f}"))
        pmh = r["note"]
        if f.sex == "F" and _MALE_ONLY_RE.search(pmh):
            counts["male-only condition in female patient"] += 1
            res.flags.append(Flag("Q16", s, r["id"], "male-only condition (prostate/BPH) documented for a female patient"))
        if f.sex == "M" and _FEMALE_ONLY_RE.search(pmh):
            counts["female-only condition in male patient"] += 1
            res.flags.append(Flag("Q16", s, r["id"], "female-only condition documented for a male patient"))
        if _SELF_ANNOTATED_RE.search(pmh):
            counts["note self-annotates a chart discrepancy"] += 1
        if f.age is not None and (f.age < 18 or f.age > 100):
            counts["age < 18 or > 100"] += 1
            res.flags.append(Flag("Q16", s, r["id"], f"age {f.age}"))
        names = {m.name.split()[0] for m in f.medications}
        for cls, members in _DRUG_CLASSES.items():
            if len(names & members) >= 2:
                combos[cls] += 1
    res.metrics["counts"] = dict(sorted(counts.items()))
    res.metrics["notes_with_two_drugs_of_same_class"] = dict(sorted(combos.items(), key=lambda kv: (-kv[1], kv[0])))
    return res


def q17_answer_length(ctx: Context) -> CheckResult:
    res = CheckResult(
        "Q17", "Gold answer length", "info",
        "The spec describes gold answers as 1-3 sentences.",
        "Longer answers are kept (they are the gold targets) but set the generation budget: max_new_tokens is sized "
        "from the answer-length distribution, not from the spec.",
    )
    dist = Counter()
    for s, r, f in ctx.records():
        dist[r["answer_type"], min(f.answer_sentences, 5)] += 1
        if f.answer_sentences > 3:
            res.flags.append(Flag("Q17", s, r["id"], f"{f.answer_sentences} sentences / {f.answer_words} words"))
    res.metrics["sentences_by_type"] = {
        t: {("5+" if k == 5 else str(k)): v for (tt, k), v in sorted(dist.items()) if tt == t} for t in ANSWER_TYPES
    }
    return res


# ---------------------------------------------------------------------------
# Q19 reference file
# ---------------------------------------------------------------------------

_CATEGORY_PREFIX = {
    "drug_interaction": "drug_interaction", "contraindication": "contraindication", "guideline": "clinical_guideline",
    "dosing": "dosing_guideline", "reference": "clinical_reference",
}


def _drug_key(name: str) -> str:
    name = re.sub(r"\s+inhaler$", "", name.lower().strip())
    return name.replace(" ", "_")


def q19_reference(ctx: Context) -> CheckResult:
    res = CheckResult(
        "Q19", "Reference file (Stretch B) integrity and coverage", "warn",
        "Key uniqueness and grammar, category/prefix consistency, drug-pair order collisions, and how many notes "
        "contain a medication pair that the reference covers.",
        "Informs the reference_lookup key normalisation (order-insensitive drug pairs) and whether enough "
        "Stretch B examples can be synthesised from existing notes.",
    )
    keys = Counter(e["key"] for e in ctx.reference)
    for k, v in keys.items():
        if v > 1:
            res.flags.append(Flag("Q19", "reference", k, f"duplicate key ({v}x)"))
    pairs: dict[frozenset, list[str]] = defaultdict(list)
    vocab: set[str] = set()
    for e in ctx.reference:
        k = e["key"]
        if not re.fullmatch(r"[a-z0-9_]+(?::[a-z0-9_]+)+", k):
            res.flags.append(Flag("Q19", "reference", k, "key does not match [a-z0-9_]+(:[a-z0-9_]+)+"))
        prefix = k.split(":")[0]
        if _CATEGORY_PREFIX.get(prefix) != e.get("category"):
            res.flags.append(Flag("Q19", "reference", k, f"prefix {prefix!r} vs category {e.get('category')!r}"))
        if not str(e.get("value", "")).strip():
            res.flags.append(Flag("Q19", "reference", k, "empty value"))
        parts = k.split(":")
        if prefix == "drug_interaction":
            if len(parts) != 3:
                res.flags.append(Flag("Q19", "reference", k, "drug_interaction key must have exactly two drugs"))
            else:
                pairs[frozenset(parts[1:])].append(k)
                vocab.update(parts[1:])
    for ks in pairs.values():
        if len(ks) > 1:
            res.flags.append(Flag("Q19", "reference", ks[0], f"same drug pair under multiple keys {ks}"))
    res.metrics["entries"] = len(ctx.reference)
    res.metrics["by_category"] = dict(sorted(Counter(e["category"] for e in ctx.reference).items()))
    res.metrics["drug_interaction_vocabulary"] = len(vocab)
    order = Counter(
        "alphabetical" if k.split(":")[1] <= k.split(":")[2] else "non-alphabetical" for ks in pairs.values() for k in ks
    )
    res.metrics["pair_order"] = dict(sorted(order.items()))
    unit_variants = Counter()
    for e in ctx.reference:
        for u in re.findall(r"\b(?:ug|\u00b5g|\u03bcg|uL|\u00b5L|\u03bcL|umol|\u00b5mol|\u03bcmol)\b", e["value"]):
            unit_variants[u] += 1
    res.metrics["micro_unit_spellings_in_values"] = dict(sorted(unit_variants.items()))

    covered_notes = Counter()
    pair_hits = Counter()
    uncovered_drugs = Counter()
    for s, _, f in ctx.records():
        names = {_drug_key(m.name) for m in f.medications}
        hits = [p for p in pairs if p <= names]
        if hits:
            covered_notes[s] += 1
        for p in hits:
            pair_hits[":".join(sorted(p))] += 1
        for n in names - vocab:
            uncovered_drugs[n] += 1
    res.metrics["notes_with_covered_pair"] = {s: covered_notes.get(s, 0) for s in ctx.splits}
    res.metrics["top_covered_pairs"] = dict(sorted(pair_hits.items(), key=lambda kv: (-kv[1], kv[0]))[:15])
    res.metrics["note_drugs_absent_from_interaction_keys"] = dict(sorted(uncovered_drugs.items(), key=lambda kv: (-kv[1], kv[0]))[:20])
    return res


ALL_CHECKS = [
    q1_schema, q2_ids, q3_duplicates, q4_tool_results, q5_arg_grounding, q6_rounding, q7_inline_bmi,
    q8_label_consistency, q9_answer_uses_result, q10_numeric_arithmetic, q11_extractive_grounding,
    q12_note_table_conflicts, q13_unicode, q14_table_parse, q15_label_ambiguity, q16_plausibility,
    q17_answer_length, q18_contamination, q19_reference,
]


def run_checks(ctx: Context) -> list[CheckResult]:
    return [check(ctx) for check in ALL_CHECKS]
