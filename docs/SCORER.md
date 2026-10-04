# Scorers

Four scorers exist. **v2.1 is the reported scorer, and v1 is always reported beside it.**

| Scorer | Code | Status |
|---|---|---|
| **v1** | `src/clinqa/metrics.py` (`score_example`) | The assignment's five metrics; written to `outputs/<label>/<split>/metrics.json` and `scored.jsonl` by every evaluation |
| Bounded-contract v2 | `src/clinqa/scoring_v2.py`, `scripts/legacy/rescore_validation_v2.py` | Historical: leaves 25–42% of items in `review`, so it cannot rank models |
| **v2.1** | `src/clinqa/scorer_v2.py`, `scripts/score_v2.py`, `scripts/score_v21_val.py` | Reported diagnostic; frozen as sha256 `b15db3d0…` (`reports/scorer_v2/scorer_v2.1.sha256`) |
| v2.2 prototype | `scripts/scorer_v22.py` (`--scorer 2.2`) | Not adopted (D-096): v2.1 plus numeric contradiction guards |

- Known v2.1 defects: [SCORER_V2_1_KNOWN_ISSUES.md](SCORER_V2_1_KNOWN_ISSUES.md).
- Full validation record: [../reports/scorer_v2/VALIDATION.md](../reports/scorer_v2/VALIDATION.md).

## v1 → v2.1 at a glance, by answer type

**The core change.** v1 asks whether the answer *reproduces the gold answer*. v2.1 asks whether it *answers the question correctly given the input*: keys are computed from the note, table and question before the prediction is read.

On the clean holdout, agreement with blinded adjudication rises from 65.7% (v1) to 92.6% (v2.1); on its numeric items, from 51% to 97%. What remains is mostly one blind spot: v2.1 checks the parts of an answer that the question asks for, not every claim the answer makes.

Validation counts below are correct under v1 → v2.1. In brackets: items v2.1 newly passes / newly fails.

| Type | v1 problem | What v2.1 fixes | What remains (see the issue list) |
|---|---|---|---|
| **Extractive** | Requires the gold's numbers at the gold's precision, with its direction words in the same sentence, so correct answers in other wording fail. Base model: 39/100 under v1 vs 93/100 under v2.1 (+55 / −1); about 58 of base's 61 wave-1 failures were false fails | Value and status checks from the table; a negation-aware state reader that binds across sentences; "most abnormal" accepts relative or absolute deviation | Short answers on free-text questions can fail the text fallback's term recall (S-10). Partial keys can pass incomplete answers (S-03). Canonical ranges are assumed for note-only labs (S-11) |
| **Numeric** | Every number the gold derives must reappear, including incidental comparisons and the gold's choice of fold vs percent. 51% agreement on clean numeric items. F ep2: 29 → 46 of 50 (+17 / 0) | Only the asked results are required. Each operation (difference, ratio, percentage, percentage points) is computed from the input with its own tolerance | **False extra claims are not checked** (S-01). Any in-band number passes even with a contradicting one (S-02). Unrecognised clauses still count as correct (S-03). The last assertion can pick the wrong statement (S-06). Numeric accuracy is therefore not established |
| **Tool** | Q5 records, whose gold arguments are absent from the input, are scored as "should call with the gold arguments", which rewards fabrication. Strict ±0.05 arguments and the exact displayed result fail benign imperial rounding. F ep2: 54 → 61 of 62; the 7 gains are exactly the 7 Q5 records | Q5 records expect abstention. Calls are scored on the executed outcome (result within tolerance, arguments grounded in the input, result reported). The stated status must not contradict the input | The status reader can bind "reduced renal function" to creatinine as low, failing correct conversions (S-04). A Q5 abstention that still states invented values passes (S-17). Strict v1 and tolerant v2.1 tool counts are both reported |
| **Uncertain** | Mostly sound: an abstention phrase, the missing field named, no fabricated value. Its gaps: a definite "safe to give" conclusion could pass, and an unsupported call was not checked | Keeps the v1 checks and adds a failure for unsupported calls and for definite conclusions despite missing information. Base: 25 → 23 of 38 (0 / −2, stricter); SFT models unchanged | The missing-field taxonomy is narrow (S-09), and the expected category is inferred from the gold wording (S-12). Hedged phrasings can be mis-scored (S-08) |

**How to read the two scorers together.** Large effects such as base vs SFT on tool use, and the Q5 fabrication contrasts, agree under both scorers and were confirmed by reading every case. Differences of a few items, especially on numeric, depend on the scorer and are not interpreted.

## v1 (rule-based, gold-driven)

**Extractive.** Every number in the gold answer that is grounded in the input must appear in the prediction at the gold's precision. Direction words (high, low, normal) in the sentences holding those numbers must match the gold's. If the gold has no grounded number, a token-F1 of at least 0.5 is required.

**Numeric.** Every number the gold *derives* (one not present in the input) must appear at the gold's precision. A guard rejects a correct result attached to another analyte's value. If the gold derives nothing, at least half of its grounded numbers must appear. Direction words must match, as for extractive.

**Tool.** Each step is scored in turn:

1. selection: the gold tool is called;
2. arguments: valid and within ±0.05 of gold; for BMI, an exact metric conversion of an imperial input is also accepted;
3. execution;
4. the result appears in the answer.

Tool end-to-end requires all four.

**Uncertainty.** The answer must use an abstention phrase, name the missing field, and state no fabricated value.

**Limitation.** v1 rewards reproducing the gold's wording and its incidental numbers. On the clean scorer holdout it agrees with blinded adjudication on only 65.7% of items, and on 51% of numeric items. Most of its errors are false fails.

## v2.1 (input-derived keys, question-driven)

The design follows MedCalc-Bench. Answer keys are computed from the inputs before any prediction is read, and checks are typed per answer type.

**1. Keys from the input** (`build_key`):

- Facts come from the table, from values stated in the question and from labs written in the note. Table rows carry their reference ranges; note-only labs get canonical ranges (S-11).
- The question is split into clauses. Each clause becomes a typed check: value, status, difference, ratio, percentage, percentage points, BMI, entity ("most abnormal", "closest to"), set ("which other values"), yes/no or threshold. A text fallback checks the gold's grounded numbers and key terms.
- "Most abnormal" accepts both relative and absolute deviation when the units agree.

**2. Tolerances.**

| Check | Tolerance |
|---|---|
| Integers and categories | Exact |
| Differences | ±max(0.051, 0.5%) |
| Ratios and percentages | ±5% |
| BMI | ±0.15 |

**3. Reading the answer** (`read`):

- Analytes are mentioned by alias.
- A state reader is negation-aware and binds states across sentences; the last assertion wins.
- Numbers are bound to the most recent analyte.

**4. Tool records** are scored on the executed outcome:

- right tool, valid schema, grounded arguments, result within tolerance, result reported, one call;
- no contradiction between the stated status and the input;
- Q5 records, whose gold arguments are absent from the input, expect abstention.

**5. Uncertain records** expect abstention naming the missing-field category.

**What "correct" means.** Every check built from the question passed. v2.1 measures the *asked part* of the answer: extra claims are not checked (S-01), and partial keys still count (S-03).

## Validation

Validation used blinded LLM adjudication with written rubrics, two independent rater groups per set, and the scorer frozen by sha256 before each holdout.

| Scorer | Development (tuned) | Holdout 1 (test) | Holdout 2 (test, unseen) |
|---|---|---|---|
| v1 | 65.8% (κ 0.17) | 67.5% (κ 0.34) | 65.7% (κ 0.26) |
| v2.0 | 99.2% | 76.7% (κ 0.48): overfit | — |
| v2.1 | 100% | 95.0% (tuned on it) | **92.6% (κ 0.72)**: 2 false passes, 6 false fails |

- **Gold self-check** (the gold answer scored as the prediction): train 98.6%, val 100%.
- **Numeric only**, on the 125 adjudicated items where both rater groups agree:

  | Scorer | False fails | False passes |
  |---|---|---|
  | v1 | 56 | 2 |
  | v2.1 | 3 | 3 |

**Caveats** (S-14, S-15):

- The holdouts are test items, and v2.1 was tuned on holdout 1, so test is reused.
- The rubric did not require unrequested claims to be true.
- The adjudicated answers came from wave-1 models. Later SFT models put their remaining numeric errors in extra claims, where v2.1 is blind.

## v2.2 prototype (not adopted)

`scripts/scorer_v22.py` runs v2.1 unchanged. On numeric answers only, it can turn a pass into a fail with two guards:

1. an abnormal-state claim must match the input;
2. a "by X" or "X above/below the limit" amount must equal |value − bound|.

Checks:

- gold self-check: 0 of 450 numeric golds fire;
- four new disagreements with the adjudicated reference, all verified real errors;
- six flips across the wave-4 validation arms, all real, none in Qwen3.5 arms.

It is not adopted because the guards were tuned on the same validation and adjudication sets, and no clean holdout exists. See VALIDATION.md §5.

## Commands

```bash
# Validation (guarded: exactly 250 IDs, no test labels, new output directory)
uv run python scripts/score_v21_val.py --out reports/<dir> --labels <label> [...]
# Test (only for frozen test runs)
uv run python scripts/score_v2.py --runs-dir outputs --split test --out reports/<dir> --labels <label>_test [...]
# Gold self-check and the scorer's own tests
uv run python scripts/scorer_v2_gold_check.py val
uv run pytest -q tests/test_scorer_v2.py tests/test_metrics.py
```
