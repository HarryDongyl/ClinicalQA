# Scorer v2: evaluate claims, not reference-answer style

Status: implemented bounded automatic checker with an explicit adjudication queue. It is not a universal semantic judge. Full semantic checkpoint selection remains blocked until unresolved obligations and sampled automatic decisions have been reviewed. All development and rescoring in this revision use validation; no test records or predictions are read by the new scoring command.

## What was wrong at the level of the objective

The desired quantity is whether an answer fulfills the question using supported facts and correct operations. The old proxy instead measures how closely the answer reproduces a particular reference explanation. These are different quantities:

1. The reference is one possible answer, not a complete specification. Its extra percentage, medication dose, intermediate division, or interpretation is not automatically required.
2. Correctness attaches to a claim: subject, value, unit, relation, and evidence. A number and a direction word need not occur in the same sentence; conversely, their appearance anywhere in the answer is not sufficient.
3. Negation changes meaning. “Not within range” entails abnormality; “not high” does not by itself entail normality. A failed keyword match is not evidence of an incorrect answer.
4. The input may contradict the reference or omit the reference range needed to reproduce it. The evaluator must expose that limitation instead of silently importing facts from gold.
5. Tool syntax, execution, answer completeness, factual grounding, uncertainty, and clinical interpretation are separate dimensions. One fluent final answer must not erase an unsupported call.

The redesign therefore uses a question contract, input-derived facts, typed checks, and pass/fail/review outcomes. It does not replace the old whitelist with an unrestricted whole-answer search, nor replace one unreliable accuracy number with another.

## Implemented architecture

`src/clinqa/scoring_v2.py` has three stages:

**Compile a contract before examining predictions.** Extract explicit measurements and ranges from structured tables and labeled lab statements in the note. Respect strict versus inclusive reference bounds. Resolve the question's target and requested operations. Compute differences, ratios, and percentages from these inputs. Store required slots, source facts, missing fields, and unresolved question obligations. Extractive/numeric contracts do not use the gold answer. Uncertainty category inference still uses the existing gold category heuristic; this dependency is recorded as a limitation.

**Check the candidate answer against the contract.** Track named analytes across sentences, preserving local coreference until another subject is named. Model range predicates as sets of possible states: not-normal means {low, high}, and not-high means {low, normal}. Preserve direction distinctions when the question asks for direction, but accept a binary out-of-range answer to a binary question. Bind derived numbers to the requested analyte and operation; verify explicit units, supplied reference boundaries, local arithmetic, and selected extra factual assertions. References and intermediate equation operands are not patient measurements or required answer numbers.

**Return evidence and unresolved scope.** Every check includes a kind, subject, status, reason, and evidence where available. Missing parsing coverage and open clinical interpretation produce `review`, with `correct: null`. They are not automatically wrong or correct. A demonstrable contradiction can fail a multipart answer even if another part remains unresolved. Explicit self-corrections with conflicting claims go to review rather than being treated as a confident paraphrase error.

The parser is deliberately bounded. An unfamiliar valid paraphrase can still require review. Open note extraction, clinical significance, ambiguous rankings, implicit units, cross-sentence references involving multiple subjects, and complex arithmetic are not fully solved by deterministic rules. Automatic pass means the supported contract checks passed; it is not certification that every clinical assertion is valid.

## Numeric rules

- Only requested results are mandatory. A difference question does not require an unasked percentage, intermediate denominator, or gold explanation.
- Absolute difference, percentage-point change, percentage relative to a boundary, and ratio are different operations. Question clauses bind operations to subjects; asking for a D-dimer difference and an aPTT factor does not require both operations for both tests.
- Reference bounds come from the corresponding input fact. If absent, the missing evidence is exposed; gold alone cannot supply it.
- “Most abnormal” needs a defined comparison scale. Comparing raw deviations across different units is generally not a well-defined ranking. Unique abnormal results can be identified automatically; ambiguous rankings or their comparative explanations require adjudication.
- Values/differences use a declared 0.05 absolute tolerance for the supported one-decimal tasks. Ratios use displayed rounding with at least one decimal of resolution; whole-number percentages are permitted. These are explicit engineering policies, not medically universal tolerances. They were not widened in response to the previously observed test conversion behavior.
- Extra false claims are still errors when the checker can establish them: wrong measurements, incorrect explicit arithmetic, incompatible units, false reference bounds, and false “all other values are normal” statements. General clinical claims need review; optional does not mean exempt from truthfulness.

## Tool, format, and uncertainty dimensions

The old scorer remains intact in `metrics.py` and is included as `legacy` on each new row. Its raw tool-selection, argument, execution and result-in-answer counters remain available. Schema validity, completion, and call budget are reported separately. This preserves the evidence for real SFT progress without making lexical answer matching responsible for protocol evaluation.

The new policy check refuses to count a gold-matching unsupported BMI call as successful. If input measurements are missing, an appropriate no-call response can succeed even if the original record was labeled `tool_call`. This is a **policy-success** measure; it must not be mislabeled as the old tool E2E measure. The report and JSON keep that distinction explicit. Existing tool conversion tolerances are unchanged.

When the old argument checker accepts a call but the independent grounding checker rejects it despite available inputs, v2 reports a verifier disagreement for review. It does not silently widen a tolerance or label a rounding disagreement as invented patient information. Validation case `val_104` exposes this disagreement in some adapters.

For uncertainty, the checker looks for the missing field and abstention, checks unsupported known/missing values using the existing detector, and rejects calls made despite missing required inputs. “Does not indicate tachycardia” is not an abstention. Unrecognized uncertainty language goes to review. The existing value detector and missing-field heuristics are not a complete claim-level entailment system; review must include apparently successful refusals as well as failures.

## Validation-only workflow

```bash
# Regression tests: synthetic v2 fixtures plus historical train/validation v1 checks.
.venv/bin/python -m pytest tests/test_scoring_v2.py tests/test_metrics.py

# Choose a NEW directory; the command refuses to overwrite prior reviews.
.venv/bin/python scripts/rescore_validation_v2.py --out reports/scorer_v2_validation_next
# Equivalent Make target:
make score-v2 OUT=reports/scorer_v2_validation_next
```

The command has no test-split option. It joins all 250 validation IDs for each supplied label, checks split hashes, compiles contracts without predictions, and scores the seven first-wave validation evaluations. It does not call a model, regenerate answers, overwrite original outputs, or invoke the old automatic selector. It records input/source hashes and produces:

- `contracts.json`: inspectable, prediction-independent obligations and input facts.
- `<label>.scored.jsonl`: legacy results, v2 checks, behavior counters, and unresolved scope.
- `summary.json` and `REPORT.md`: pass/fail/review counts and coverage. No new four-task macro winner is declared.
- `changed_scores.jsonl`: changes from legacy decisions for targeted diagnostics.
- `blinded_review.jsonl`: all predictions, including automatic passes, shuffled without model labels or gold text. This permits false-positive auditing, not merely appeal of failures.
- `review_key.json`: identity mapping, kept separate from an adjudicator.
- `input_manifest.json`: reproducibility hashes.

Reported intervals are coverage intervals conditional on the automatic labels: pass/N through (pass+review)/N. They are not confidence intervals or proven bounds on clinical accuracy. Comparing resolved-only percentages across models is prohibited because models can have different parsing coverage.

## Adjudication rubric and release gate

For every unresolved required obligation, and a balanced sample of automatic passes/failures, a reviewer should:

1. Read the question and input before the candidate answer. Confirm the contract itself is complete and grounded.
2. Identify the minimally required claims. Separate requested facts from optional explanation.
3. Classify each claim as supported, contradicted, not supplied, or unresolved, citing the input span or reproducible arithmetic.
4. Check all requested subquestions, negation, subject binding, units, and additional assertions. Do not require shared vocabulary or a fixed sentence layout.
5. If a question/reference is defective, flag the record; do not award or remove credit by inventing missing evidence.
6. Record the final decision and rationale without model identity. Disagreements need independent review; the reference answer can be inspected afterward as another potentially fallible annotation.

An optional future LLM judge must implement this same structured rubric, be pinned to a model/version/prompt, and be calibrated against human decisions on both correct and incorrect outputs. It cannot be treated as truth simply because it uses fewer regexes. No external judge or clinical data upload is used in this revision.

Freeze the adjudicated rubric and report its coverage before selecting a new checkpoint. The legacy `epochs`, `select`, `compare`, and `freeze` commands still implement v1 selection; they must not be used to announce a v2 winner. Generation with new labels can continue, followed by the validation-only v2 command. Existing historical reports remain reproducible and explicitly historical.

## Tests and limitations

Synthetic tests cover paraphrase/sentence-boundary invariance, not-within polarity, analyte swaps, incompatible units, contradictory answers, unrecognized language, optional gold numbers, question-specific numeric operations, wrong extra facts, formula precedence, percentage notation, reference-versus-measurement distinctions, explicit self-corrections, absent BMI inputs, and input conflicts. The tests never assert that copying arbitrary gold must be correct in v2.

This release prioritizes transparent coverage over a misleading fully automatic score. Completing the adjudication queue is outstanding evaluation work, not a hidden zero score. No training, prompt win, throughput gain, or higher-learning-rate advantage is claimed until the corresponding validation experiment runs.
