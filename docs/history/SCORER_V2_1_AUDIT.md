> **Archived 2026-10-03.** Superseded by `docs/SCORER_V2_1_KNOWN_ISSUES.md`. Kept unchanged below as a historical record; relative links and some script paths (now `scripts/legacy/`) may be stale.

# Scorer v2.1 audit and next experiments

Date: 2026-09-30. Verdict: suitable for exploratory diagnostics, not approved for automatic checkpoint selection or as a validated full-answer accuracy measure. The historical v1 and bounded-contract scorer are retained as distinct evidence, not treated as disposable duplicates.

## Installation follow-up (2026-09-30)

The previously missing core has now been recovered from Downloads and installed at `src/clinqa/scorer_v2.py`; its SHA256 matches the report. The supplied tests were installed and all 16 pass. All 250 saved validation keys and 1,750 scores reproduce exactly. The source-missing findings below describe the original audit state and are resolved.

The correctness blockers remain: executed mutations of val_004 (omit the medication or substitute atenolol), val_005 (omit stage) and val_003 (reverse 2 out of 3) all pass. Source inspection confirms incomplete-contract acceptance, a numeric-membership text shortcut and injected canonical reference ranges. Reproducibility does not validate those policies. See W2-INSTALL-005 in the journal for the crossed experiment design and Q5 relabel specification.

## Scope and reproducibility

Reviewed the three new scripts, saved validation keys and scores, all seven validation score inventories, development adjudication labels and selected validation trajectories. Read the supplied VALIDATION.md, which includes test-derived results and development history; those results are not used here to choose training settings. No raw test predictions, test adjudication packets or test records were opened, and the agreement script was not run because it automatically accesses test.

Each saved validation evaluation has exactly the same 250 IDs as canonical validation. Recomputed per-type pass counts agree with summary.json across all seven runs. This verifies aggregation, not scorer correctness.

The essential implementation is missing from the available project: `score_v2.py` and `scorer_v2_gold_check.py` import `clinqa.scorer_v2`, while the repository only contains the different module `clinqa.scoring_v2`. Running the supplied score entry point with `--help` raises ImportError. The expected core hash is b15db3d0b12a4a798e4114e9b1b9573733debbb3362ae9492a4cd00f804a48f3. Searching the Clinical/Heidi and Downloads locations found no matching source filename. Downloads/scorer_v2.zip contains no Python source. Downloads/tests/test_scorer_v2.py exists and was inspected, but cannot execute without the missing module. No implementation has been reconstructed or silently substituted.

Consequently, this is a report, wrapper-code and artifact audit. Core source review, fresh scorer reproduction and adversarial execution remain incomplete. The previously reported 83 passing tests concern the older bounded scorer, not this v2.1 implementation.

## Blocking findings

### P0: the evaluation boundary has been crossed

VALIDATION.md explicitly states that v2.1 fixes were derived from holdout1_test. This conflicts with the established validation-only development policy. Freezing a hash after these changes does not remove their test origin. Holdout2 may be disjoint from holdout1 at question level, but it is still part of the original test split; the test set cannot continue to be described as an untouched final evaluation. Earlier project history also records whole-test model evaluation, so adjudication-packet disjointness alone is insufficient to establish complete non-exposure.

Preserve this history transparently. Stop further test-driven changes. Treat v2.1 as an exploratory measurement branch with disclosed contamination; do not use its test-derived tolerances or fixes as justification for the next training decisions. Recover a pre-exposure scorer snapshot if possible, and derive a new rubric from task requirements plus permitted train/validation evidence. With data fixed, no procedure can manufacture a new independent final test; describe subsequent results as validation development until a genuinely untouched evaluation source is available. This review does not modify the dataset.

### P0: partial contracts are being presented as full correctness

Validation keys contain 21 partial cases (15 numeric, 6 extractive) and 20 text-fallback cases (19 extractive, 1 numeric). The script includes every boolean in the same task accuracy and macro. Concrete contract omissions:

- val_004 asks for heart rate AND current beta-blocker. The key checks only heart rate 118; the beta-blocker obligation is absent.
- val_005 asks for a difference AND a stage. The key checks only the 10.6 difference.
- val_001 asks for the HbA1c excess AND another elevated lab. The key checks only an entity set; the requested 5.9-point calculation is missing.
- val_085 asks for a BUN excess AND a BUN/creatinine ratio. The key checks a difference and the raw BUN value, not the requested ratio, yet labels coverage structured.

These omissions are demonstrable from saved keys. Without the core module, a mutated-answer false-pass rate cannot be measured; a claim that those mutations were executed would be unjustified. The appropriate next checks are deletion/corruption of each omitted answer component. A passing subset must not become an unqualified full-answer pass. Track obligation coverage and use unresolved status when semantic completeness cannot be determined. Merely checking for the word structured is insufficient.

### P0: adjudicator agreement conceals an incorrect acceptance policy

Development item_000 maps to filtered epoch one, val_075. The response correctly states 7.6 minus 1.2 equals 6.4, but describes the value as approximately 5.3 times the reference limit. Both raters acknowledge the multiplier error and explicitly excuse it because the factor was not requested. The saved scorer passes it too.

This is a rubric defect, not independent confirmation of correctness. An answer need not include an unrequested calculation, but a false calculation it does include must still affect factuality. Where wording is ambiguous, annotate that ambiguity rather than exempting all extras. Report requested-answer completeness separately from contradictions/unsupported additions; neither metric should silently erase the other. This correction follows the task's grounding objective and validation evidence, not a test-derived example.

### P1: key provenance can contradict the proposed prompt

In val_085, the input supplies BUN 52.1 and creatinine 1.0 but no numeric BUN upper reference bound. The key nevertheless requires a 32.1 difference, implying the unsupplied bound 20. The reference answer contains 20. The exact implementation source of that bound cannot be established without the core module, but its absence from the input is verified.

Under the reference-only prompt candidate, the model should report the available ratio and explain that the requested range excess lacks a supplied bound. Scoring it against mandatory 32.1 would punish intended compliance. Record a source span for every bound/target, distinguish input-derived from gold-derived or external facts, and settle this policy before the prompt ablation. Do not optimize gold self-check pass rate as if the gold were infallible.

### P1: base bias is reduced, not eliminated

Base val_001 correctly gives the requested 5.9 excess and potassium as another high lab. The scorer fails it with set:false_member(HbA1c), although HbA1c necessarily belongs in the first clause of this question. This is a demonstrated answer-scope error. In contrast, some SFT answers also assert that potassium is the ONLY other abnormal lab, which conflicts with sodium in the supplied evidence. Those answers require a separate contradiction check, not the same unscoped entity rejection. Repairing base underestimation requires claim and question-clause binding, not merely broader word matching.

## Calibration and reporting limitations

The development packet contains 120 unique question IDs: 45 numeric, 30 extractive, 25 tool and 20 uncertain. Its key shows 49 v1/v2 disagreements, not exactly half. The final A/C reference has 113 correct and only 7 incorrect responses. An always-correct predictor would already obtain 94.2% agreement. V2.1 matches all 120, but this is explicitly a tuned development result, not held-out generalization. Even zero false passes among seven negative references provides weak assurance about future false-pass rates.

Model coverage is uneven: base 31, filtered epoch one 19, filtered epoch two 23, raw5e5 epoch one 18, raw5e5 epoch two 18, raw1e4 epoch two 11. The historical selected raw1e4 epoch-one model has zero development adjudication examples. Include it in future calibration; sample comparable question IDs across models so model-specific scorer bias can be estimated. Same-family LLM agents may share errors. Model blinding does not establish independence, and showing the reference answer can anchor all raters despite a warning that it may be wrong.

The agreement script has additional reproducibility limitations:

- It reads current score directories without enforcing scorer hashes or frozen-version associations. It cannot reproduce historical v2.0 agreement after those directories contain v2.1 outputs.
- It collapses duplicate item IDs through dictionaries and silently interprets any non-correct label as false. Validate IDs, allowed labels and scorer versions explicitly.
- It excludes disputed holdout2 labels. That yields agreement conditional on rater agreement, not agreement on the full sample. Preserve disputed cases and report coverage/bounds or independent resolution.
- It automatically reads test and writes a temporary key. A development command should require an explicit validation-only path and avoid overwriting evidence.

`score_v2.py` also silently intersects record IDs with available trajectories, overwrites an existing report directory and records only a short scorer version. Current artifacts happen to be complete, but future missing records could alter denominators and rankings. Add exact unique-ID checks, input/source/trajectory/prompt hashes, output non-overwrite protection, complete boolean/status validation, and explicit missing-category handling. `_rate` coerces absent or null values to false; that is unsuitable for unresolved correctness.

The Downloads unit tests include useful negation, cross-sentence, arithmetic and missing-input fixtures. They also explicitly accept imperial rounding, whose development provenance must be established given the earlier test-only observation. A train/val gold-pass threshold of 97% is a consistency diagnostic, not scorer validity. Add positive/negative paired mutations for omitted clauses, wrong extra facts, subject swaps, missing reference ranges, signed values, boundary inclusivity, tool-result mismatch and out-of-budget trajectories. Do not retune toward known test failures.

## What the validation results support

Saved v2.1 counts are internally consistent. Base has 93/100 extractive and 41/50 numeric passes, compared with 99/100 and 44/50 for raw1e4 epoch one. Filtered5e5 epoch two has 99/100 and 41/50. These remain provisional measures because of the defects above; the numerical uplift from v1 is not itself proof of validity.

The tool decomposition remains more informative than the macro: saved expected-call success is base 30/55, raw1e4 epoch one 55/55, filtered epoch one 54/55 and filtered epoch two 53/55. Missing-input Q5 success is respectively 7/7, 1/7, 7/7 and 6/7. The denominators and strict versus outcome-based checks must remain visible. Seven Q5 cases support a useful warning signal, not a precise population safety estimate.

Filtered epoch two minus raw1e4 epoch one has a macro difference of +0.003676, or +0.37 percentage points. A paired, answer-type-stratified bootstrap over the 250 validation question IDs (10,000 replicates, seed 42) gives a percentile 95% interval of approximately -3.15 to +3.86 percentage points. There are eight question-level wins and seven losses. This conditional interval uses the saved labels; it excludes scorer error, training-seed variance, repeated model-selection effects and possible encounter clustering. It does not establish equivalence or a new winner.

## Next experiments and release gates

### E0: measurement release, before model ranking

Recover the missing core source and matching dependency snapshot. Reproduce validation outputs in a fresh directory and compare every key/score. Preserve original reports unchanged. Resolve the four blocking issues above and version the corrected contract, source rules and partial-answer policy.

Freeze a scorer development/acceptance protocol using only permitted evidence. The existing validation data has already been inspected, so any newly partitioned portion must not be advertised as historically untouched. Use question/encounter grouping, comparable model outputs and deliberately incorrect semantic mutations to test mechanisms; synthetic mutations are regression tests, not estimates of real-world accuracy. Prefer independent adjudication of difficult cases. If only LLM adjudicators are available, describe the evidence as model agreement, not clinical ground truth. Audit automatic passes and fails, not just unresolved cases. A semantic parser/judge is optional; it must have its own error handling and evidence checks.

Release criteria: all IDs and source hashes verified; no partial contract reported as complete; no known contradictory extra assertion excused solely for being optional; no unsupplied range introduced into an input-only target; frozen rubric and explicit coverage; unresolved cases cannot silently determine a checkpoint winner. Existing instructions that a deterministic fail can never be challenged require qualification: a verified numeric calculation is authoritative, but an erroneous regex extraction is not.

### E1: prompt-only, fixed weights

Use the existing six-arm design: v1 versus reference v2 on base, raw1e4 step125 and filtered5e5 step121. Hold generation batch at 2 and all budgets fixed. Regenerate both arms under the same code state. Read range provenance, numeric obligations, unsupported assertions, over-refusal, tool completion and tokens separately. Do not start this comparison with val_085's incompatible key. Generation can be saved before evaluator completion, but do not select a prompt from unvalidated scores.

### E2: filtered-data/LR control and efficiency

Run filtered 1e-4 against the existing filtered 5e-5 baseline with v1 training prompt, two epochs and both checkpoints retained. First retain micro-batch 1/accumulation 16 for a clean control, or explicitly reproduce both LR arms under any changed batching. Evaluate micro-batch 4/accumulation 4 separately, keeping effective batch 16. Preflight longest sequences, padding/masks, peak memory and throughput. Test generation batch 4 then 8 on fixed weights, with batch 2 as reference. Do not change decoding or budgets at the same time.

### E3: bounded higher-LR experiment

With scorer, data, prompt and batching frozen, compare filtered 1e-4 with 1.5e-4; make 2e-4 conditional on stable loss and validation behavior. Save both epochs, inspect CE plus tool/missing-input guardrails, and report paired differences. Repeat the most promising comparison with additional seeds only if the apparent gain merits the cost. Avoid an expansive search over a small, repeatedly inspected validation set.

### E4: annotation and explanation experiments

Audit the 78 train-only Q5 proposals before testing a relabeled view. Report changed sample count, class mix and optimizer exposure; use matched-exposure controls if attributing gains specifically to labels. Keep canonical validation/test labels unchanged.

Test a short pre-call sentence separately. The parser permits call-associated text, but existing SFT targets are empty there. Check all pre-call claims, execution success and token cost. Numeric explanation or training-prompt v2 are additional separate factors. New base models and RL remain deferred: RL would optimize an unreliable reward if introduced now.

## Structure and cleanup

README is the navigation entry point; EXPERIMENT_JOURNAL is the chronological decision record; this audit is the current release assessment. `metrics.py` remains scorer v1; `scoring_v2.py` is the earlier bounded-contract design; `score_v2.py` is only a wrapper for the missing newer `scorer_v2.py`. `make score-v2` and `make prompt-ablation` currently invoke the bounded scorer, not the supplied v2.1 wrapper. The shared v2 name is ambiguous and must not be used as a sufficient version identifier.

Cleanup is intentionally limited to generated Python/test/lint caches and Finder metadata outside .git/.venv. Preserve all source, tests, original metrics, trajectories, data, checkpoints, adjudication packets, scorer snapshots and design history. The removal manifest records paths and sizes. No Downloads files are deleted. Redundant-looking generated reports are retained because they demonstrate changed measurement and cannot yet be reliably regenerated from the missing source. No scoring policy or training config is changed during this audit.
