# Scorer v2.1: known issues

This is the single list of known v2.1 defects. It supersedes `docs/history/SCORER_V2_1_AUDIT.md` (2026-09-30) and the defect notes scattered across `reports/scorer_v2/VALIDATION.md`, `docs/PROJECT_WALKTHROUGH.md` (stage 17) and D-092, D-096 and D-101.

**Scope.**

- Scorer: `src/clinqa/scorer_v2.py`, frozen as sha256 `b15db3d0…` (`reports/scorer_v2/scorer_v2.1.sha256`). Entry points: `scripts/score_v2.py` and `scripts/score_v21_val.py`.
- Design: `docs/SCORER.md`.

**Status.** v2.1 is the reported scorer (D-096). It is a **diagnostic, not a validated accuracy measure**. Numeric conclusions rest on item review, not on v2.1 alone. Nothing below is fixed in v2.1, which stays frozen. `scripts/scorer_v22.py` is an unadopted prototype that addresses S-01 and S-02 for numeric answers.

**How to read an entry.**

- **Type**: FP (false pass: a wrong answer scored correct) or FF (false fail: a correct answer scored wrong).
- **Evidence**: item IDs (val_*, test_*) with the model label where it matters.
- **Handling**: what the report does about it now.

## Summary

| ID | Issue | Type | Answer types | Measured examples | Severity |
|---|---|---|---|---|---|
| S-01 | Extra claims are never checked | FP | numeric, extractive | F ep2: val_117, val_194; F′ test: test_191, test_223, test_290, test_293, test_330, test_273 | High |
| S-02 | Any in-band number passes; contradicting numbers are ignored | FP | numeric | F ep2: val_062 | High |
| S-03 | Partial keys count as full correctness | FP | numeric, extractive | 15 numeric and 6 extractive partial val keys; val_004, val_005, val_001, val_085 | High |
| S-04 | Bound-relative or incidental wording read as an analyte state | FF | tool (conversion), numeric | A-sft2: val_008, val_071, test_135, test_214; F′ test_026; C test_012, test_261; Q35 test_373; A-sft2-Q35 test_173, test_214, test_373 | Medium |
| S-05 | Set check rejects analytes the question itself names | FF | numeric | val_001 (base, Qwen3.5) | Medium |
| S-06 | Last-assertion rule picks the wrong statement | FF and FP | extractive, numeric | 3 on holdout 2; Qwen3.5 val_062; test_391 (F′ passes a contradiction, Q35 fails a correct answer) | Medium |
| S-07 | Plural and blanket sentences bind states to the wrong analyte | FF | numeric | 4 train golds and val_009 under the v2.2 guards | Medium |
| S-08 | Hedged or final-line answers mis-scored | FP | uncertain, numeric | 2 on holdout 2 | Low |
| S-09 | Uncertain "missing field" taxonomy too narrow | FF | uncertain | 2 on holdout 2 | Low |
| S-10 | Text fallback: grounded numbers alone pass, short correct answers fail | FP and FF | extractive | 20 text-fallback val keys; test_336 (A-sft2, Q35 name the right drug and fail on recall) | Medium |
| S-11 | Canonical reference ranges injected for note-only labs | Policy | numeric, extractive | val_085 (BUN bound 20 not in input) | Medium |
| S-12 | Uncertain category inferred from the gold answer | Policy | uncertain | All 38 val uncertain keys | Low |
| S-13 | Vitals use conventional ranges and accept the gold's state | Policy | numeric, extractive | Heart-rate items such as val_062 | Low |
| S-14 | Validation evidence is weaker than the agreement figure suggests | Process | all | See entry | High |
| S-15 | Error profile shifts as models improve | Process | numeric | F′ vs F; wave 1 vs wave 4 | High |
| S-16 | Tooling: silent ID intersection, overwrite, null-as-false | Tooling | all | See entry | Low |
| S-17 | Q5 abstention accepted without checking fabricated values | FP | tool (Q5) | C-filtered test_288 | High |
| S-18 | A quoted standard reference range is flagged as a fabricated value | FF | uncertain | A-sft2-Q35 test_163, test_307 | Medium |

**Measured scale.**

- Holdout 2 (clean, 110 items): 92.6% agreement (κ 0.72), with 2 FP and 6 FF.
- Numeric only, all three adjudicated sets (125 items where both rater groups agree): 3 FP and 3 FF against the reference. That reference shares S-01: the rubric did not require unrequested claims to be true.
- On wave-4 outputs, the v2.2 guards found 6 verified FPs across nine validation arms (all Qwen3 arms; none in Qwen3.5 arms), and 4 more in the adjudicated sets.

---

## A. What is checked (coverage)

### S-01 Extra claims are never checked (FP, high)

- **Mechanism.** `build_key` (`scorer_v2.py:590`) derives checks only from the question's clauses. `run_check` (`:824`) evaluates only those checks, so a false statement the question did not ask about cannot fail the answer. The only exception is `set:false_member` (`:871`), and only inside set checks. `docs/SCORER.md` states that extra false claims "are still errors when the checker can establish them"; the implementation does not do this outside set checks.
- **Evidence.**
  - F ep2 val_117 writes "ferritin … below by 4.2" (true 5.2).
  - F ep2 val_194 calls a normal creatinine (1.2, range 0.7–1.3) "elevated".
  - base_test test_367 calls a normal ferritin low.
  - raw_lr1e4_test test_293 says haemoglobin 15.7 is "1.3 below its lower limit".

  The blinded raters also passed these items. On the final test, F′ answers that v2.1 passes include:
  - test_330: a correct 48.2 deficit, then "a 48.2% reduction" (about 80% against 60);
  - test_191: platelets of 119.5 called normal against 150–400;
  - test_223: total cholesterol of 238.7 called normal against <200;
  - test_290: HDL of 53.5 called "mildly low by 13.5" against >40;
  - test_293: haemoglobin of 15.7 said to be 0.2 above a lower limit of 12.0;
  - test_273: glucose given the larger percentage increase, then potassium called proportionally more elevated.
- **Impact.** Numeric is over-credited for models whose remaining errors are in commentary (S-15). F ep2 v2.1 numeric is 46/50 and would be 43/50 with the guards.
- **Handling.** Reported as a limitation. Numeric conclusions use item review. H2 is reported as unresolved (D-101).
- **Fix later.** Claim-level checking. The `scripts/scorer_v22.py` state and amount guards are a starting point; they need a clean holdout of current-model outputs.

### S-02 Any in-band number passes; contradicting numbers are ignored (FP, high)

- **Mechanism.** Numeric checks pass if `any(_in_band(x, lo, hi) for x in candidates)` (`run_check`, `:824` onward). Other numbers bound to the same subject are not compared against the key.
- **Evidence.** F ep2 val_062: "exceeds the upper limit by 9 bpm (100 − 99 = 1)" passes, because 1 is in the band.
- **Impact.** Self-contradictory answers score correct.
- **Handling.** Item review for numeric claims.
- **Fix later.** Fail when a number bound to the subject and operation lies outside the band, unless the answer explicitly corrects itself. This is the v2.2 amount guard.

### S-03 Partial keys count as full correctness (FP, high)

- **Mechanism.** A clause that `build_key` does not recognise sets `coverage = "partial"` (`:781`). The item is still scored correct or incorrect on the recognised checks alone.
- **Evidence.**
  - On val, 15 of 50 numeric keys and 6 extractive keys are partial.
  - val_004 checks heart rate but not the beta-blocker it asks for.
  - val_005 checks the 10.6 difference but not the CKD stage.
  - val_001 checks the second abnormal lab but not the 5.9-point excess.
  - val_085 checks the excess but not the requested BUN/creatinine ratio, yet is labelled structured.
  - Mutations that omit the missing component still pass (`docs/history/SCORER_V2_1_AUDIT.md`).
- **Impact.** Incomplete answers can pass.
- **Handling.** The coverage label is stored per item (`coverage` in each `*.scored.jsonl`). Reports describe v2.1 as "asked-part" correctness.
- **Fix later.** Report obligation coverage, and return `unresolved` instead of pass when an obligation has no check.

### S-10 Text fallback passes on grounded numbers alone (FP, medium)

- **Mechanism.** `check_text` (`:797`) returns `text:numbers`, a pass, when the gold's grounded numbers appear in the prediction (`:808`). Wording and direction are not checked in that branch.
- **Evidence.** 20 val keys use the text fallback (19 extractive, 1 numeric).
- **Handling.** Reported as part of S-03 coverage.
- **Fix later.** Require the gold's key terms as well as its numbers, or mark the item unresolved.

## B. How answers are read (reader and binding)

### S-04 Bound-relative wording read as an analyte state (FF, medium)

- **Mechanism.** The state reader (`read`, `:298`; `_STATE_PATTERNS`, `:200`) maps "reduced", "low", "below" and "<" onto the nearest analyte, even when the word describes another quantity or a threshold. `score_tool` then fails a correct conversion as `status_contradicts_input` (`:963`).
- **Evidence.**
  - A-sft2 val_008 and val_071: "markedly reduced renal function" is read as creatinine low.
  - A-sft2 test_135: "above the normal fasting threshold … below the diabetes threshold" is read as glucose low.
  - A-sft2 test_214: "within the desirable range (<5.18 mmol/L)" is read as total cholesterol low.
- **Impact.** A-sft2's grounded tool score is understated by two items on val (51 vs 53/55) and two on test (87 vs 89/90). Whether D-100 passes does not change, because the discordance criterion still fails.
- **Handling.** Disclosed with the A-sft2 result (D-101) and the test report.
- **Fix later.** Neutralise bound-relative phrases before state extraction, as the v2.2 state guard does. Only bind a state word that follows the analyte within the same clause.

### S-05 Set check rejects analytes the question itself names (FF, medium)

- **Mechanism.** In a "which other value is also abnormal" set check, `set:false_member` (`:871`) treats every analyte the answer calls abnormal as a claimed member, including the analyte the first clause asked about.
- **Evidence.** val_001: base and Qwen3.5 answers correctly state the HbA1c excess and then list the other abnormal labs, and are failed as `set:false_member(HbA1c)`.
- **Handling.** Item review.
- **Fix later.** Exclude analytes named in earlier clauses of the question from the claimed set.

### S-06 Last-assertion rule picks a non-final statement (FF, medium)

- **Mechanism.** Status checks use the analyte's last asserted state (`:855`, "last assertion wins"). A later comparative or hypothetical sentence can override the answer's actual claim.
- **Evidence.**
  - Three false fails on holdout 2 (VALIDATION.md §3).
  - Qwen3.5 val_062: "is still within normal limits. It does not exceed the upper limit by any beats per minute" is failed.
- **Handling.** Item review.
- **Fix later.** Prefer the sentence that answers the question; treat conflicting states as `review`.

### S-07 Plural and blanket sentences bind states to the wrong analyte (FF, medium)

- **Mechanism.** `_PLURAL` (`:225`) and `_BLANKET` (`:282`) assign "are within normal range" or "all other values fall within" to earlier or later mentions in the sentence.
- **Evidence.**
  - Gold answers train_038, train_1081, train_1546 and train_820: under the v2.2 guards these produce "said normal" conflicts.
  - val_009 under the same check.
  - test_136 (raw_lr1e4_test): "the Wells score is expected to be low, and the D-dimer …" binds "low" to D-dimer.
- **Impact.** Mostly latent in v2.1, because states are only checked where a key asks for them. It blocks any claim-level extension.
- **Fix later.** Bind plural predicates only to the analytes listed in the same coordinated phrase.

### S-08 Hedged or final-line answers mis-scored (FP, low)

- **Evidence.** Two false passes on holdout 2:
  - "cannot be safely initiated" was counted as an appropriate hedge;
  - a final "Answer:" line contradicted the body on a "most" question.
- **Fix later.** Score the final explicit answer line when one exists, and flag body/answer conflicts.

### S-09 Uncertain "missing field" taxonomy too narrow (FF, low)

- **Evidence.** Two false fails on holdout 2: the answer named the missing item as "prior value" where the key expected "timestamp".
- **Fix later.** Widen the synonym sets per missing-field category, built from train answers only.

### S-17 Q5 abstention accepted without checking fabricated values (FP, high)

- **Mechanism.** On Q5 records (`expected = "abstain"`), `score_tool` accepts an answer that makes no call and contains an abstention phrase. It does not run the fabricated-value check that `score_abstain` applies to uncertain records.
- **Evidence.** C-filtered test_288: "The patient's weight is documented at 50.0 kg and height at 150.0 cm; however, the BMI calculation cannot be completed …". Neither value is in the input. v2.1 scores it correct, so C shows 5/10 natural-Q5 failures under v2.1 against 6/10 fabrications when every case is read.
- **Handling.** Natural-Q5 fabrication is reported from the frozen P1 classifier plus reading every case, never from v2.1 alone.
- **Fix later.** Apply the fabricated-value detector to Q5 abstentions.

### S-18 A quoted standard reference range is flagged as a fabricated value (FF, medium)

- **Mechanism.** Uncertain records are scored with `v1.score_uncertain` (via `score_abstain`), whose fabricated-value detector rejects numbers that are absent from the input. A standard range quoted for context, such as the BMI normal range 18.5–24.9, is such a number.
- **Evidence.** A-sft2-Q35 test_163 and test_307 correctly state the documented weight (63.7 kg; 133.6 lb), say that height is missing and that BMI cannot be calculated, and add "(18.5–24.9 kg/m²)". Both fail as `fabricated_value`. The Qwen3.5 relabel model, which omits the range, passes.
- **Handling.** Reported as false fails in the test report; scores are not repaired.
- **Fix later.** Exempt recognised reference ranges and thresholds from the fabricated-value check.

## C. Scoring policy

### S-11 Canonical reference ranges injected for note-only labs (policy, medium)

- **Mechanism.** `_CANON_RANGES` (`:477`) supplies standard ranges for labs that appear only in the note (`note_facts`, `:491`). Keys can therefore require a difference against a bound the input never states.
- **Evidence.** val_085: no numeric BUN bound is in the input, but the key requires the 32.1 excess against 20.
- **Impact.** This is consistent with the task design: Q11 shows that 46 extractive golds cite standard ranges. It contradicts a "use only supplied ranges" prompt, and prompt v2 was not adopted for this reason.
- **Handling.** Documented. The policy is kept because it matches the dataset contract.

### S-12 Uncertain category inferred from the gold answer (policy, low)

- **Mechanism.** Abstention keys call `v1.classify_uncertain(question, gold_answer)` (`build_key`, `:603`). The expected missing-field category depends on the gold wording, not only on the input.
- **Impact.** The uncertain score is not fully input-derived, unlike the other types.
- **Handling.** Documented limitation.

### S-13 Vitals use conventional ranges and accept the gold's state (policy, low)

- **Mechanism.** `_VITAL_RANGES` (`:170`) applies adult resting thresholds when the table has no range, and status keys for vitals also accept the state the gold asserts.
- **Impact.** Vitals states are partly gold-driven and convention-dependent.
- **Handling.** Documented. The v2.2 guards skip vitals for the same reason.

## D. Validation evidence and process

### S-14 Validation evidence is weaker than the agreement figure suggests (process, high)

- **Test exposure.** v2.1 fixes were derived from holdout 1, which consists of test items. Holdout 2 is the only clean estimate, and it is also drawn from test. Test is therefore reused (disclosed in every test report).
- **Base rates.** The development reference has 113 correct and 7 incorrect items. An always-correct scorer would already agree 94.2%, so zero false passes on seven negatives is weak assurance.
- **Shared rubric blind spot.** Raters were not required to check unrequested claims (S-01). Agreement with them does not measure S-01 at all.
- **Coverage.** Raters are LLM agents of one family; model coverage across adjudication sets is uneven; disputed holdout-2 items are excluded from the agreement figure.

### S-15 Error profile shifts as models improve (process, high)

- **Observation.** The adjudicated sets came from wave-1 models, whose numeric errors sat in the asked part, where v2.1 is strong. Wave-3 and wave-4 SFT models rarely miss the asked part, so their remaining numeric errors sit in extra claims (S-01, S-02), which v2.1 cannot see.
- **Evidence.**
  - Numeric discordance between F ep2 and Q35-relabel ep2: v1 13 vs 4, v2.1 3 vs 3, informal item reading about 7 vs 1.
  - The v2.2 guards flip no Qwen3.5 answers but flip three F ep2 answers.
- **Impact.** Scorer accuracy measured on one model generation does not transfer to the next. Model comparisons on numeric can change sign with the scorer.
- **Handling.** Numeric differences of a few items are not interpreted. A single same-recipe refit (F vs F′) also moved the macro by 1.7 pp.

### S-16 Tooling (low)

- `score_v2.py` scores the intersection of record IDs and available trajectories, can overwrite an output directory, and treats absent or `null` booleans as false (`_rate`).
- The guarded wrapper `scripts/score_v21_val.py` enforces exact validation IDs and refuses to overwrite, but the test path (`score_v2.py --split test`) does not.
- `scripts/legacy/scorer_v2_agreement.py` reads current score directories without checking scorer hashes, so historical v2.0 agreement cannot be recomputed from the current directories.

## What v2.1 is still good for

- Tool scoring on executed outcomes: name, grounding, result within tolerance, result reported. S-04 is the main exception.
- Abstention on Q5 records, and over-call and over-refusal counts.
- Extractive questions with structured keys, and asked-part numeric correctness.
- Coarse comparisons with large effects. For example, base vs SFT on tool use (30 → 54/55 on val; 48 → 89/90 on test), and the Q5 fabrication contrasts, which were confirmed by reading every case.

Small numeric or macro differences (a few items, or about 1–2 pp) should not be read as model differences under v2.1. Further test-split examples (partial keys test_006 and test_043; ambiguous "closest to" test_347) are listed in `reports/w4/test/TEST_REPORT.md`.
