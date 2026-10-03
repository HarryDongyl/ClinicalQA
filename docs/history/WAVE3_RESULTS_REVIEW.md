> **Archived 2026-10-03.** Superseded by `docs/PROJECT_WALKTHROUGH.md and docs/EXPERIMENT_JOURNAL.md`. Kept unchanged below as a historical record; relative links and some script paths (now `scripts/legacy/`) may be stale.

# Wave 3 completed-results review

Date: 2026-10-02. Scope: downloaded validation, P1 and train-fit outputs only. No test results were used. This is an AI output and measurement review, not human clinical adjudication. Qwen3.5 is running separately; no local Qwen3.5 outcome is available in this review.

## Executive finding

The strongest result is a data-policy improvement: explicitly teaching abstention on the 78 input-deficient Q5 training records removes the observed numeric fabrication on missing-measurement probes while retaining grounded tool performance. The numerical-reasoning headline is less reliable: diagnostic v2.1 reports 46/50, but accepts demonstrably contradictory or arithmetically incorrect answers. Do not describe this as 92% verified numerical accuracy or 96.69% verified clinical macro accuracy.

## Runs and comparability

Seven validation runs contain 250 unique records each. Six arms also have 34 P1 probes; the relabel epoch-one checkpoint has no P1 run. Control and relabel epoch two each have a 200-record train-fit diagnostic.

- R0-v1, R0-v3 and R0-v3-FS4: Qwen3-4B-Instruct-2507, inference-only prompt comparisons. FS4 contains four fixed train demonstrations, including one metric BMI demonstration and no unit-conversion tool demonstration.
- C-filtered-s42 (`w3_c_filtered_s42`): 1,922 filtered rows, LR 1e-4, two epochs, mb1/ga16, seed 42, NF4, LoRA r16, checkpoint 242.
- F-s42 (`w3_relabel_lr1e4_s42_step000125` and `step000250`): the same recipe on 2,000 rows, adding 78 reviewed Q5 abstention targets. Epoch two is the prospectively designated primary endpoint; epoch one is diagnostic. Both use prompt v1.
- R0-8B (`w3_r0_8b`): Qwen3-8B, thinking disabled, zero-shot. No 8B SFT or GPU smoke outcome is available.

All completed arms used RTX 4090, generation batch 2, one tool call, at most two assistant turns, 256 new tokens per turn and 512 total. The six 4B arms have identical recorded source-code hash maps. C and F epoch two have identical recorded inference protocol objects. Different commit IDs and dirty working-tree flags alone do not imply different inference implementations.

Adapter SHA256 identifiers:

- C: `8758304b0666b0609cb1c2096919f6cb265b874a6470438cdc15a2e28505596f`
- F epoch one: `2366eac4b1af1f7941430b6d3662ac1788e34f8a7f615dcd0d071e128c18183c`
- F epoch two: `f556435fb557917295627cd2d1a4650fc95adb661efc2c4bd5837474c4c5f0ce`

C versus F tests a training-policy package, not a pure label intervention: row count, class mix and optimizer steps change (242 to 250). Only seed 42 is available.

## 1. Missing-input behavior: a strong, specific improvement

Every C and F epoch-two P1 response and every natural-Q5 response was inspected in addition to rerunning the versioned `p1-2` classifier.

- P1 numeric fabrication falls from 25/34 for C to 0/34 for F. Both make zero actual tool calls on these probes. The control failure is therefore fabrication in answer text, which a tool-routing-only metric misses.
- On the corresponding unmodified inputs, valid-call success stays 33/34 for both; `val_105` is the same remaining failure. There is no newly lost partner success.
- Across all 55 grounded tool questions, both obtain 54/55 under the existing strict task checks: BMI 39/40 and unit conversion 15/15.
- On seven natural validation Q5 cases, numeric fabrication falls from 4/7 (`val_041`, `val_096`, `val_134`, `val_245`) to 0/7; neither arm makes an actual call.

Interpretation: filtering bad targets helped but did not adequately teach the missing-both-inputs response. Explicit negative examples teach when an answer is unsupported, beyond whether a tool should be invoked.

Limits: 0/34 has a Wilson 95% upper bound of 10.15%, so it does not establish a population error rate below 5%. These are paired perturbations of validation examples, not an independent external cohort. F still makes a qualitative unsupported statement in `val_135` that height and weight are documented while their values are missing. The numerical grounding screen is not a universal hallucination or clinical-safety guarantee.

The frozen engineering gate recomputes to suggested pass, with no new partner failures. The report remains unapproved; this review does not write `gate_fs42.json` or launch seeds. Replication is justified, subject to recording the reviewed gate through the existing workflow.

## 2. Numeric: genuine gains coexist with evaluator errors

Diagnostic v2.1 changes from C 40/50 to F 46/50. All six apparent gains were inspected:

- `val_028`: credible gain. TIBC 430.1 exceeds 370 by 60.1; C incorrectly describes everything as normal, F identifies the elevation.
- `val_114`: credible gain. F identifies hematocrit 25.5 versus 36 and approximately 29.2% below; C selects the wrong analyte.
- `val_186`: credible gain. F consistently identifies bilirubin as the largest relative elevation; C contradicts its own ranking.
- `val_054`: not a trustworthy gain. The question's “most significant” criterion and gold are ambiguous/inconsistent. F additionally claims a 194% triglyceride elevation where the calculation is 92.2%.
- `val_062`: false positive for F. It correctly places 99 within 60–100, then says it exceeds the upper limit by 9 and also writes 100−99=1. Passing the central fact does not make this answer consistent.
- `val_124`: not a substantive gain. C already gives the correct 68.9 cholesterol difference and the near-boundary triglyceride result. The scorer favors F's use of “closest.”

Thus at least three of six apparent gains are not reliable evidence of improved numerical reasoning. This targeted review cannot be extrapolated into a corrected total accuracy.

Additional counterexamples:

- `val_239`: F gets the primary AST subtraction right but falsely says ALT 37.3 exceeds 56 by 18.7; v2.1 passes it.
- `val_075`: F repairs an additional multiplicative claim, but v2.1 passes both responses and misses the improvement.
- `val_001`, `val_085`, `val_185`: range membership, sign and subtraction errors remain.
- `val_202`: F's largest-excess ranking is wrong and it reports LDL's difference as 19.3 instead of 119.3.

The bounded scorer appropriately leaves many items unresolved: C has 18 pass / 4 fail / 28 review; F has 21 pass / 2 fail / 27 review among 50 numeric cases. This is not a verified ranking of complete semantic accuracy either.

Next measurement action: audit all 50 numeric answers for the finalists using a frozen claim-level rubric: input-derived reference range, selected analyte, direction, arithmetic, units, ranking criterion, and contradictory or incorrect additional claims. Mark under-specified questions and disputed gold explicitly rather than forcing pass/fail. Develop general evaluator rules and independent synthetic regression fixtures; do not copy validation-specific repairs into training labels or tune a prompt repeatedly to these cases. Preserve old scorer reports and label any revised evaluation as post-inspection.

## 3. Prompt and few-shot results

Correct grounded tool-task outcomes, out of 55:

- R0-v1: 29 (BMI 24/40, conversion 5/15).
- R0-v3: 9 (BMI 7/40, conversion 2/15).
- R0-v3-FS4: 29 (BMI 27/40, conversion 2/15).
- C and F epoch two: 54 each.
- R0-8B: 46 (BMI 31/40, conversion 15/15).

V3 has no tool call in 32/40 grounded BMI questions and 11/15 conversion questions. The result supports a tool-compliance regression under this prompt, not the claim that a longer or stricter prompt is automatically better. It is a composite prompt change, so the experiment does not isolate one offending sentence.

FS4 improves BMI tool selection to 33/40, but six selected BMI calls fail the existing argument checks. Conversion tool selection remains only 4/15. Its one BMI tool demonstration may explain the uneven coverage, but this is a hypothesis rather than an isolated causal result.

P1 zero fabrication alone is misleading: base v1 and v3 both have 0/34, yet their original-partner valid-call success is only 20/34 and 5/34. FS4 has two automatic fabrication flags. Always pair missing-input behavior with useful behavior on intact inputs.

Decision: retain v1 as the current comparator. Do not promote v3 or FS4. A later prompt experiment should distinguish an obligation to call a tool on supported tool tasks from a prohibition on inventing missing arguments, and cover both tool schemas if demonstrations are used. Define that intervention from the task contract and train records, then freeze it before another evaluation.

## 4. Larger base model is not sufficient

R0-8B completes more grounded tool tasks than R0-v1, but the P1 classifier flags fabrication on 28/34 probes, and the seven natural Q5 cases exhibit unsupported tool arguments. It reaches the generation budget in 26/250 validation trajectories. Its template, prior post-training and budget interaction differ from 4B.

This supports a grounding problem under the current protocol. It does not prove that 8B capacity is inferior, and there is no 8B SFT result to compare. Deferring that training in favor of the already running Qwen3.5 extension remains reasonable.

## 5. Aggregate scores and confidence diagnostics

Diagnostic v2.1 macro scores: R0-v1 73.80%, v3 66.96%, FS4 77.55%, C 91.41%, F epoch one 94.38%, F epoch two 96.69%, R0-8B 68.77%. These are scorer outcomes, not verified clinical accuracies.

F epoch two versus C is +5.28 percentage points, with an exploratory type-stratified paired bootstrap interval of [+2.22, +8.67] points (10,000 resamples, seed 42, unadjusted). Legacy v1 gives +3.07 points [+0.25, +6.23]. Resampling does not fix scorer bias or measure training-seed variation.

F's extractive count falls from 100/100 to 99/100; `val_191` omits the “after two weeks” timing detail while retaining the medication sequence. Its uncertain count improves from 35/38 to 37/38, including missing-dose/allergy coverage in `val_025` and `val_110`; this is not an audit of all treatment recommendations.

C10a final-answer mean-log-prob AUROC under v2.1 is approximately 0.750 for C and 0.640 for F epoch two. Error sets and scorer labels differ, so do not call this a calibrated deterioration in confidence. C10b grounded call-prefix AUROC is 1.0 for both, despite C's 25 fabricated P1 answers. Predicting whether a call prefix should appear is different from predicting answer truthfulness. No ECE was computed.

## 6. Training and cost

C and F take approximately 42.7 and 43.9 minutes to train, with median recorded steps around 10.21 and 10.13 seconds and peak allocated memory around 7.81 GiB. No loss-divergence evidence appears across the two epochs. This does not establish that longer training cannot overfit.

Teacher-forced validation loss is slightly worse for F (epoch two 0.3470 versus C 0.3414) while missing-input behavior improves. Canonical unsupported Q5 targets and the different metric objective limit its suitability for checkpoint selection.

Train-fit legacy numeric outcomes are C 32/50 and F 34/50. They remain scorer-dependent and cannot establish an architectural capacity ceiling.

Validation generation takes 1,083.6 seconds for C, 1,000.9 for F and 846.0 for FS4. FS4 uses roughly 4,484 first-turn prompt tokens versus 1,294 for C/F, but measured wall time is not slower in this run. Do not advertise a measured SFT latency advantage over FS4 from token count alone. Tool turns and host/runtime overhead differ; there is no repeated serving benchmark.

## 7. Next decisions, without changing the running Qwen3.5 round

1. Finish the frozen Qwen3.5 runs. Compare each base against its own SFT, and compare A-Q35-4B filtered against C-filtered-s42. Comparing Q35 filtered directly with F relabeled confounds model family and data policy.
2. Read every Qwen3.5 P1 and natural-Q5 response, and pair those outcomes with intact-input tool success. A higher numeric score cannot compensate for invented measurements. If Q35 filtered repeats the control's fabrication, a matched Q35 relabel run is the next targeted intervention.
3. Replicate F with seeds 43/44 after the existing reviewed engineering gate is recorded. Keep the same recipe; quantify per-seed P1 and paired-partner outcomes as well as scorer diagnostics. Do not use this single seed to launch an unrestricted LR search.
4. Complete the claim-level numeric audit before selecting a numerical-reasoning winner across model families. Preserve a review/ambiguous category. A training intervention, if justified, must use train-derived examples and a separately frozen validation protocol.
5. Only then decide whether a prompt/tool-demonstration experiment or a bounded numeric training intervention is worth the GPU budget. Current evidence does not require RL.

Hardware disclosure: D-089 records the running Qwen3.5 stage on A100 80GB with flash-linear-attention and potentially concurrent generation/training. Core results are RTX 4090. Cross-family performance includes a kernel/hardware component; timings under contention are not comparable serving-cost estimates. Preserve runtime/kernel provenance when results arrive.

## Evidence and reproducibility

Fresh CPU rescoring and diagnostics are saved alongside this review: `v21/`, `bounded_v2/`, `p1/`, `trainfit/`, `training/`, `gates/`, `c10_v1/`, `c10_v21/`, `results_overview.json`, and `paired_diagnostics.json`. Existing scoring and analysis scripts completed successfully. No source-code changes, GPU runs, gate approvals or test-set evaluations were performed for this review.

The earlier test-consumption disclosure remains in force: the old test set is not an untouched future holdout. These new decisions use validation, P1 and train-fit only. Subsequent evaluations on the same validation set are adaptive and must be described accordingly.
