> **Archived 2026-10-03.** Superseded by `docs/PROJECT_WALKTHROUGH.md and docs/DECISIONS.md`. Kept unchanged below as a historical record; relative links and some script paths (now `scripts/legacy/`) may be stale.

# Wave 3 experiment plan — prioritized after interview review

Updated 2026-10-01. Status: core plus the requested 8B stage implemented; four artifact approvals completed after AI grounding review; no new GPU training or generation launched. This replaces the earlier run queue, archived under history/interview_review_2026-10-01/. Existing decision history remains intact. Interpret INTERVIEW_PREP through INTERVIEW_REVIEW, SCORER_V2_1_AUDIT and WAVE3_REVIEW. All development uses train/validation; no test-derived intervention or new final-test claim is part of this queue.

## Current execution status — supersedes earlier draft counts and readiness notes

See WAVE3_APPROVAL_REVIEW.md and DECISIONS D-075 through D-086. Q5 review accepts 78/78 with 14 revised answers; the relabel view has 2,000 rows. Prompt v3 is approved with clarified unit-conversion wording. The four unchanged demonstrations produce 4,382–4,639-token validation prompts. P1 is frozen at 34/40 eligible sources, with six question-measurement exclusions; the 5% point gate permits at most one fabrication. Train-fit is 200 IDs, 50 per type, on both the filtered control and F-s42. The requested 8B filtered SFT stage is implemented, rather than merely an optional future proposal; it still requires GPU smoke validation.

Local checks: 254 tests passed and the relabel mask audit has zero problems. The four requested artifact approvals are complete and are explicitly AI review, not human clinical adjudication. The evaluator/quality gate is separate: known semantic scorer defects and newly documented P1 classifier limitations remain unresolved. Do not approve F-s42 or launch its seed continuation solely from the automatic suggested gate. No GPU results exist. The methodological plan below remains applicable where not superseded by these concrete implementation details.

## 1. Main question and fixed control

Test whether audited Q5 relabeling improves missing-input behavior compared with the existing filtered model, while preserving correct calls. Then determine whether SFT offers a useful quality/cost advantage over strong fixed prompts. Keep optional architecture/precision experiments separate.

For the 4B SFT control, retain the pinned Qwen3-4B-Instruct-2507 revision, NF4, LoRA r16/alpha32/dropout0.05 on the existing seven projections, LR 1e-4, cosine, 3% warmup, two epochs, micro-batch1/accumulation16, seed42, native template and system_v1. Canonical data stays unchanged. Save both epochs; epoch two is the prospective primary endpoint, not a claim it was previously best. Epoch-one reporting is diagnostic. Fix generation budgets/precision/batch across matched comparisons; use batch2 as the safe starting point and rerun relevant controls if needed. A longer few-shot context needs a separate memory preflight and measured token counts.

## 2. Gate E0 — evaluator, data and reproducibility (CPU plus targeted review)

- Fix or independently adjudicate the demonstrated incomplete-answer, wrong-extra-claim, numeric-role and reference-source defects before semantic ranking. Preserve v1/v2.1 as versioned diagnostics. Do not wait for a universal clinical judge, but require coverage for the endpoints actually used. Unresolved obligations remain explicit, not automatic failures or passes. Record a blinded calibration packet including correct and intentionally incorrect variants plus actual base/SFT outputs; label provenance and scorer version matter more than a small headline agreement statistic. A 40-60 item targeted audit is an initial debugging set, not proof of low deployment error.
- Audit the 78 train-only Q5 proposals. Approve only genuinely unsupported calculation targets; retain original IDs/inputs, record evidence and transformations. Build q5_relabeled from 1,922 unchanged filtered rows plus k accepted replacements. Counts are 1,922+k, not automatically 2,000. Add the transformation-aware training-view audit, distinct formatting output, mask/length checks and hashes before training.
- Reconcile train-extra installation/export and record current commits/dirty-file hashes. The repository already has Git history. Compute exact supervised-token counts using the real template/masks; these are token shares, not gradient influence. Extract existing runtime and memory diagnostics without new GPU runs.
- Define intended-answer and full-visible-response factuality metrics separately. Check in-call and in-text fabrication, not just the absence of a call. Retain complete records/denominators and fail evaluation explicitly on missing outputs.

## 3. E1 — stronger baselines before expensive expansions

Evaluate on the same 250 validation questions:

- R0-v1: existing parity prompt; reuse outputs only if the protocol matches.
- R0-v3: one frozen stronger prompt, selected from task requirements and permitted training evidence, no validation-output prompt sweep.
- R0-v3-FS: the same strong prompt with four fixed audited training demonstrations, one per type. Record IDs, order, exact rendered messages and tool execution. Gold-self-check success alone is insufficient for demonstration approval. Keep demonstrations out of the train-fit diagnostic subset.

Report task outcomes, fabrication/over-refusal, tool completion, mean input/output tokens, wall-clock seconds per completed request, throughput and peak memory. Few-shot cost must include prefill; do not describe a 7k-token input as compatible with a 2k formatting cap without checking the actual inference path. No silent truncation. If only FS needs a smaller batch, measure deployment cost explicitly and disclose the batch difference; do not confuse batching with model quality. Stronger prompts are comparators, not a prompt-engineering ceiling. If FS matches SFT, report that; any cost/robustness claim still needs measurement.

## 4. E2 — Q5 relabel control and seed replication

Train F-s42 on the audited relabel view under section 1. Compare with filtered1e4 epoch two, generated under the identical protocol. This estimates a policy change including changed sample count/class mix/updates. Do not call it a pure label effect. If mechanistic attribution matters, add a separately defined exposure-matched control with fixed update/token budget. Do not silently alter epoch schedules to force a favorable comparison.

Freeze P1-expanded before generating new answers. Validation contains 40 grounded BMI source records, not 55. For each eligible source, create one both-missing counterpart by removing every required measurement mention across note/table/question and any derived BMI that reveals the answer. Exclude cases that cannot be edited naturally and record why. Preserve the original answerable partner. Label intended behavior independently of the model and verify generated probe contracts. Report eligible/excluded counts and count each source as one cluster. These are synthetic validation probes, not a fresh test set.

Run the original/edited pairs on filtered1e4 epoch two, F-s42 and all three R0 variants. Report natural seven-Q5 outcomes separately. Use both-missing refusal/fabrication plus valid-call preservation on original partners; a model that refuses everything fails the intent.

Provisional engineering release conditions, frozen before F-s42 outputs:

- Zero schema/parse failures on completed evaluation records.
- P1 fabrication point rate at most 5% (at n=40, at most two cases), with the Wilson interval and case list reported. This is a point gate, not a claim the true rate is below 5%; even zero failures gives an upper bound about 8.8%.
- No newly unsupported calls/measurements on the natural seven-Q5 slice relative to the control; inspect ambiguous statements rather than blindly trusting the existing regex.
- No new observed valid-call failures on the 40 original paired BMI cases relative to the matched control. This is a strict operational screen, not statistically demonstrated noninferiority. A failed/inconclusive screen triggers review rather than moving the threshold.
- Paired task outcomes and visible-claim factuality must be reported; incomplete evaluator coverage prevents a semantic winner declaration.

If F-s42 clears the reviewed gates, run F-s43 and F-s44 unchanged, evaluating each on the same endpoints. Report all seeds, not the best one. If it fails, diagnose annotation/scorer/behavior errors before spending on repeats; any changed recipe becomes a new version. Three relabeled seeds versus one historical filtered seed measure relabel stability, not a balanced multi-seed causal estimate. Replicate filtered controls at seeds43/44 only if claiming a robust across-seed treatment effect.

## 5. E3 — train-fit and calibration diagnostics

D-TRAINFIT: freeze 200 stratified train IDs excluding few-shot demonstrations, then generate with F-s42. Score with audited obligations plus historical diagnostics. Report approximately 40 numeric cases and their actual errors; low fit alone does not diagnose capacity. Do not train on validation examples or use diagnostic train results as generalization evidence.

C10a: use mean final-answer logprob for correctness ranking (AUROC, risk-coverage, per-type distributions). Include exact alignment and missing/truncated token checks. Do not compute answer-correctness ECE from raw logprob or its exponent and label it calibration. A separately fitted correctness-probability model needs grouped calibration/evaluation separation or cross-fitting and enough labeled cases; otherwise omit that ECE.

C10b: report call-prefix probability versus input-grounded call policy, annotated labels as a separate diagnostic, and actual executed-call confusion matrices. Verify immediate-call grammar and token identity. Brier/ECE, if shown, must be labeled prefix-event/policy diagnostics with bin counts; do not assert probability of eventual tool use. Avoid treating repeated seeds/checkpoints on one question as independent samples.

Use paired question differences, discordant counts and type-stratified bootstrap for macro; show seed SD separately. A CI containing zero means inconclusive. No independent-binomial MDE table for macro. Power/sensitivity planning should specify plausible discordance rates and the actual paired endpoint.

## 6. E4 — choose a targeted data/representation ablation

After the control works, prioritize A-IMPL if the matched implicit-allergy slice/probe confirms the hypothesized shortcut. Rewrite a fixed audited set of training questions with unchanged meaning/answerability, not merely removal of one keyword. Keep all other settings at F-s42. It is a question-form intervention, not proof about the model's internal strategy.

A-VIS is an alternative or subsequent numeric-only explanation intervention on audited, complete input-derived calculations. Gold agreement alone cannot validate a rationale. Report eligible target count and supervised-token changes. Check all Working and Answer claims. Do not include imperial call-line changes motivated by test-only errors. Both ablations use seed42 initially; positive results are exploratory until confirmed. They do not enter the fixed control retrospectively.

## 7. Optional extensions — each has a separate trigger

**Quantization:** first R0-BF16 versus R0-NF4 with identical prompt and generation protocol, after longest-context memory preflight. If it reveals a material accuracy/latency tradeoff or this engineering question is a priority, train A-BF16 on the same approved relabel view. Evaluate NF4-trained and BF16-trained adapters under a common serving precision; preferably fill the 2x2 train/serve precision grid. Match actual module dtypes and preparation semantics where possible. A diagonal comparison measures a full-recipe difference, not training quantization alone. No promised 13 GB memory ceiling.

**8B:** first evaluate a pinned Qwen3-8B zero-shot comparator with its own non-thinking template. Its official card supports enable_thinking=False, but template/masks/stop tokens/call-prefix diagnostics need re-auditing. Only add A-8B training if audited numeric failures and baseline results justify the cost or if the model comparison itself is a required deliverable. A single 8B result does not prove/disprove a capacity bottleneck; post-training and templates differ. Do not bundle with BF16 or rank changes.

**Rank64:** consider only after clean train examples still fail and optimization/decoding/evaluator issues have been inspected. Match model/data/precision, use r64/alpha128 to hold alpha/r constant, and report parameter count and runtime. It is not automatically triggered by noisy train accuracy near validation accuracy. Run either the rank or size experiment first according to the diagnosed question, not both by default.

Tool-result tampering, MiniCheck, serialization changes, a data-size curve, RL and broader sweeps remain deferred. No new test run is required to answer this plan's primary questions; any later reused-test report must be labeled exploratory and remain outside development decisions.

## 8. Budget, implementation and deliverables

Planning assumption pending user preference: core first, approximately 3-5 GPU hours on the observed RTX 4090, excluding CPU annotation/evaluator work, setup and optional ablations/extensions. Three control trainings alone are about 3 x 43-44 minutes; longer targets/context and new GPU types need measured costs. Baseline/few-shot evaluation, P1 pairs and train-fit can consume the remaining budget. Run F-s42 plus baselines first and replace estimates with measured cost before launching additional jobs. Optional BF16/8B/rank work gets a separate budget; no spending is authorized by this document.

Implementation prerequisites: versioned evaluator endpoint checks; reviewed relabel builder and transformation audit; probe generator with manifests; prompt/few-shot message builder and length audit; fixed epoch/seed manifests; diagnostics scripts; correct train dependency export; updated report. C10/8B/BF16 names in the interview document do not mean these implementations already exist.

Deliverables: input/config/prompt/view/scorer hashes; per-item paired outcomes; all seed results; natural-Q5 and synthetic-P1 sections with denominators; complete visible-claim failure taxonomy; resource costs; and a report stating what remains unresolved. Preserve original v1/v2 outputs and historical decisions. No benchmark outcome is to be written before the run exists.


## Proposed addition (2026-10-02, not implemented)

- **P1-RAW**: run the 34 frozen P1 probes and the 7 natural Q5 items on the existing wave-1 `raw_lr1e4` checkpoint-250 adapter. Use the wave-3 inference protocol and the `p1-2` classifier, inference only. This is a label-only control for F-s42: same rows, steps and seed; only the 78 Q5 labels differ. Run it before F-s43/44. Rationale: INTERVIEW_PREP.md §5.8.
- **Done 2026-10-02 (CPU only):** clinical-context check (`scripts/clinical_context.py`), call-prefix ECE in `w3_analyze.py c10`, and numeric audit stratification (`scripts/numeric_audit_prep.py`). Results: INTERVIEW_PREP.md §5.14.
- **Planned 2026-10-02:** Stretch A (`calculate_egfr`), see [STRETCH_A_PLAN.md](STRETCH_A_PLAN.md). Not implemented.
