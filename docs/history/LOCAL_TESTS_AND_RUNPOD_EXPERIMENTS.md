> **Archived 2026-10-03.** Superseded by `docs/RUNBOOK.md`. Kept unchanged below as a historical record; relative links and some script paths (now `scripts/legacy/`) may be stale.

# Local verification and RunPod experiment inventory

Date: 2026-10-01. Status: local checks completed; GPU experiments not launched. This operational inventory supplements EXPERIMENTS_WAVE3.md. All new experimental decisions use train/validation only. Experiment names below are proposed identifiers, not claims that their configs or outputs exist.

## Local checks actually executed

- Full suite: `./.venv/bin/python -m pytest -p no:cacheprovider`: **240 passed, 4 warnings, 43.69 seconds**. Warnings concern the tiny resume-test Conv1D LoRA setting and pinned memory without an accelerator; no test failed or was skipped. This includes a real small CPU Trainer resume test, not a full 4B training run.
- Offline native-tokenizer mask audit: raw **2,000/2,000**, filtered **1,922/1,922**, zero problems. Both rendered length ranges are **1,219–1,614 tokens**, below the existing 2,048 training cap. Supervised token means are 76.1 and 74.2; supervised token shares are 5.5% and 5.4%. These are token shares, not gradient influence. New relabel/rationale/few-shot formats need fresh audits.
- Read-only `audit_train_view` checks: raw_lr1e4, q5filtered_lr5e5 and all three completed wave-two mb1 LR configs pass canonical hashes, implementation/config provenance, ordered membership and expected counts. Effective batch is 16 throughout. Raw explicitly acknowledges 78 Q5 flags; filtered has zero remaining flags. Passing raw is policy compliance, not approval of unsupported targets.
- `bash -n`: all four shell scripts pass syntax checking. This does not execute or validate their GPU workflows.
- All 14 wave-two v2.1 score files contain exactly 250 unique IDs matching canonical validation: **3,500 records**, no missing/duplicate IDs. This checks completeness, not semantic score correctness.
- Local environment: torch 2.14.0, transformers 5.17.0; CUDA unavailable. CUDA kernels, bitsandbytes, 4B training memory, throughput and long-context inference are not verified locally.

Evidence is exported alongside this document in `local_checks_2026-10-01/checks.json` and the raw/q5_filtered mask reports. Canonical data, adapters and historical score outputs were not regenerated. The suite checks canonical split structure/hashes, including test-file integrity; no test performance was used for a new development choice.

## Blocking local prerequisites before new GPU jobs

1. Repair or independently adjudicate the known scorer endpoint defects: incomplete answers, incorrect extra claims, number-role reversal and unavailable reference ranges. Passing the current suite does not validate clinical semantic accuracy. Preserve historical scorers; add targeted regression cases and a blinded calibration packet. Until resolved, v2.1 remains diagnostic and cannot declare a winner.
2. Review all 78 train-only Q5 proposals. Build a versioned relabel view containing 1,922 unchanged filtered rows plus k approved replacement rows; do not assume k=78. Preserve canonical files and original inputs/IDs. Add transformation-aware training guards: the current guard only accepts raw/q5_filtered and would reject a relabeled view.
3. Freeze stronger prompt v3 and four audited training demonstrations. Implement the few-shot message path, retain complete tool interactions, measure actual context lengths and forbid silent truncation. The current mask results do not cover this path.
4. Build and audit P1 paired probes from up to 40 grounded validation BMI sources. Remove all required measurement mentions and derived answer leakage in edited partners; document exclusions. Preserve original answerable partners. Keep seven natural Q5 records separate. This is synthetic validation evidence, not a new holdout.
5. Add run configs/manifests for new views/seeds, probe generation/evaluation, train-fit IDs and diagnostics. Freeze hashes, metrics and release gates before new outputs. Save full visible responses and per-item outcomes.

These prerequisites were reviewed, not implemented in this local-test pass. Existing scripts/configs support historical controls; the proposed Wave3 workflow is not yet turnkey.

## RunPod stage 0: environment and smoke checks

Use the existing locked setup script and train dependency extra. Verify the host driver against the checked-in setup requirement, CUDA availability, dependency versions and immutable input/adapter hashes. Run a fresh GPU smoke job before paid full training; test longest training examples and separately the longest few-shot inference context. Use fresh run/output directories and explicit resume only for the same run/config.

Starting settings: training micro-batch 1, accumulation 16; evaluation batch 2. The previous mb4 allocation retries and slower generation do not justify increasing batch sizes by default. Record actual BF16 support/dtypes; the current config uses compute_dtype=auto, so the realized dtype must be checked. Freeze the serving precision and decoding protocol across matched comparisons.

Do not run `make core`, `make final-eval`, or the old wave-two round as a substitute for this queue. They do not implement the new selection contract. No new test evaluation is scheduled.

## Core GPU queue: first tranche

### R0-v1, R0-v3 and R0-v3-FS4: inference only

Question: how much benefit remains for SFT against stronger prompting?

Run the pinned 4B base with the parity prompt, one frozen stronger zero-shot prompt, and that prompt plus four audited training demonstrations. Each receives the same 250 validation questions and P1 pairs. Reuse R0-v1 only when prompt/template/precision/decoding protocol matches; otherwise regenerate. Do not sweep prompts against validation outputs.

Report semantic outcomes, full-response factuality, call/schema completion, missing-input fabrication, over-refusal and measured cost. Include input/output tokens, latency, throughput and peak memory. A few-shot batch reduction must be disclosed for latency comparisons. Readiness: R0-v1 infrastructure exists; v3/few-shot assembly and audits remain required.

### C-filtered-s42: matched inference control, usually no retraining

Use existing filtered LR1e-4 epoch-two checkpoint (step242). Run validation and P1 under the new common inference protocol. Existing wave-two batch4 outputs are not the default latency comparator for batch2. Verify checkpoint identity and availability on the Pod before reuse. This is a prospective reference recipe, not proof that LR1e-4 is optimal.

### F-s42: one new training run

Train the approved relabel view at LR1e-4, two epochs, seed42, NF4, LoRA r16/alpha32/dropout0.05, existing seven target projections, cosine/3% warmup, micro-batch1/accumulation16, system_v1 and max length2048. Pin the existing model revision cdbee75f17c01a7cc42f958dc650907174af0554. Save both epochs; epoch two is primary, epoch one diagnostic. The view has 1,922+k rows, so calculate steps from the approved count rather than copying step242.

Evaluate on all 250 validation items and P1 pairs against C-filtered-s42 and the base prompt arms. This comparison changes data policy, sample/class mix and optimizer exposure; it is not a pure label-only causal effect. An exposure-matched control is optional if that narrower attribution is required.

Freeze the planned operational gates: zero parse/schema failures; P1 fabrication point rate <=5% with counts and Wilson interval; no new unsupported calls/measurements on seven natural Q5 cases; no new valid-call failures on original grounded BMI partners relative to control. At 40 edited pairs, <=2 failures passes the point gate, but even 0/40 has an upper Wilson bound near 8.8%. Incomplete semantic coverage prevents a winner declaration. Missing outputs invalidate the run rather than shrinking the denominator.

## Core GPU queue: conditional continuation

### F-s43 and F-s44: two new training runs only after F-s42 passes review

Hold the complete recipe fixed and repeat validation/P1. Report all seeds and seed variability. Three relabeled seeds versus one filtered seed demonstrate stability, not balanced multi-seed causal evidence. If making the stronger across-seed treatment claim, additionally train filtered controls C-filtered-s43 and C-filtered-s44 with the same protocol.

### D-TRAINFIT: inference diagnostic

Generate F-s42 outputs on 200 frozen stratified training examples, excluding few-shot demonstrations, including approximately 40 numeric examples with actual counts recorded. Audit errors rather than treating low fit as automatic evidence of insufficient capacity. This is training-fit diagnosis, not generalization evaluation.

Confidence extraction should be collected during these GPU runs if needed. AUROC/risk-coverage, paired bootstrap, confidence intervals and report aggregation run locally from saved outputs/logprobs. Mean answer logprob is a ranking statistic, not a calibrated probability. Verify token alignment and missing/truncated records before analysis. Call-prefix probability and actual executed-call behavior require separate endpoints.

## Optional GPU experiments: trigger individually

- **A-IMPL:** one seed42 training ablation on audited meaning-preserving implicit-allergy question rewrites, only after matched validation probes establish the hypothesized weakness. Requires a new data builder/audit. Follow with replication if claiming a reliable gain.
- **A-VIS:** alternative or subsequent seed42 numeric-only explanation supervision, with input-grounded complete calculations and full Working/Answer factuality. Audit changed token exposure and lengths. Do not add imperial-call explanations motivated by test-only observations.
- **R0-BF16:** inference comparison against matched NF4 base, after a memory preflight. If worthwhile, **A-BF16** trains the same approved view; compare adapters under common serving precision and preferably complete the 2x2 train/serve precision grid. A diagonal comparison cannot isolate training quantization. No unmeasured memory guarantee.
- **R0-8B:** pinned non-thinking 8B zero-shot inference, with independently verified template, masks and stop tokens. **A-8B** training follows only if baseline/error evidence or a specific model-comparison requirement justifies it. Model size is confounded with post-training/template differences.
- **A-R64:** r64/alpha128 training only after clean train-fit errors survive evaluator/optimization/decoding checks. Change rank alone; inspect parameter count, memory and runtime. Do not launch both rank and size expansions automatically.
- **Exposure-matched and filtered-seed controls:** additional training only for the corresponding attribution claims above. Define their budgets and estimands before execution.

RL, another broad LR sweep, tool-result tampering, MiniCheck, serialization changes and a data-size curve are deferred. The completed LR ladder should first be re-analyzed with repaired endpoints locally. It need not be paid for again merely because a new wave has started.

## Run count, budget and stopping rules

Initial commitment after prerequisites: **one F-s42 training job**, three base inference arms, matched filtered inference, P1 evaluation, and train-fit inference. P1 has up to 80 records per arm; original partner generations can be reused from matching validation outputs, leaving up to 40 extra edited generations per arm. Reuse requires identical prompts and inference settings. Conditional continuation adds **two training jobs**, not an unconditional launch of every optional ablation.

The historical filtered runs took about 43 minutes each on the observed 4090. Three similar trainings alone therefore cost about 2.15 GPU hours; relabeled targets and context may change this. The prior 3–5 hour core estimate is provisional, excluding implementation/annotation/setup and optional experiments. Measure the first long-context and F-s42 jobs before committing the remainder.

Stop expansion when data/evaluator gates fail, outputs are incomplete, or new fabrication appears. Diagnose and version a changed recipe rather than moving gates after seeing results. Archive run/config/prompt/data/scorer hashes, per-item outcomes, cost measurements and all seeds. Keep test-derived observations quarantined.
