# Decisions — merged wave-three review

Revision 4 updated 2026-09-30: wave-one outcome decisions (D-037 to D-040) and the wave-two round (D-041 to D-052). Revision 3 was dated 2026-09-28. [PLAN.md](history/PLAN.md) is the implementation specification for wave one, and [EXPERIMENT_JOURNAL.md](EXPERIMENT_JOURNAL.md) records the evidence behind each decision. Every entry gives the decision and the reason for it. Entries marked *user decision* were made by the user; where one overrides earlier advice, the entry names that advice and the confound it introduces. Planned results are never recorded as measured.
## Current decisions (original IDs retained)

- D-001: NVIDIA CUDA, Transformers + TRL + PEFT; single 24GB GPU target confirmed by the user.
- D-002: Default Qwen3-4B-Instruct-2507. A second-model comparison is optional and time-boxed. This supersedes the earlier mandatory comparison of 2–3 models.
- D-003: **Superseded:** original single-call targets are the Core baseline. Sequential explicit imperial conversion is a separate R3 ablation, not the default formatter.
- D-004: Core first, R2 data-quality ablation second, then R3/diagnostics and optional Stretch B. No requirement to beat base on every class before optional work.
- D-005: Repository documentation and comments remain English; a Chinese user-facing plan is supplied separately.
- D-006: Assignment stays in docs/ASSIGNMENT.md; requirements.txt is the dependency export.
- D-007: Target Python 3.11 + uv lock. The copied local virtualenv interpreter is broken and must be recreated on the GPU host. Training dependencies remain unpinned until the compatibility smoke.
- D-008: Keep the existing deterministic tools, with explicit round(2) conversion policy and round(1) BMI.
- D-009: Generated data analysis plus regression tests; heuristics produce review candidates, not automatic proof of a bad gold answer.
- D-010: Canonical files remain byte-identical to supplied source files; verify live hashes before analysis/views.
- D-011: docs/ for design, reports/ for generated evidence; personal _document/ is ignored.
- D-012: Raw baseline keeps all 2,000 rows. A separate train-only Q5 selection currently retains 1,922; excluded IDs and full review packets are saved. No relabeling, no test/val filtering.
- D-013: Normalize harmless unit aliases; schema validation is stricter than executor coercion. No semantic argument repair in the model runner.
- D-014: The repo is initialized but currently uncommitted. Do not claim a versioned/reproducible commit until it exists; no commit or push was performed during this revision.
- D-015: SFT only for this assignment. RL/DPO are future research options with explicit gates in PLAN section 11.
- D-016: Q10's old arithmetic findings were parser false positives. Keep train_1890 and test_306. Regression tests prevent decimal backtracking at x/× suffixes.
- D-017: Test labels have already been audited. No gradient updates or model/hyperparameter selection on test; freeze the final comparison before generating model test outputs.

## Resolution of previous pending items

- P-001: unit_convert returns 2 dp for dataset compatibility; record the policy.
- P-002: Markdown tables, all rows retained, empty units explicitly shown.
- P-003: Short grounding/abstention policy, including explicit question values. Separate general explanation from documented patient facts; do not invent reference ranges as patient evidence.
- P-004: Native template and a tools column; verify tokenizer rendering and assistant masks.
- P-005: Direct Core traces; sequential chains only in optional R3.
- P-006: No upsampling.
- P-007: Default model as D-002; Qwen3.5-4B optional after its own compatibility smoke.
- P-008: Initial 4B QLoRA r16/alpha32, LR1e-4, two epochs, effective batch16, 2048-token starting cap after actual length inspection.
- P-009: Fact-slot, source-aware and execution-aware scoring; no universal numeric tolerance or last-tool-only success criterion.
- P-010: Add over-refusal, over-calling, unsupported arguments, actual-result incorporation and tool E2E. Do not confuse abstention behaviour with calibrated probabilities.
- P-011: R1 raw versus R2 Q5-filtered first; R3 direct versus chains second if time remains.
- P-012: Do not delete train_1890. Q5-filtered view is an explicit heuristic experiment with a review packet, not a replacement for the unchanged Core view.
- P-013: Optional separate Stretch B with about 30 curated examples, grouped split, NOT_FOUND handling and core regression check.
- P-014: Resolve exact CUDA/dependency versions only on the actual GPU; record smoke-test evidence.
- P-015: Python 3.11 remains the target. CPU checks in this revision used the available Python 3.13 with copied pure-Python packages, not a claim of GPU-environment validation.

## Revision 3 decisions (2026-09-28, agreed with the user in a design interview)

Implementation scope and interfaces. These supersede earlier entries where they conflict.

- D-018: Execution: RunPod single 24GB GPU, run by the user from the README runbook. Development and unit tests run locally on CPU; Qwen3-0.6B is a smoke-test model only, never a reported result. Persistence: network volume at /workspace (HF cache, checkpoints, outputs); final adapters pushed to a private HF Hub repo; small outputs (trajectories, scores, metrics) are committed.
- D-019: Scope: must-ship is R0 (base + tools) and R1 (raw 2,000). R2 (Q5-filtered) is the only optional training run; if trained, spot-check 15 stratified Q5 packets and report false positives. R3, Stretch B and hyperparameter sweeps are next steps only.
- D-020: Single model Qwen/Qwen3-4B-Instruct-2507 (revision pinned). Qwen3.5 dropped from scope.
- D-021: Loss masks are built by us, not by TRL assistant_only_loss (the stock Qwen3 template has no generation markers). Prefix-diff rendering with an exact-prefix assertion; the assistant header is masked, assistant content and its <|im_end|> are supervised.
- D-022: Training loop: plain transformers Trainer + PEFT with a padding collator over pre-tokenized input_ids/labels. One R1 run with the P-008 config, no sweep; save per epoch and select on val. Log train_log.jsonl + TensorBoard; W&B off by default.
- D-023: Inference: HF generate, greedy, nf4 base for both R0 and adapters. Budget: one successful call, two assistant turns, 256 new tokens per turn, 512 total.
- D-024: Wire format: Qwen native <tool_call>{"name","arguments"}</tool_call> via apply_chat_template(tools=...). Call turns carry no text. substance is always emitted (null for body measurements). Units are free strings with supported pairs listed in the description (no enum). Tool responses are JSON: {"result": x} or {"error": "..."}.
- D-025: System prompt lives in configs/prompts/system_v1.txt, hashed into every run manifest, frozen after R1; any change is a new version.
- D-026: Over-calls are executed and counted; calls past budget stop with stop_reason=budget and count as failures; malformed JSON is not repaired (schema_status=invalid, no retry).
- D-027: Argument scoring: metric-source values must match the input at source precision; imperial-derived values pass within half a unit of the last digit of either gold or round(exact conversion, 1); unit_convert value exact, units/substance compared after alias normalization. Q5 rows are scored against gold and also counted as unsupported arguments.
- D-028: Text scoring is deterministic (no LLM judge in headline metrics): extractive slot/number/unit/direction match with token-F1 fallback; numeric key result + direction; uncertainty = abstention phrase + named missing field + no fabricated value; tool E2E = tool + args + executed result in the final answer; over-refusal on non-uncertain rows. Scorer validated on fixtures before model outputs exist; ~40 val predictions manually audited after R0.
- D-029: Logging: trajectories.jsonl (rendered prompt, raw turns, parsed calls, schema status, executor I/O, stop reason, token counts, latency, per-token logprobs, and the first-token call-vs-answer decision logprob) contains no gold. scored.jsonl is joined by ID. error_analysis.md lists failure categories, example IDs and paired R0/R1 flips. Supports later rescoring and RL/DPO scoping without rerunning the GPU.
- D-030: Test set runs once from configs/final_eval.yaml; any bug-fix rerun is disclosed.
- D-031: Version control: baseline commit of Phases 0–1b, then one commit per module; submission is a private GitHub repo. Processed-view manifests and Q5 review packets are committed; view train.jsonl copies are regenerated.
- D-032: Python 3.11 pin kept; uv rebuilds .venv. GPU stack goes in the train extra and is locked on RunPod after the smoke test.
- D-033 (post-implementation review, 2026-09-28): a schema-invalid call stops the rollout (stop_reason=schema_error) instead of receiving an error response and a second turn, which was an implicit retry contrary to D-026. Executor errors on schema-valid calls are still returned as {"error": ...}.
- D-034: unsupported_args is recorded for every call (arguments not recoverable from input measurements/numbers) and aggregated per subset; checkpoint near-ties prefer fewer unsupported calls, then fewer over-calls, then the earlier checkpoint.
- D-035: scorer hardening from an adversarial review on train/val only (test never read): negated direction words are dropped; direction is read from sentences containing the fact numbers; bound words ("upper limit") and non-lab hyper/hypo terms are non-directional; verbal thresholds ("less than 500") are context, not facts; abstention covers contractions and "needs to be obtained"; loose words (unknown/missing) count only near a documentation noun; fabrication checks accept only same-kind input values and cover spelled-out units, bare "weight was N" and asserted "no allergies"; numeric answers fail when the derived result is attached to another analyte's value. Each rule has a fixture in tests/test_metrics.py.
- D-036: final-eval refuses a dirty/uncommitted tree and any adapter that is not checkpoints[selected] in outputs/<run>/selection.json; the final config hash and commit are written into each test run.json.

## Revision 4 decisions (2026-09-30)

These supersede earlier entries where they conflict. Superseded: D-014 (the repository is now committed; wave one is at `cb0a940`, "Final test evaluation"); D-022's "one R1 run, no sweep" (wave one trained three runs); the strict one-factor order in EXPERIMENT_JOURNAL W2-PLAN-003 (replaced by D-041); and the W2-PLAN-003 E3 skip condition (replaced by D-045).

### Wave-one outcome

- D-037: **The wave-one winner is kept as selected, and test was run once.** The frozen rule picked raw_lr1e4 step125: maximum grounded macro, 0.005 near-tie margin, unsupported-call and over-call tie-breaks within the grounded subset. It scored 87.78% legacy grounded macro on val. It was kept although it makes unsupported BMI calls on 6 of 7 validation Q5 cases, where filtered step121 makes none. *Reason:* the rule was frozen before outcomes were seen. Replacing the winner after seeing a regression would be outcome-driven selection, and it would invalidate the single test run. The Q5 regression is reported next to the result (journal section 3) instead of being hidden by re-selection. Test result, legacy scorer: 84.72% grounded macro vs 45.28% for base, Q5 no-call abstention 1/10 vs 9/10.
- D-038: **Test is consumed; wave two is validation-only.** No wave-two decision reads test records, trajectories, scores or adjudication packets. `scripts/scorer_v2_agreement.py` reads test and is not run. Any future test run is labelled a *reused-test exploratory evaluation* under a new run identity, and the original final-evaluation artifacts are preserved. *Reason:* test was inspected for data auditing and has been evaluated once. A second selection against it would no longer estimate held-out performance. *Disclosure:* candidate scorer v2.1 contains fixes derived from test holdout1 (W2-AUDIT-004). This cannot be undone. It is disclosed, and no further test-driven change is allowed.
- D-039: **The Q5-filtered view is the wave-two training base, and LR is raised to at least 1e-4.** Evidence (legacy scorer, val, paired stratified bootstrap with 5,000 resamples):
  - LR effect within the raw view: raw1e4 minus raw5e5 at epoch one is +3.36 points grounded macro, 95% interval +0.66 to +6.61.
  - Filtering effect at 5e-5: filtered minus raw is -0.14 points, 95% interval -3.68 to +3.34.
  - Safety: filtered step121 keeps Q5 no-call abstention at 7/7, with unsupported arguments on 1/56 called examples vs 7/62 for raw1e4.

  *Reason:* filtering gives the safety gain without a measurable grounded-score cost, and the higher LR is the only intervention with an interval excluding zero. Filtered at 1e-4 has never been trained, so it is the first rung (D-045). *Caveats:* the Q5 set is only 7 validation cases. Filtering also changes class balance and step count (1,922 examples, 121 steps per epoch vs 125). Q5 is a heuristic flag, not an adjudicated label.
- D-040: **Three scorers are kept side by side, and none selects a model yet.**
  - v1 legacy (`metrics.py`): kept as the historical record. It is the only implementation with paired bootstrap comparison.
  - Bounded v2 (`scoring_v2.py`): kept.
  - Candidate v2.1 (`scorer_v2.py`): installed hash-matched (`b15db3d0…`), and it reproduces 1,750 validation scores. It is diagnostic only. Known defects: false passes on executed negative probes; numeric membership alone can pass `check_text`; no coverage gate; canonical ranges are injected when none are supplied (val_085).

  Next scorer design: deterministic checks remain authoritative. A pinned, cached semantic judge resolves only `review` items against a fixed rubric. It is calibrated once on a stratified validation sample (development and holdout halves, predeclared agreement and false-pass thresholds), then frozen and applied automatically to every checkpoint, with no per-item human labelling during training. The backend is not chosen. *Reason:* the audit showed v2.1 passing wrong answers, so selecting with it would reward scorer artifacts (W2-AUDIT-004, W2-INSTALL-005).

### Wave-two round

- D-041: **One bundled GPU round, per user decision.** One pod session runs E1 (prompt ablation on fixed weights) and the filtered LR ladder (D-045), both in the new batch setup (D-042). Command: `make w2-round`. *Reason:* pod setup, preflight and GPU smoke are paid once. The jobs are independent: E1 uses existing weights, and the ladder uses the v1 training prompt. Causal reading comes from matched contrasts, not wall-clock order (W2-INSTALL-005). *Consequence:* the prompt result cannot feed into the training prompt this round, so training stays on v1 (D-048).
- D-042: **Batch setup: training micro-batch 4 with accumulation 4, and generation batch 4 (user decision).** The effective batch of 16 is unchanged.
  - *Reason:* throughput. Wave-one mb1 training peaked at 7.81 GiB of 23.5 GiB.
  - *Overrides:* W2-AUDIT-004 and W2-INSTALL-005, which advised training filtered 1e-4 at mb1 to match wave one and preflighting the batch changes separately.
  - *Confounds, stated rather than removed:*
    1. Every wave-two run differs from the wave-one runs in micro-batch as well as view or LR. The optimizer math is nominally the same at effective batch 16, but padding, reduction order and bf16 numerics differ. Filtered 1e-4 at mb4 vs filtered 5e-5 at mb1 therefore does not isolate LR, and the mb1 control `w2_filtered_lr1e4_mb1` is not run (E2b stays unmeasured, D-048).
    2. Batched greedy generation with left padding can change outputs. Wave-one outputs (batch 2) are therefore not matched controls (D-043).
  - The three LR rungs share one setup, so they are comparable with each other.
  - *Why batch 4 and not 8:* 8 has not been tested for memory or latency at 512 total new tokens, and 4 is the smaller step.
  - *VRAM risk:* the estimated mb4 peak is about 20–22 GB, dominated by fp32 logits over the 151k vocabulary at 2,048 tokens. If the first mb4 run hits CUDA OOM, `scripts/run_w2_round.sh` writes `outputs/w2_mb4_oom.txt` and switches the whole ladder to mb2 with accumulation 8 (same effective batch). All rungs therefore stay in one setup. This fallback is a deviation to be reported if it happens.
- D-043: **Controls are regenerated under one code state and one generation batch.** The E1 v1 arm regenerates all four fixed checkpoints with the wave-two code and batch 4. Two comparisons follow:
  - `w2p_v1_<run>` vs the wave-one label (`scripts/compare_rollouts.py`) measures drift from generation batch, code and host. That drift is the noise floor for reading v1-vs-v2 and ladder differences.
  - Ladder checkpoints are compared with `w2p_v1_raw_lr1e4` and `w2p_v1_q5filtered_lr5e5`, not with the wave-one outputs.

  *Reason:* the evaluation source hash changed after wave one (scorer modules were added), and so did the generation batch. Wave-one outputs are no longer matched controls.
- D-044: **E1 crosses four fixed epoch-one checkpoints with prompts v1 and v2, and base is the primary arm.** The checkpoints are base, raw_lr1e4 step125, raw_lr5e5 step125 and q5filtered_lr5e5 step121; the output is 8 validation labels `w2p_{v1,v2}_<run>`. The two configs differ only in `format_config` and `prompt_sha256` (checked on CPU). Adapters are downloaded from the pinned HF revisions (journal section 9) and hash-checked against the wave-one manifests before use. raw_lr5e5 completes the crossing proposed in W2-INSTALL-005, at about 20 GPU-minutes extra.
  - *Reason for base as primary:* the adapters were trained with v1, so v2 is an input shift for them. The adapter arms show whether the rule still acts after SFT.
  - The W2-PLAN-003 decision rule is retained, with the D-043 drift as its noise floor.
  - *Blocked:* no prompt decision is made until the scorer range policy is fixed. In val_085, v2.1 injects a canonical reference range that the supplied data lack, which conflicts with v2's rule to use the supplied range. Scoring v2 with that policy would penalize the behaviour v2 asks for.
- D-045: **LR ladder on the filtered view: 1e-4, then 1.5e-4, then 2e-4.** Setup is micro-batch 4 (or the D-042 fallback) and the v1 training prompt, with everything else as wave one: base revision, LoRA r16/alpha32/dropout 0.05, cosine schedule with 3% warmup, 2 epochs, seed 42, 2,048 max length. Configs: `configs/train/w2_filtered_{lr1e4,lr1p5e4,lr2e4}_mb4.yaml`. Both epoch checkpoints (steps 121 and 242) are generated with v1. Epoch two is reported separately, not used as an extra hidden selection opportunity.
  - *Skip rule:* 2e-4 is skipped only if the 1.5e-4 run is numerically unstable, meaning non-finite loss or grad_norm, or no manifest (`scripts/w2_epochs.py stable`). This replaces W2-PLAN-003's "stable and not worse".
  - *Reason for the skip rule:* "not worse" cannot be judged on the pod, because no scorer is approved (D-040), and judging it with the legacy scorer would make the legacy scorer the selector. The 2e-4 run costs about $0.65, and saving its outputs lets it be scored later.
  - *Why a ladder:* the only interval excluding zero points up in LR (D-039), and wave one never went above 1e-4.
- D-046: **Budgets and decoding are unchanged.** Greedy decoding; one tool call, two assistant turns, 256 new tokens per turn and 512 total; same base and tokenizer revision; NF4. *Reason:* one factor per contrast. Only the prompt (E1) or the LR (ladder) changes within a comparison, on top of the declared batch change (D-042).
- D-047: **The pod only generates. Scorer v2.1 and bounded v2 run on CPU after the pod is stopped.**
  - On the pod, `clinqa.evaluate generate` also runs the legacy scorer, which is cheap and gives a sanity signal.
  - v2.1 runs only through `scripts/score_v21_val.py`, which refuses test-like labels, an existing output directory and missing labels. `make w2-score` runs both scorers, the legacy paired comparisons and the rollout diffs.
  - *Reason:* scorer work stays off the GPU bill, scoring can be repeated when the scorer changes (outputs keep full trajectories), and the wrapper enforces D-038 mechanically.
- D-048: **Deferred, not run this round.**
  - E4 Q5 relabel: the 78 train-only proposals need the annotation audit in W2-INSTALL-005.
  - E5 pre-call sentence: it needs a separate data transformation, because training targets have empty call content, so a prompt line alone is not expected to produce reasoning.
  - Prompt v2 inside training.
  - The mb1 control (superseded by D-052: the ladder itself now runs at mb1).
  - Generation batch 8.
  - Seed-43 repeat.
  - Qwen3-8B.

  *Reason:* each changes something this round holds fixed, and the $10 budget does not cover them together.
- D-049: **This round declares no winner.** The legacy, bounded v2 and v2.1 results are reported side by side. Paired intervals come from the legacy `compare` (the only paired-bootstrap implementation), and the legacy `select`/`freeze` commands are not used. A winner can be named only after two things: v2.1's defects are fixed using validation-only evidence, and the semantic judge is calibrated and frozen (D-040). *Reason:* announcing a winner from a scorer known to pass wrong answers would bias every later decision.
- D-050: **The cross is optional, and off by default.** `CROSS=1` also generates prompt v2 on the ladder checkpoints. It is decided on budget after the main round. *Reason:* the prompt-by-LR interaction is secondary. E1 already measures the prompt on fixed weights, and v2 is an input shift for v1-trained adapters.
- D-051: **Cost ceiling and stop rule.** Spend so far is about $2.97 of the $10 budget (RTX 4090 Secure at $0.74/hr).
  - *Planning estimate from wave-one timings:* setup and smoke take about 15 minutes. E1 takes about 75 minutes (2 base passes at about 7 minutes, 6 adapter passes at about 10). Three training runs take at most about 42 minutes each at the wave-one rate; mb4 may be faster, which is unmeasured. Six epoch passes take about 60 minutes. The total is roughly 4–4.6 hours, about $3.0–3.4. CROSS adds about 1 hour, about $0.75.
  - *Ceiling:* $5 for this round, keeping total spend at or below about $8.
  - *If the ceiling is approaching:* drop CROSS first, then 2e-4, then stop after the current job. Outputs are pushed after every stage.
  - Before the pod is terminated, adapters are uploaded to private HF and `git push` must succeed. The pod is created only after the user approves the price.

- D-052: **The LR ladder trains at micro-batch 1 with accumulation 16 (user decision, mid-round).** This supersedes the training half of D-042; generation batch 4 is unchanged.
  - *Reason:* measured in W2-RUN-007. mb4 ran at about 13.9 s/step vs 10.0 s/step for wave-one mb1, with repeated allocator OOM-retry stalls on ~3.7 GB logit allocations. At mb4 the round was projected to reach the $5 ceiling (D-051).
  - *Consequence:* the ladder configs are `configs/train/w2_filtered_{lr1e4,lr1p5e4,lr2e4}_mb1.yaml`. Filtered 1e-4 at mb1 now matches wave one in batch setup, so the micro-batch confound of D-042 no longer applies to training. The generation-batch confound (batch 4 vs wave-one batch 2) still applies, and ladder generations stay matched to the E1 v1 references. `MB=mb4 make w2-round` keeps the old path. The aborted mb4 partial run is not a result.
  - E1 was not re-run. Its outputs were generated before this change, and their protocol hash is unchanged, because only the runner and configs changed.

## Status after wave two

Measured: three filtered LR runs at micro-batch 1, both epoch outputs, and eight fixed-weight prompt evaluations. These are recorded in `reports/w2_round/`. Wave-three code, relabel audit, seed variance, stronger prompted baseline and probes remain unimplemented/unmeasured. Scorer validity remains unresolved.

## Wave-three proposals (merged 2026-09-30; implementation not activated)

A four-layer review: loss/optimisation, input/output representation, data, evaluation. It is based on wave-1 and wave-2 results, rescored with scorer v2.1. The supplied narrative is now in [ROUND4_FINDINGS.md](history/ROUND4_FINDINGS.md), with review corrections; independently verified evidence is in [WAVE3_REVIEW.md](history/WAVE3_REVIEW.md); the run list is in [EXPERIMENTS_WAVE3.md](history/EXPERIMENTS_WAVE3.md). These are imported proposals. The review qualifications below take precedence; they do not silently supersede historical evaluation safeguards. No code has been changed for them yet.

**Layer 1: loss and optimisation**

- D-053: The loss stays the token-mean cross-entropy over supervised assistant tokens, normalised by `num_items_in_batch` across gradient accumulation (HF Trainer default). There is no per-type, per-example or per-segment weighting. This means per-type gradient share follows supervised tokens, not example counts (extractive about 16%, numeric 23%, uncertain 20%, tool 41%, of which the call turn is about 10%). The failures that remain are numeric reasoning and missing-information coverage, neither of which is a gradient-share problem. No verified evidence supports up-weighting call tokens for tool SFT.
- D-054: The hyperparameters are locked:
  - QLoRA r16/α32, all linear projections;
  - lr 1e-4, cosine schedule, 3% warmup;
  - 2 epochs, effective batch 16 (micro-batch 1 × accumulation 16).

  Wave-2 lr 1e-4/1.5e-4/2e-4 differed by ≤0.03 macro, within noise. The final configuration runs 3 seeds. The mb4 run was stopped on purpose and is not part of the final configuration.

**Layer 2: input/output representation**

- D-055: `system_v1` is used for both training and inference. `system_v2_reference` (v1 plus one line: "use only supplied reference ranges") is rejected:
  - Changing it at inference only flipped 0–2 items per type in paired comparisons, all CIs include 0, and it did not improve numeric.
  - It contradicts 36 train gold answers that correctly use standard ranges when the input gives none (Q11).

  Wave-2 `w2p_*` is reported as an inference-time prompt-robustness check, not a prompt improvement.
- D-056: R0 is reported twice, both inference-only:
  - R0-v1: the same prompt as SFT, for parity;
  - R0-v3: a strong prompt (at most one call, convert imperial units inside the arguments with the factors given, the call turn contains only the call, name the missing field, brief clinical context), as the prompt-engineering ceiling.
- D-057: Representation parity is verified and unchanged:
  - one `render()` path for training and inference;
  - fixed order note → Markdown table → question;
  - identical `TOOL_SCHEMAS` in every sample;
  - answer_type only in sidecars;
  - the call turn's `</tool_call><|im_end|>` is supervised and tool responses are masked;
  - no `<think>`.

  The call turn stays empty in the core configuration.
- D-058: Ablation R-VIS ("visible reasoning"), one run on the control seed:
  - numeric targets become `Working:` (lines rendered from the input-derived scorer-v2.1 key, added only when every check agrees with gold) followed by `Answer:` (gold, unchanged);
  - imperial tool call turns get one conversion line using "≈" and the gold metric values;
  - scoring reads only `Answer:`;
  - `call_logprob` moves to the actual `<tool_call>` position.

**Layer 3: data**

- D-059: The 78 train Q5 records (no weight and no height, yet gold calls BMI) are relabelled as uncertain with a two-part answer: what is documented, and what is missing. This supersedes exclusion (D-012/D-019 Q5 view) for the final configuration. Reasons:
  - After filtering, only 1 train example covers "both measurements missing".
  - Filtered models at lr ≥ 1e-4 fabricate the values in text on Q5 val items (17/42 evaluations, versus 1/28 at lr 5e-5).
  - Raw models fabricate them inside the call instead (6–7/7).

  Acceptance: Q5 fabrication (call or text) ≤1/7, and the no-call rate on grounded tool items no worse than the control.
- D-060: Numeric data is unchanged: no upsampling and no gold edits. Gold picks relative deviation in 13/15 ambiguous "most abnormal" items; only 5 of those have matching units. The weakness is reasoning, which R-VIS addresses.
- D-061: Allergy shortcut. Allergy questions that mention allergy are 36/36 correct; implicit ones are 8/18, and train has 51 explicit versus 5 implicit. The core configuration is unchanged and a slice metric is added. Ablation R-IMPL, one run on the control seed: 25 of the 51 explicit questions are rewritten into implicit form, with gold unchanged and the total count unchanged.
- D-062: Q11 is kept. When the input gives no range, the task requires the standard range; this is not label noise.
- D-068: A consolidated `reports/DATA_QUALITY.md` covers:
  - splits and statistics;
  - Q5;
  - coverage gaps and shortcuts;
  - "most abnormal" scale;
  - Q11;
  - other flags (Q6, Q7, Q13, Q14, Q16, Q17), each kept with its handling;
  - leakage.

**Layer 4: evaluation**

- D-063: Model selection. This supersedes PLAN section 6 and the `eval_core.yaml` selection rule.
  - Metric: v2.1 four-type macro on the full val set.
  - Hard gates: Q5 fabrication ≤1/7; grounded no-call rate not above the control; zero parse errors.
  - Configurations are compared by mean ± sd over 3 seeds, with ep2 fixed in advance (no epoch picking).
  - Ablation vs control: same-seed paired bootstrap; a CI that includes 0 means no difference.
  - v1 is reported in parallel but does not decide.
- D-064: Scorer v2.1 is frozen (sha256 `b15db3d0…`, `reports/scorer_v2/scorer_v2.1.sha256`) as the primary scorer. v1 is always reported as the spec-literal simple implementation. The LLM-parser plus deterministic-verifier scorer (v3) is deferred to next steps.
- D-065: Test protocol:
  - full 400 records, final configuration (3 seeds) plus R0-v1 and R0-v3 only;
  - ablations on val only;
  - v1 and v2.1 both reported;
  - disclosed: this is the second model evaluation on test, and test outputs were used to validate the scorer (v2.1 rules were adjusted on 120 test items; estimated bias ≈0.7pp);
  - the model list is frozen in `configs/final_eval.yaml` and committed before running.
- D-066: Counterfactual probes P1–P5, about 40 pairs derived from val and scored with input-derived keys, diagnostic only:
  - P1: remove a measurement;
  - P2: add a measurement;
  - P3: switch metric to imperial;
  - P4: change a value;
  - P5: remove the word "allergy" from the question.

  They run on the control seeds, R-IMPL and R0.
- D-067: Reporting:
  - the five assignment metrics under their spec names and spec denominators (tool selection over all grounded tool records; tool arguments over calls, strict and outcome), v1 and v2.1 side by side;
  - a diagnostics table: tool end-to-end, Q5 abstention and fabrication (call and text), over-call, over-refusal, self-correction, slices, probes;
  - Wilson CIs, seed sd, paired bootstrap.


### Review qualifications governing D-053 through D-068

- IDs are stable: incoming D-037..D-052 map to D-053..D-068 respectively. Existing D-001..D-052 and their historical meaning remain intact. Exact incoming wording is preserved in `history/wave3_import_2026-09-30/`; imported proposals above are subject to this section.
- **D-053**: retain unweighted CE as a control. Supervised-token share is not measured gradient influence; do not claim the residual errors are proven unrelated to weighting. Verify loss normalization with the installed Trainer and PEFT wrapper before treating it as established.
- **D-054**: 1e-4 is a conservative control choice, not evidence that all LRs are equivalent. The v2.1 epoch-two paired 2e-4 minus 1e-4 macro difference is +2.72 pp, unadjusted bootstrap interval about [+0.35, +5.53] pp. This is exploratory, single-seed and scorer-dependent; see the review. Lock epoch two prospectively without claiming it is the empirical optimum.
- **D-055/D-062**: retain v1 as the incumbent, not a proven prompt winner. The claim of only 0-2 flips per type is false for base. Missing-range policy is a task-contract choice; the gold's use of an external range alone does not prove that it is required. Evaluate this policy explicitly and distinguish general reference knowledge from patient evidence.
- **D-056**: call R0-v3 a stronger prompted baseline, not a ceiling. Its gains do not isolate the SFT effect; R0-v1 remains the parity comparator.
- **D-058**: A-VIS bundles numeric explanations and call-turn conversion text, so any gain is a bundled representation effect unless separate arms are added. Gold agreement cannot certify input-derived keys. Score the final answer for completeness AND all visible text/tool arguments for correctness; never hide erroneous Working text. Moving call_logprob changes its interpretation to a prefix-conditioned diagnostic.
- **D-059**: 78 accepted relabels and a 2,000-row view are conditional on per-record annotation audit. Filtering versus relabeling changes sample count, class mix and steps (242 versus 250 at two epochs); report a policy comparison, not an isolated label effect.
- **D-060/D-061**: claimed ambiguous-numeric/allergy counts and fabrication ratios need an exact run roster, record IDs, rule version and evidence. Repeated outputs on seven Q5 questions are not 42 independent patients. Suggested mechanisms remain hypotheses; preserve contextual meaning when rewriting questions.
- **D-063/D-064**: v2.1 stays a frozen diagnostic, not a validated automatic selector. Its verified false passes and range-policy problems are not fixed by freezing. No semantic winner is approved without evaluating those failure modes. A CI containing zero means insufficient evidence of a difference, not equivalence. Mean +/- SD across three seeds is descriptive and does not remove question-level uncertainty.
- **D-065**: any further test evaluation is reused-test exploratory evaluation. The proposed 0.7 pp contamination-bias estimate is unsubstantiated and must not be presented as a correction or bound. Freeze a prospective run list but preserve original final-test artifacts; no test-driven development.
- **D-066**: counterfactual edits must update or remove all relevant mentions and derived facts across note, table and question. Numeric results must be recomputed. Generate and inspect probes before model outputs. Include A-VIS in the probe roster if drawing conclusions about its numerical behavior.
- **D-067**: the assignment defines tool selection over tool_call examples; report full annotated and grounded-subset denominators separately. Conditional argument accuracy must be paired with call coverage/end-to-end success. Strict tolerance is a project policy, not a spec-prescribed universal +/-0.05. Define noninferiority margins and the zero-event/small-n limitations before applying gates.
- Code prerequisites are still proposals. The final-eval path currently assumes per-run selected checkpoints; fixed epoch/seed manifests and multi-seed baselines require deliberate validation changes, not bypassing provenance checks. Existing test contamination disclosures remain applicable.

### Supplemental finding qualification

ROUND4_FINDINGS is now available with reviewed corrections. D-058's imperial conversion-line component is deferred: the supplied rationale explicitly uses test-only failures, and exploratory ablations are still development. A-VIS in the reviewed plan is numeric-only. The training coverage counts support D-059 as a hypothesis, but do not replace annotation review. AUROC, token share and the leakage-bias arithmetic must not be interpreted as perfect policy behavior, gradient influence or a validated bias correction.

## Interview-review planning update (2026-10-01)

These proposals refine D-053..D-068 without reusing their IDs. EXPERIMENTS_WAVE3.md is the current queue; no new code or GPU run is activated by this update.

- D-069: Prioritize audited relabeling and strong fixed zero-/few-shot baselines before precision, rank or model-size sweeps. Seed replication follows the first control's gate.
- D-070: P1-expanded starts from 40 grounded validation BMI cases, with original answerable partners, audited edits and source-clustered reporting. Natural Q5 remains separate; synthetic point gates are not population safety guarantees.
- D-071: Answer-token confidence supports ranking diagnostics only unless an independently assessed probability mapping is built. Call-prefix calibration must identify the event/label policy and distinguish eventual tool use.
- D-072: BF16 training and serving are distinct factors. Compare a common serving precision or complete the train-by-serve crossing. Measured memory preflight precedes fit claims.
- D-073: Train-fit is a diagnostic, not an automatic rank trigger. Model-size comparisons require their own base comparator and template audit and do not isolate capacity.
- D-074: Preserve known scorer defects, test-consumption disclosures and existing Git history in interview explanations. No clean-test or estimated 0.7 pp contamination-bias claim is supported. The imported narrative is subject to INTERVIEW_REVIEW.md.

## Wave-three implementation (2026-10-01)

Implements the EXPERIMENTS_WAVE3 core queue plus the user-requested 8B SFT run. Nothing below is a measured result; the GPU round has not run. Journal: W3-PLAN-005.

- D-075: **Q5 relabel view.** `q5_relabeled` = the 1,922 unchanged filtered rows + the Q5 candidates a reviewer accepts in `configs/w3/q5_relabel_review.jsonl`, relabelled `uncertain` with the reviewed answer and no tool call. IDs, notes, tables and questions are unchanged; rejected candidates stay excluded. All 78 rows must be decided (accepted true/false and reviewer) before the view builds. The training audit is transformation-aware: it rebuilds the expected view from canonical train + the review, checks the review hash recorded at build time, and enforces 1,922 + k rows. The proposal file is preserved unmodified. F-s42 vs C-filtered-s42 is a policy comparison (count, class mix and steps change), not an isolated label effect.
- D-076: **Wave-three generation protocol.** Every matched arm generates at batch 2 under one code state with new labels (`w3_*`); wave-two batch-4 outputs are not reused as controls. The control C-filtered-s42 is `w2_filtered_lr1e4_mb1` checkpoint-242 (adapter hash `8758304b…`, checked against its manifest), regenerated. Epoch two is the prospective primary endpoint; epoch one is diagnostic. Each `run.json` now records mean/max first-turn prompt tokens, new tokens per request and seconds per request.
- D-077: **Qwen3-8B template.** Model and tokenizer `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218`, `enable_thinking: false`, passed through `chat_template_kwargs` that `load_tokenizer` attaches to the tokenizer so every render (formatting, training, inference) uses it. The 8B generation prompt ends with an empty `<think>\n\n</think>\n\n` block that the template omits from earlier assistant turns. A tool row therefore cannot supervise its call turn inside the full conversation without a train/inference mismatch. With `segmented_turns: true`, that turn becomes its own sequence ending at the call, rendered exactly as at inference; the final answer is supervised in the full sequence. Every assistant turn is supervised once; the think block is masked. Loss normalisation is unchanged: token-mean over the accumulated batch, and optimizer steps are still counted per record. The 4B single-sequence contract is untouched; the 4B mask audit and token-length report are byte-identical after the change. Local audit (`reports/w3/mask_audit_8b.json`): 1,922 rows, 0 problems, 422 tool rows as 2 sequences, max 1,618 tokens. `<tool_call>`, `<|im_end|>` and `<|endoftext|>` keep the same IDs as in 4B; the call prefix is one token. An 8B GPU smoke (`configs/train/w3_8b_smoke.yaml`) must pass before A-8B.
- D-078: **Prompted baselines.** R0-v1 (parity prompt), R0-v3 (one stronger prompt, `configs/prompts/system_v3.txt`) and R0-v3-FS4 are inference only. v3 is drafted from the task requirements and train evidence, not tuned on validation outputs, and the user must approve it before use. FS4 uses four train demonstrations, one per type in a fixed order, under a fixed rule: q5_filtered rows with no quality flag; the tool demo is calculate_bmi with metric-only inputs; each pick is the median-length record. The tool demo is executor-checked against gold. Approval is recorded separately; generation refuses unapproved demos. Measured first-turn prompts with the 4B tokenizer are 4,361–4,618 tokens. Generation has no prompt cap, and the 2,048 cap applies to training conversations only.
- D-079: **P1 probes.** From the 40 grounded validation BMI sources, edits remove every weight/height/BMI mention: whole lines, mid-line clauses, and table rows. Six sources are excluded because the question itself states the measurements. 34 probes are drafted (`configs/w3/p1_probes.json`). Automatic checks run on every probe: no parser measurement, no BMI value, no plausible-range unit value outside weight-change phrases, and the gold BMI arguments would now be flagged by Q5. 29 probes needed mid-line edits and are listed for human review. The set is frozen (`approve p1`) before any probe output exists. Original partners are scored from the same label's full validation run. Intended behaviour: no call, and say what is missing.
- D-080: **D-TRAINFIT.** 200 train IDs (50 per type, `random.Random(42)`), drawn from rows unchanged in both views and excluding the demonstrations, are frozen in `configs/w3/trainfit_ids.json`. They are generated on C-filtered-s42 and on F-s42 epoch two. This is a diagnostic only, not generalisation evidence.
- D-081: **F-s42 gate, frozen before outputs** (`configs/w3/gates.yaml`, sha256 recorded in each gate report):
  - zero parse or schema failures;
  - P1 fabrication k/n ≤ 0.05. With n = 34 this allows at most one case; the Wilson CI is reported;
  - no new unsupported calls on the 7 natural Q5 items versus the control;
  - no new valid-call failures on the P1 original partners.

  `scripts/w3_analyze.py gate` writes a suggested decision only. Seeds 43/44 run (`STAGES=seeds`) after a person records `configs/w3/gate_fs42.json` with `approved: true, decision: pass`. A failed screen triggers review, not a moved threshold.
- D-082: **A-8B, one run.** The locked filtered recipe on q5_filtered (1,922 rows): LR 1e-4, two epochs, mb1/ga16, seed 42, NF4, r16/α32/0.05 on the same seven projections. It is compared with C-filtered-s42 (same data and recipe on 4B) and with R0-8B (zero-shot 8B). It deliberately does not use the relabel view, so it does not depend on the annotation review and changes only the backbone and its template versus C-filtered-s42. Disclosure: on the wave-two filtered ladder, 2e-4 scored higher than 1e-4 under diagnostic v2.1 (+2.72 pp, 95% CI [+0.35, +5.53], unadjusted, post-inspection, single seed); 1e-4 remains the conservative locked choice and no winner is declared. A single 8B run does not isolate capacity: post-training and template also differ (D-073).
- D-083: **C10 as implemented** (`scripts/w3_analyze.py c10`, CPU, validation only).
  - C10a uses the mean log-prob of the final-answer tokens, excluding the end-of-turn token, as a correctness ranking score: AUROC, risk–coverage and per-type values, under legacy v1 and diagnostic v2.1 correctness. Truncated and misaligned turns are excluded and counted. No ECE is computed.
  - C10b uses the first-token `<tool_call>` probability against two labels: the annotated policy (gold `tool_call`) and the input-grounded policy (`tool_call` and not Q5). It reports AUROC, Brier, binned counts, executed- and attempted-call confusion matrices, and a greedy token-identity check. Near-ties under 1e-3 are reported separately. This is labelled a prefix-event diagnostic, not eventual tool use.


## Wave-three artifact approvals (2026-10-01)

- D-084: Following the user's delegated review request, Codex completed an AI input-grounding review of the 78 train-only Q5 candidates. Accept all 78; revise 11 compound-question targets and three weight-change targets, retain 64 draft targets. The derived relabel view has 2,000 rows with 378 uncertain and 422 tool rows; original inputs and canonical data stay unchanged. AI review is not human clinical adjudication; metadata now records that distinction. See WAVE3_APPROVAL_REVIEW.md and the per-row evidence.
- D-085: Approve v3 after clarifying deterministic unit conversion in argument preparation under the one-call budget. Keep the four fixed few-shot messages unchanged and remeasure their validation prompt lengths: 4,382–4,639 tokens. Approve/freeze the unchanged 34 P1 probes and retain six documented exclusions. Supersedes the draft/needs-user-review status of these artifacts in D-078/D-079 under the user's current delegated review request.
- D-086: Artifact approvals do not approve the F-s42 quality gate or seed continuation. Local synthetic classifier checks expose wrong-tool fabrication omissions and hypothetical-number false positives; full-visible-response coverage also requires review. Inspect every P1/natural-Q5 response or repair/version the classifier before gate approval; never move the threshold after outputs. No model winner, GPU outcome or human clinical review is claimed.
- D-087: **P1 classifier repaired and versioned (`p1-2`) before any wave-three output exists.** The three helper defects from D-086 are fixed:
  - any tool attempt that passes an ungrounded body measurement is now call fabrication; this covers `unit_convert` as well as `calculate_bmi`, and schema/parse errors count too;
  - every visible assistant turn is checked, not only the final answer;
  - weight/height/BMI values present in the input (e.g. a documented weight change) are not counted;
  - sentences with explicit example, conditional or threshold wording go to mandatory review instead of the fabrication count. Words that also occur in asserted claims ("indicates", "category") do not trigger this.

  Any tool call on a probe is also reported, as is "intended behaviour" (no call, no asserted value, says what is missing). `configs/w3/gates.yaml` pins `p1_classifier: p1-2`, and the gate refuses a mismatched script. The thresholds are unchanged. This does not replace D-086's required review: every one of the 34 probe responses and the 7 natural-Q5 responses is still inspected, and the gate report lists the cases needing review first. Regression tests reproduce the reviewer's synthetic fixtures (`reports/w3/gate_review_caveats.json`).
- D-088: **Qwen3.5 replaces Qwen3-8B as the model-family extension (user request, 2026-10-01).** The Qwen3-8B SFT (A-8B, D-082) is deferred: its configs stay, and the `8b` stage runs only when named. New arms, all on validation and P1, prompt v1, batch 2:
  - zero-shot R0-Q35-4B (`Qwen/Qwen3.5-4B` @ `851bf6e8…`);
  - zero-shot R0-Q35-9B (`Qwen/Qwen3.5-9B` @ `c2022362…`);
  - one SFT run, A-Q35-4B, on the locked filtered recipe: q5_filtered 1,922 rows, LR 1e-4, 2 epochs, mb1/ga16, seed 42, NF4, r16/α32/0.05.

  Integration facts verified locally:
  - The checkpoints are multimodal (`Qwen3_5ForConditionalGeneration`) with hybrid Gated DeltaNet / full-attention layers (3:1). They load text-only through `AutoModelForCausalLM`: the vision tower and MTP head are not loaded, and transformers 5.17 remaps the checkpoint prefix. Loading now fails if any parameter is missing or mismatched instead of silently initialising it.
  - The non-thinking template gives every assistant turn after the question the same empty think block as the generation prompt. The strict single-sequence contract therefore holds; no segmentation is needed.
  - Tool calls are XML (`<function=…><parameter=…>`), selected by `tool_call_format: xml`, and parameters are typed by the tool schema. All 422 training calls round-trip exactly.
  - Max training length is 1,781 tokens. `<tool_call>`, `<|im_end|>` and `<|endoftext|>` are each one token, with Qwen3.5's own IDs.

  Recipe deviation: the original seven LoRA module names exist only in the 8 full-attention layers. The target list adds the linear-attention projections `in_proj_qkv`, `in_proj_z` and `out_proj` so that every token-mixing layer is adapted: 30.5M LoRA parameters, versus 21.2M with the original names and 33.0M for Qwen3-4B. Linear attention uses the transformers torch fallback (no flash-linear-attention or causal-conv1d installed), so speed is measured, not assumed.

  Comparisons:
  - R0-Q35-4B vs R0-v1: same prompt, different base model;
  - R0-Q35-9B vs R0-Q35-4B: size within the family;
  - A-Q35-4B vs C-filtered-s42: same data and recipe, different backbone, template and call format;
  - A-Q35-4B vs R0-Q35-4B: the SFT effect.

  None of these isolates capacity (D-073). Qwen3.5 arms are generated in a later code state than the core arms. The change adds Qwen3.5 support only, and the 4B mask audit and length reports remain byte-identical.
- D-089: **Qwen3.5 round on an A100 80GB with two speed-ups (user choice, 2026-10-02).** RTX 4090 Secure had no CUDA-13 host. The `q35` arms run on an A100, while the core arms and their comparators (R0-v1, C-filtered-s42) were generated and trained on an RTX 4090. Greedy outputs can differ slightly across GPU kernels, and wall-clock costs are hardware-specific; Qwen3.5-vs-core differences therefore include a hardware component, which is disclosed with every such comparison.

  Speed-ups:
  1. `flash-linear-attention==0.5.2` is installed as the locked Linux-only extra `qwen35` (`make setup` installs it). It provides Gated DeltaNet kernels in place of the transformers torch fallback. Only fla-core and einops are added to the lock; existing pins are unchanged. `run.json`/manifest `model_load.linear_attention_kernel` records which implementation ran, and all Qwen3.5 arms share it.
  2. `PARALLEL=1` (the default) runs the host-bound zero-shot decoding lane as a second process beside the smoke/training lane on the same GPU. Outputs are unaffected, but those arms' seconds-per-request is measured under contention and must not be compared as a clean cost.

  Not changed, to keep the recipe and protocol: generation batch 2, training micro-batch 1 / accumulation 16, NF4, no packing.


## Completed wave-three core review (2026-10-02)

- D-090: Treat F-s42 epoch two as the leading missing-input policy candidate, not a verified overall clinical winner. Reviewed P1 numeric fabrication improves 25/34 to 0/34 with intact-partner success unchanged at 33/34; natural Q5 improves 4/7 to 0/7. This is a single-seed training-policy comparison. The fresh gate suggests pass, but this record does not approve the gate or launch seeds.
- D-091: Do not promote prompt v3: grounded tool-task success regresses from 29/55 to 9/55. FS4 restores 29/55 but leaves conversion coverage weak. Preserve v1 as comparator; any further prompt intervention needs a new frozen identity.
- D-092: Diagnostic v2.1 numeric 46/50 is not verified numerical accuracy. Three of six apparent C-to-F gains are unreliable under output review. Audit all finalist numeric outputs with a claim-level rubric and an ambiguous/review category before choosing a numerical winner. Preserve existing scorer versions and disclose post-inspection changes; validation cases are not new training targets.
- D-093: Continue the frozen Qwen3.5 round. Compare its filtered SFT with the filtered 4B control, assess its P1/natural-Q5 behavior, and consider a matched Q35 relabel run only after its results arrive. Do not attribute filtered-versus-relabeled differences to backbone alone. Preserve D-089's A100/kernel/concurrency caveats. Current evidence does not establish a need for RL. Evidence: WAVE3_RESULTS_REVIEW.md and W3-RESULTS-009.

## Wave-four round one and backbone choice (2026-10-02)

- D-094: **Round one results.** Evidence: `reports/w4/r1/` and the walkthrough, stage 16.
  - H1 holds: Qwen3.5 relabel epoch two P1 fabrication is 0/34, against 29/34 for Qwen3.5 filtered. Intact partners 33/34, natural Q5 7/7, grounded tool 54/55.
  - P1-RAW (2,000 raw rows, 250 steps, seed 42) is 32/34, which rules out row count and training length as the cause.
  - Parity regeneration is 250/250 identical.
  - On the same relabel data the families tie: v2.1 macro +0.25 pp [−2.15, +3.10] for Qwen3.5 over F, with identical safety metrics.
  - Gate: 4 of 5 checks pass. The P1-partner check fails on val_104, an imperial-conversion drift (174.9 cm for 68.9 in, 0.106 off; BMI, category and context correct). It is recorded as a review case and was not re-thresholded.
- D-095: **Later experiments use Qwen3-4B-Instruct-2507 (F-s42, `w3_relabel_lr1e4_s42`), not Qwen3.5 (user decision).**
  - **Reason: cost at equal performance.**
    - Qwen3.5's XML tool calls and its empty think block make every conversation about 11% longer (p50 1,525 vs 1,375 tokens; max 1,781 vs 1,614).
    - Measured throughput is lower: 1,793 vs 2,081 tokens/s.
    - Peak memory is higher: 11.3 vs 7.8 GiB.
    - Qwen3.5 adds a flash-linear-attention dependency.
    - The metrics show no material gap (D-094). H2 is unresolved: the scorers disagree, and the audit is pending.
  - **Caveats, disclosed.**
    - The speed comparison is confounded by hardware: Qwen3 ran on an RTX 4090, Qwen3.5 on an A100.
    - The +11% sequence length is the confirmed cause. Kernel efficiency of the linear-attention layers is unprofiled.
    - The choice deviates from Wave4-Q2, which said a gate pass makes Qwen3.5 the final candidate, and it was made after seeing round-one results. The bias risk is low because it is a cost choice between metric-tied candidates, not selection on a favourable score.
  - **Consequences.**
    - Stretch A (round two) uses the Qwen3 path: `w4_q3_relabel_egfr_lr1e4` on top of F-s42. This is recorded as `stretch_a_backbone: q3` in `configs/w4/gate_q35.json`, independently of the gate decision.
    - The frozen test list makes F-s42 the final model. Q35-relabel stays in it as the cross-family comparator.
    - Qwen3.5 results remain a reported replication of H1, not the deployed model.
- D-096: **Scorer v2.1 stays the reported scorer.**
  - v2.1 does not check false *extra* numeric claims. Example: F-s42 epoch two writes "by 4.2" for a true 5.2 (val_117), calls a normal creatinine elevated (val_194), and writes "by 9 bpm" for a true 1 (val_062).
  - This is reported as a known limitation (`reports/scorer_v2/VALIDATION.md` §5).
  - The v2.2 contradiction-guard prototype (`scripts/scorer_v22.py`) is kept as future work and is not adopted. It has no clean holdout, and its guards were tuned on the evaluation sets. Numeric conclusions rest on the claim-level audit.
- D-097: **F-s42 refit.** The wave-three F-s42 adapter (`w3_relabel_lr1e4_s42`, final sha256 `f556435f…`) was never uploaded to the Hub and its checkpoint is no longer available.
  - Stretch A's zero-shot arms on F and the final test need an F adapter. It is retrained as `w4_q3_refit_relabel_lr1e4_s42` (F′) with the identical recipe, data view, format, seed and steps. Only the run identity differs (`configs/train/w4_q3_refit_relabel_lr1e4_s42.yaml`).
  - F′ is evaluated on both epochs (val, P1, train-fit) under the wave-three eval config. Both epochs are compared item by item with the original F-s42 outputs (`reports/w4/refit/`). The comparison is reported, not a gate, because GPU kernels need not reproduce training bit for bit.
  - Wave-three conclusions keep citing the original F-s42 outputs. Stretch A and the frozen test list use F′, and this is disclosed with them.
  - Run as `STAGES="refit r2"` (`run_w4_round.sh`); the refit is uploaded with `UPLOAD=1`.
- D-098: **Frozen test list (option B; replaces the Qwen3.5-centred list).** Four models, frozen and committed on the pod before any test output exists.
  - The list:
    - `f_refit_test`: F′ epoch 2, the final model.
    - `c_filtered_test`: C-filtered epoch 2, the same-family control.
    - `r0_v1_test`: Qwen3 base with prompt v1.
    - `q35_relabel_test`: Q35-relabel epoch 2, the cross-family comparator.
  - Each model uses the eval config its validation outputs were generated with: `eval_w4_v1.yaml`, protocol-identical to `eval_w3_v1.yaml`, for the three Qwen3 models; `eval_w4_q35_4b.yaml` for Q35.
  - Not on the list:
    - Q35-filter and R0-Q35-4B: their questions are answered on val and P1.
    - The eGFR A-sft adapter: test has no annotated eGFR items, so Stretch A is reported on its own 19 + 19 set only.
  - Pre-specified report: natural-Q5 fabrication (n = 10, underpowered), grounded tool tasks, and the five assignment metrics under v1 and v2.1. Test reuse is disclosed (wave 1 and scorer validation).
- D-099: **No further experiments before the refactor.** Remaining work:
  - finish round 2 (Stretch A on F′), which is running;
  - one frozen test run (D-098);
  - the claim-level numeric audit (CPU).

  Seeds, broader uncertainty probes and new interventions are next steps. F′ supplies one same-recipe retrain: P1 0/34 again, but only 136/250 validation items identical and v2.1 macro 95.0 vs 96.7. Run-to-run variance of about 1–2 pp is therefore reported, and differences of that size are not interpreted. Documentation-only refactoring (stage A) may start now. Code, config and path refactoring (stages B–D) starts only after the test run, because protocol hashes cover every `src/clinqa/*.py` file and the configs.
- D-100: **Stretch A revision (A-sft2), designed after round 2.**
  - **Diagnosis of A-sft** (`reports/w4/r2/`):
    - **Age fabrication** on 16/19 probes, mostly an invented 65. This is a threshold shift, not lost discrimination: call probability on probes is 0.27–0.82, while positives-vs-probes AUROC is still 0.983. Zero-shot F′ has AUROC 1.0 with probes mostly below 0.5. Causes: 40 positives against 6 age negatives, and question templates that presuppose "age and sex" in both.
    - **KDIGO mapping not learned.** Errors are ±1 category in both directions. G3a and G5 had 3 and 1 training examples, against about 410 BMI calls for 4 WHO classes.
    - **Core regression** 54 → 51 of 55. Three of the four failures also occur in F or F′ (val_029, val_105, val_233); only val_052 (imperial drift) is new. F′ vs A-sft discordance (5) equals F vs F′ (5).
  - **Changes**, user decisions after grill-me:
    1. Prompt v1e2 = v1e plus the KDIGO category table. No zero-shot v1e2 arm, so A-sft2 vs the zero-shot arms also differs in prompt.
    2. Stage-balanced data: `data/stretch_a_v2/`, 200 rows from 160 of the 164 eligible train notes. 120 positives (all G3a/G3b/G4/G5 plus 29 G1 and 29 G2). 60 age negatives, 40 of them paired with positives and stratified by category. 20 sex negatives.
    3. Four neutral question templates added to the four original ones, round-robin within each kind.
    4. Positive answers state the KDIGO range before the category. Negative answers state what is documented, with values, and what is missing.
    5. Training from base with the F recipe (2,200 rows), as before. Epoch 2 is the primary endpoint.
  - **Evaluation.** The same frozen 19 + 19 sets. They were inspected during the diagnosis.
  - **Pre-registered criteria** (revised after the A-sft failure; the original criteria and result stay reported):
    - eGFR end-to-end ≥ 15/19 and probe fabrication ≤ 2/19, both unchanged;
    - P1 ≤ 1/34 and table-eGFR over-call ≤ 1/21, both unchanged;
    - core grounded tool ≥ 52/55, with discordance vs F′ ≤ 5 (was ≥ 53/55; relaxed by one item on the measured retrain variance: F ep1 53, F′ 54, F′ ep1 55);
    - natural Q5 ≥ 6/7 (new).

    A fail is reported, the final model stays F′, and there is no third revision.
  - **Disclosed.** Age-negative sources are 65% G1/G2, against 48% of positives, because the 20 unpaired negatives come from the remaining G1/G2 notes. Changing `data_views.py` and `training_data.py` changes the protocol hash for later runs, so views are rebuilt; existing `train_sha256` values are unchanged.
  - Run as `STAGES=r2b` (`run_w4_round.sh`).
- D-101: **Stretch A A-sft2 result, final models and the test list (2026-10-03).** Evidence: `reports/w4/r2b/`.
  - **A-sft2 epoch 2** (`w4_q3_relabel_egfr2_lr1e4_step000276`):
    - eGFR end-to-end 17/19 (was 9/19);
    - probe fabrication 0/19 (was 16/19);
    - P1 1/34: one real text fabrication, "170 cm, 100 lb" on val_038;
    - natural Q5 7/7, table-eGFR over-call 0/21.
  - **The D-100 core criteria fail.**
    - Grounded tool tasks are 51/55 against ≥ 52. Two of the four failures (val_008, val_071) are v2.1 reader false fails: "markedly reduced renal function" is bound to creatinine as low.
    - Discordance with F′ is 9 against ≤ 5; it is still 7 after correcting those two items. Numeric is 41/50 against F′'s 44/50.
    - The pre-registered result is a fail and stays a fail.
  - **Final models (user decision).**
    - **F′** (`w4_q3_refit_relabel_lr1e4_s42`, checkpoint-250) is the core final model: two tools, prompt v1, as pre-registered.
    - **A-sft2 epoch 2** is the Stretch A deliverable: three tools, prompt v1e2. Its core cost is disclosed.
  - **Test list** (D-098 plus A-sft2): `f_refit_test`, `asft2_test` (`eval_w4_q3_tools3_v1e2.yaml`, checkpoint-276; core test only, because test has no eGFR items), `c_filtered_test`, `r0_v1_test` and `q35_relabel_test`.
    - The run is on an RTX 4090. Qwen3.5 needs flash-linear-attention, which is unverified on Ada. If its import or a two-item smoke generation fails, `q35_relabel_test` is dropped before freezing and the drop is disclosed; no adapter is substituted.
    - F′ and A-sft2 differ in prompt and tool list, so their test comparison is between two deployable configurations, not a controlled ablation.
  - **H2 is reported as unresolved.** v1 gives 10 vs 3 (p = 0.092), v2.1 gives 3 vs 3, and an informal reading gives about 7 vs 1. None meets the pre-registered p < 0.05, and the backbone choice (D-095) does not depend on it. The claim-level audit is a next step.
  - **After test:** `docs/SCORER_V2_1_KNOWN_ISSUES.md` (the single scorer issue list), then the repository restructure. The restructure moves 8 superseded scripts to `scripts/legacy/` with updated references, adds config and script indexes, a concise README and `CLAUDE.md`, and archives superseded docs. The two final adapters are made public on the Hub.

## Test result and repository consolidation (2026-10-03)

- D-102: **Test result.** The frozen five-model list ran once at `7ae0034` on an RTX 4090. All hashes match and there were no reruns. Qwen3.5's run records the Transformers linear-attention implementation; use of the flash-linear-attention path is not verified. Report: `reports/w4/test/TEST_REPORT.md`.
  - Natural-Q5 fabrication (frozen p1-2 classifier, every case read): F′, A-sft2 and Q35-relabel 0/10; C-filtered 6/10; R0-v1 1/10. C's test_288 invents 50 kg and 150 cm, then says BMI cannot be calculated; v2.1 credits it as an abstention (S-17), so v2.1 shows C at 5/10.
  - Residual fabrication outside Q5: on test_020 (uncertain), F′ invents 145.5 lb and A-sft2 invents 68.0 kg while refusing BMI.
  - Grounded tool tasks: F′ 89/90, Q35-relabel 88/90, A-sft2 and C-filtered 87/90, R0-v1 48/90.
  - v2.1 macro: F′ 97.5, Q35-relabel 97.4, A-sft2 96.2, C-filtered 95.8, R0-v1 79.3. A-sft2 − F′ = −1.28 pp [−2.66, −0.06]; two of A-sft2's three tool failures are v2.1 reader false fails (S-04).

  F′ stays the core final model and A-sft2 the Stretch A model (D-101).
- D-103: **Repository consolidation** (user decisions, after the test run as D-099 requires).
  - `docs/SCORER_V2_1_KNOWN_ISSUES.md` is the single list of scorer defects (S-01 to S-16); the old audit is archived.
  - The reviewer path is `README.md` → `reports/REPORT.md` → `reports/DATA_QUALITY.md`. `docs/` keeps `ASSIGNMENT`, `DECISIONS`, `EXPERIMENT_JOURNAL`, `PROJECT_WALKTHROUGH` (follow-up notes removed), `SCORER`, `SCORER_V2_1_KNOWN_ISSUES`, `STRETCH_A` and `RUNBOOK`. Superseded documents are moved unchanged to `docs/history/` behind a one-line archive banner.
  - Eight superseded scripts move to `scripts/legacy/`, with active references (Makefile, scripts, active docs) updated. Historical prose in `DECISIONS.md` and the journal, and config comments, keep the old paths; `scripts/legacy/README.md` maps them.
  - Configs are not moved, because their paths and hashes are recorded. `configs/README.md` and `scripts/README.md` index them, and a project `CLAUDE.md` records the rules.
  - Interview preparation material is removed from the repository.
  - `_data/` is kept: it holds the provided originals that `make data` copies and checksums.
  - Every repository file is in English. The full unit suite passes on CPU (273 passed, 4 skipped for missing GPU packages), and `make data analyze views-all data-check` reproduces the committed data manifests.
  - Delivery: `w4` fast-forwards into `master`, tagged `submission-v1`. The two final adapters are made public on the Hub.
- D-104: **Corrections after an external read-only audit of the final artefacts** (2026-10-03; nothing was regenerated, rescored or reselected).
  - Natural-Q5 fabrication for C-filtered on test is **6/10**, not the 5/10 first reported from v2.1. test_288 invents values while declining, and v2.1's Q5 branch does not check fabricated values (new issue S-17). F′ invents a weight on test_020 (uncertain), so fabrication is reduced, not eliminated.
  - The A-sft2 discordance in D-101 (9; 7 after correction) is counted over all 250 core validation items (A-sft2 better on 2, F′ on 7). Tool-only discordance is 3.
  - D-100's statement that KDIGO errors were "±1 category" is wrong: sa_val_003 (eGFR 67, G2) was written as G3b. Both of A-sft2's remaining eGFR failures are the two G5 items.
  - The "1–2 pp retrain variance" wording in D-099 and D-100 rests on one same-recipe refit (114/250 outputs and 1.7 pp of macro differ). It is an observation, not a variance estimate.
  - The Qwen3.5 run records the Transformers linear-attention implementation; flash-linear-attention use is not verified. Inference `peak_vram_gb` is not reset between sequential arms and is not compared.
  - `reports/REPORT.md`, `reports/w4/test/TEST_REPORT.md`, `docs/STRETCH_A.md`, `docs/PROJECT_WALKTHROUGH.md` and `docs/SCORER_V2_1_KNOWN_ISSUES.md` carry the corrected statements. The report now ends with its key findings, as the assignment requires.
