# Decisions — revision 4

Revision 4 updated 2026-09-30: wave-one outcome decisions (D-037 to D-040) and the wave-two round (D-041 to D-052). Revision 3 was dated 2026-09-28. [PLAN.md](PLAN.md) is the implementation specification for wave one, and [EXPERIMENT_JOURNAL.md](EXPERIMENT_JOURNAL.md) records the evidence behind each decision. Every entry gives the decision and the reason for it. Entries marked *user decision* were made by the user; where one overrides earlier advice, the entry names that advice and the confound it introduces. Planned results are never recorded as measured.
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

## Still to measure

Measured in wave one: GPU, driver and bf16 support; the pinned training stack; tokenizer lengths; mb1 throughput and memory; baseline and SFT results under the legacy scorer.

Still unknown, and not replaced by predicted or fabricated results:
- the inference prompt v1 vs v2 effect (E1);
- the filtered view at 1e-4 and above;
- the generation drift from batch 4;
- scorer validity (v2.1 repairs, semantic-judge calibration);
- manual adjudication of the Q5 train review packet and relabel proposals;
- seed variance.
