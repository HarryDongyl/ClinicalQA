# Decisions — revision 3

Updated 2026-09-28 following the project review and user authorization to update the plan and data processing. [PLAN.md](PLAN.md) is the current implementation specification. Routine design choices below are settled recommendations, not blockers waiting for repeated confirmation. GPU compatibility and measured results remain unknown.

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

## Still to measure

GPU model/driver/bf16 support; compatible pinned training stack; actual tokenizer lengths; throughput and memory; model baseline/SFT results; manual adjudication of the Q5 train review packet. None is replaced by a predicted or fabricated result.
