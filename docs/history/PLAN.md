> **Archived 2026-10-03.** Superseded by `reports/REPORT.md and docs/DECISIONS.md`. Kept unchanged below as a historical record; relative links and some script paths (now `scripts/legacy/`) may be stale.

# Implementation Plan — revision 3

Updated 2026-09-28. Target hardware: one NVIDIA GPU with 24GB VRAM.

## 0. Revision 3 summary

Revision 3 (2026-09-28) fixes implementation interfaces after a design interview; see DECISIONS D-018 to D-032. Where this file and D-018+ disagree, D-018+ wins. Build order:

1. Git hygiene and baseline commit. 2. uv environment (Python 3.11). 3. schemas.py + formatting.py (test-first). 4. audit_masks.py, reports/formatted_examples.md, reports/mask_audit.md. 5. metrics.py with scorer fixtures (test-first). 6. infer.py state machine with a fake-model test. 7. train.py + 0.6B smoke. 8. evaluate.py + error analysis. 9. RunPod: R0 val, R1 train, val, checkpoint selection. 10. R2 (optional). 11. final-eval on test once. 12. reports/REPORT.md including "what RL/DPO would target".

Make targets: format, audit-masks, smoke, train RUN=, eval RUN= SPLIT=, core, final-eval.

## 1. Objective and scope

Build a reproducible SFT pipeline that learns when to answer from a single encounter note/table/question, when to call a deterministic tool, and when to state missing information. The deliverable is executable evidence of these behaviours, not a general medical agent.

Core uses only the supplied train split and exactly two tools. Keep all original files and splits unchanged. Do not add external training data. The default Core run preserves all 2,000 records and the original answer-type distribution. Quality filtering is a separate, explicit experiment.

**RL is not required and is out of scope for the current implementation.** Tool calling and abstention can be supervised with the supplied demonstrations; neither requires a policy-gradient objective. The first priority is trustworthy supervision and evaluation. See section 11 for the conditions under which RL could become useful.

## 2. Current status: implemented versus planned

Implemented:

- Canonical raw copies, checksums, deterministic tools and quality analysis.
- Strict JSON ingestion: duplicate keys, NaN/Infinity and floating-point overflow are rejected with line context.
- Structural validation before feature extraction: types, IDs, table shape, tool arguments and finite results.
- All source files validated before copying any split; analysis verifies live bytes against the manifest.
- Q10 numeric-token boundary fix. The former three arithmetic flags were regex false positives; train_1890 and test_306 are retained.
- Explicit raw and Q5-filtered **train-only raw-record views**, with provenance, counts, exclusion IDs and review packets. These are not chat-formatted training conversations.
- Regression tests and regenerated analysis reports. Actual verification details are in reports/REVISION_2.md.

Not yet implemented:

- Native chat formatting, exact tokenizer lengths and assistant loss masks.
- Model rollout runner, SFT training, model evaluation and measured model results.
- Chain ablation, stress diagnostics or Stretch B training.

The copied .venv contains packages but its Python executable points at a missing interpreter. Recreate the environment on the GPU host from the lock. The existing train extras are empty; do not claim the GPU stack is pinned or tested yet.

## 3. Experiment matrix and data policy

### R0: base model + tools

Use the same system policy, schemas, inference runner, decoding and quantization as each adapter comparison. All examples see both tools. Never expose answer_type, gold answers, gold arguments/results, quality flags or IDs in the model prompt.

### R1: raw SFT — required Core baseline

Train on 2,000 raw examples (800 extractive, 400 numeric, 500 tool, 300 uncertain). Preserve original single-call targets and answers. No oversampling, relabeling or external data. This is the assignment-compatible baseline, with conflicting supervision explicitly acknowledged.

### R2: Q5-filtered SFT — first optional ablation

Exclude only Q5-flagged training rows, never val/test. The current heuristic selection is 1,922 examples (800 extractive, 400 numeric, 422 tool, 300 uncertain); **do not exclude train_1890**. Retain all originals. Record the altered distribution rather than claiming it is unchanged.

The view manifest marks these as heuristic candidates, not newly human-adjudicated labels. Review the 78 full train packets before using R2 as the preferred checkpoint; record any false positives and regenerate the selection rule and manifest. Current view creation is an auditable experiment, not certification that every exclusion is correct.

R1 vs R2 uses the same seed, hyperparameters, prompt and runner. Equal epochs imply slightly different update counts; disclose this. Add matched-update controls only if time permits.

### R3: R2 + explicit imperial conversion chain — second optional ablation

This supersedes the old default-chain decision. First establish direct-call performance. Then compare sequential unit_convert -> unit_convert -> calculate_bmi traces on grounded imperial examples. Optional stages must not block Core.

Build chains from input-only measurement provenance. Gold may validate a training trace but must not choose evidence at inference. Carry actual executor outputs into later calls and regenerate the final target if rounding changes the result. Audit dependent clinical wording when the BMI changes. Skip unnecessary conversions when a consistent metric restatement is present.

### Evaluation split policy

Test labels have already been inspected for data auditing, including the Q10 correction. Describe test as audited but unused for training/model selection, not as an untouched blind test. Freeze model choices, prompts, thresholds and scorer rules before generating final test comparisons. No test-driven tuning.

Always report the original full 400-example test set. Also report the explicitly named Q5-grounded subset (390 examples; 90 tool examples) and the 10 Q5 cases separately. This subset is not a guarantee of otherwise clean labels.

## 4. Formatting contract — next work

Create formatting.py and schemas.py. Serialize note verbatim, all table rows as Markdown, and question in a separate section. Render empty units explicitly without inventing a unit. The question can supply measurements; a reference to “recorded height” without a number is not evidence.

Use the model's native chat template with a tools column and structured assistant tool_calls/tool responses. Keep internal schema conversion in one place and validate against the pinned tokenizer. Do not hand-compose a second incompatible ChatML protocol.

System policy:

- Ground patient-specific statements in the note, table and explicit question values.
- Use calculate_bmi for requested BMI calculations, unit_convert for supported conversions; simple comparisons need no tool.
- State the necessary missing/conflicting information instead of estimating it.
- Missing allergy documentation does not mean NKDA.
- Use actual tool results; keep the final answer concise.
- Distinguish general explanation from documented patient facts. Do not present absent reference ranges as if they came from the patient's table.

Keep answer_type and all gold/quality metadata in a sidecar. Do not use a regex router or gold answer_type to make tool-use decisions for the model.

Only assistant call/final-answer tokens receive loss. System, user, tool results and padding are masked. Calls with no content field must still have supervised tokens. Verify rendered text and labels for at least six full examples: extractive, numeric, uncertain, metric BMI, conversion and imperial BMI.

Measure complete conversations with the actual tokenizer, including tool schemas, results and targets. Start at 2,048 tokens; increase to 3,072 if necessary and feasible. Record p50/p95/p99/max and all overlength IDs. No silent target truncation. Disable packing initially.

Acceptance: raw view preserves every ID exactly once; six rendered examples and mask audits are saved; training and inference prompts match; no label leakage; no silent truncation.

## 5. Inference and tools

Implement a small state machine: generate -> parse -> validate -> execute -> append real tool response -> generate final answer. Both tools are available on every example. Never inject gold results during rollout.

Keep existing conversion coefficients and BMI rounding. unit_convert round(2) is an explicit data-compatibility policy, not an additional requirement in the assignment. Unit spelling normalization is allowed; semantic argument repair is not.

Strict JSON-schema checks reject bool-as-number, numeric strings, non-finite values, unknown tools/arguments and missing required fields. Report strict parsing and normalized parsing separately. First version does not guess arguments or auto-repair malformed JSON.

Budgets: R1/R2 up to one successful call and two assistant turns; R3 up to three calls and four assistant turns. Apply a total generated-token cap, detect repeated calls, and keep invalid/empty/over-budget outputs in the denominator. If self-repair is added later, use the same policy for base and adapter and report first-pass versus recovered success.

Save raw turns, parsed calls, schema status, executor output, final answer, stop reason, token counts and latency by ID. Gold joins happen during scoring, not generation. Teacher-forced diagnostics must be labeled separately from free rollout.

## 6. Model and training configuration

Default: Qwen/Qwen3-4B-Instruct-2507, a non-thinking checkpoint. Pin model/tokenizer revisions and template hashes. Qwen3.5-4B is a newer optional candidate, with its own loader/template/adapter compatibility checks; do not substitute it without a smoke test. A zero-shot win does not prove better SFT adaptation.

Single-GPU starting configuration:

```yaml
model: Qwen/Qwen3-4B-Instruct-2507
method: qlora
quantization: nf4
double_quant: true
compute_dtype: bfloat16
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
learning_rate: 0.0001
epochs: 2
microbatch: 1
gradient_accumulation: 16
scheduler: cosine
warmup_ratio: 0.03
max_grad_norm: 1.0
max_length: 2048
assistant_only_loss: true
packing: false
gradient_checkpointing: true
seed: 42
```

This is the planned project configuration, not a claim that all keys map verbatim to a particular library version. Prefer SDPA for the initial setup. Disable use_cache during training; enable it for generation. Validate bf16 support on the actual GPU. QLoRA reserves memory for conversations; it is not automatically faster than bf16 LoRA.

Pin torch/CUDA wheel, transformers, TRL, PEFT, datasets, accelerate and bitsandbytes after a successful GPU compatibility smoke test; freeze the resolved lock. Python 3.11 remains the target. No remote training has been run yet.

Before full training: run 8–16 representative examples, inspect masks/trainable parameters/finite loss, overfit a tiny batch, save and reload an adapter and verify that it is active at inference. Fail early on all-masked assistant turns or incorrect EOS/template handling.

Save each epoch. Log per-type loss and rollout metrics. Select on a predeclared macro average of four task success rates on Q5-grounded val, with tool end-to-end success as its tool component. Also publish all 250 val results. Near ties favor fewer unsupported calls, then an earlier checkpoint. Do not require every type to beat base before reporting the experiment or attempting optional work.

For runtime planning, R1 is roughly 125 optimizer steps/epoch and R2 roughly 121 at effective batch 16. Measure warm steps and estimate remaining work, plus real validation-generation time. VRAM capacity alone cannot determine runtime. Record peak memory, tokens/s, hardware, download/setup time and wall-clock training/eval time separately.

## 7. Evaluation definitions

Keep the five required metrics and add explicit error categories:

- Extractive: question-relevant fact slots (entity, number, unit, direction/negation), with normalized exact/fuzzy match for unsupported question families and token F1 only as a diagnostic. Numeric coverage alone is insufficient.
- Numeric: target quantity, arithmetic and direction must agree. Distinguish “x times the upper limit” from “x times above it.” Use complete number tokens and declared precision; no broad tolerance to conceal parser errors.
- Tool selection: correct requested tool on all gold-tool examples; missing calls count as failures. Core single-call compatibility is separately visible for chains.
- Tool arguments: schema-valid and correct arguments on the unconditional expected-call denominator; separately show conditional-on-call diagnostics.
- Uncertainty: identifies the relevant missing field, does not invent its value/status, and accurately states available evidence.
- Additional metrics: over-call, over-refusal, execution success, actual-result incorporation, unsupported arguments and tool end-to-end success.

For R3 validate all calls and dependencies, not just the last tool. Extra wrong calls cannot disappear because the final BMI is correct.

Separate gold agreement from visible-input consistency and actual execution consistency. Direct metric values require source precision; converted values use the prescribed formula. A hidden metric gold reconstructed from rounded inches may differ from a valid execution. Report precision-aware comparisons and rounding disagreements explicitly; do not use a universal +/-0.15 tolerance for unrelated units.

Q5 tool labels conflict with abstention: a system abstaining on all 10 test Q5 rows cannot exceed 90/100 original tool-selection accuracy. Preserve the original score and explain the conflict rather than rewarding fabricated parameters.

Report numerators/denominators, binomial confidence intervals and paired-by-ID bootstrap differences where useful. Unscorable predictions remain visible and count conservatively as failures in the headline, with scorable-subset diagnostics. These are abstention behaviour metrics, not calibrated probability estimates; do not claim ECE/Brier calibration without a meaningful confidence score.

Validate the scorer with 20–30 positive/negative fixtures: paraphrases, wrong numbers/units/negations, swapped analytes, extra calls, malformed JSON, blank answers, number dumping and universal refusal. Gold-copy checks test reference matching, whereas a grounding checker should reject unsupported Q5 gold. Audit a stratified sample of approximately 40 val predictions manually before freezing the evaluator.

## 8. Optional diagnostics and Stretch B

First optional experiment: R1 vs R2, focusing on unsupported calls and over-refusal as well as task accuracy. Only seven Q5 val cases exist, so report counts/cases rather than strong statistical claims.

Second: R2 vs R3, reporting grounded imperial execution and additional latency/tokens/call failures. Do not simultaneously change prompt/table layout/rank.

Optional paired diagnostics: remove/add the sole height; change a measurement and every duplicate occurrence; retain weight delta but remove current weight; permute table rows; change irrelevant vitals. Create about 20 pairs from train or fixed val development cases, outside Core training and outside the official score. A train-derived diagnostic is not independent generalization evidence.

Stretch B only after runnable Core and enough time remain. Use a separate third-tool registry/config/adapter. reference_lookup resolves unordered drug-pair aliases to actual keys, returns deterministic NOT_FOUND, and carries the reference hash/key in the tool response. NOT_FOUND does not imply no interaction. Do not certify the supplied reference's clinical validity.

Curate about 30 train-note-derived examples (lookup required, note only, missing reference), split approximately 18/6/6 with disjoint notes and preferably disjoint drug pairs. Report actual overlap and small-sample limitations. Replay selected Core train examples if needed and check Core val for regression. No vector database is needed.

## 9. File interfaces and acceptance gates

Existing commands:

```bash
make setup
make data
make analyze
make views
make test
make all
```

Direct view commands (raw is the default):

```bash
uv run python -m clinqa.data_views --variant raw
uv run python -m clinqa.data_views --variant q5_filtered
```

Each view writes data/processed/<variant>/{train.jsonl,manifest.json,q5_review.jsonl}. Evaluation keeps using canonical val/test. These commands do not fine-tune a model.

Next modules: formatting.py, schemas.py, audit_masks.py, infer.py, train.py, metrics.py, evaluate.py. Keep format/train/eval make targets explicitly unimplemented until real code exists. Planned configs: format_core.yaml, train_raw.yaml, train_grounded.yaml, train_chain.yaml, eval_core.yaml.

Future make core should run format -> mask audit -> train -> val evaluation. make final-eval should be separate and use a frozen model/config list, not rerun test during ordinary development.

Per-run artifacts: adapter/tokenizer/template, resolved config, git commit/dirty status, all data and schema hashes, exclusion IDs, dependency/GPU information, seed/precision, raw predictions, metric/error files and measured runtime. Do not commit model weights or virtualenvs.

## 10. Time budget and demonstration

The assignment suggests approximately four hours total; report actual cumulative effort including prior analysis. From this state, target a four-hour continuation only if environment and GPU speed permit:

- 0–20 min: environment and corrected audit, freeze R1/scorer contract.
- 20–55 min: formatter, schemas and tool runner.
- 55–85 min: masks, tiny overfit, adapter reload and base validation.
- 85–160 min: R1 training; complete evaluator fixtures/docs while GPU runs.
- 160–205 min: validation rollout and checkpoint selection.
- 205–240 min: frozen final comparison, artifacts and reproducibility report.

Drop second model, R2/R3 and stretch before sacrificing core evaluation. If the time cap arrives, report unfinished steps honestly. Additional time goes to R2 first, then chains or paired diagnostics, then Stretch B.

Five-minute interview narrative: task/24GB trade-offs; conflicting BMI supervision and the checker false positive; actual base/adapter tool and missing-evidence examples; controlled results with denominators; limitations and next steps. Never present planned experiments as completed.

## 11. Why no RL now; when to reconsider

SFT learns the demonstrated decision and output sequence, including assistant calls and final answers after tool results. A deterministic tool is an execution environment, not inherently an RL training requirement. LoRA/QLoRA specifies which parameters/precision are used; it is not a separate learning objective competing with SFT.

There is currently no measured SFT failure to justify RL, only 2,000 examples, contradictory gold, and a newly corrected heuristic evaluator. Turning these heuristics into rewards now risks optimizing false positives, numeric copying or universal abstention. Online rollout training also adds compute and reproducibility work on the 24GB/time budget.

Reconsider only after SFT+data fixes plateau on a documented failure, and after a reliable train-only rollout reward plus an independent val evaluator exist. A future GRPO/RLVR experiment could reward schema-valid, input-grounded, executable trajectories and correct results; reward no-call when evidence is missing; penalize unsupported/extra calls. Numeric match alone is insufficient. Use a separate comparison against SFT with matched compute, independent held-out stress cases and over-refusal monitoring. Do not use val/test to generate training rewards.

DPO is a possible offline preference-optimization extension if trustworthy chosen/rejected pairs exist; it is not mandatory RL or automatically simpler than fixing the data. Neither DPO nor RL belongs to current Core or the chosen Stretch B.

## 12. Research basis and limits

- [ArchEHR-QA 2026 QLoRA](https://arxiv.org/html/2604.14175v1): related small-model clinical QA; its external-data stage is not allowed in this Core.
- [BioTool, May 2026](https://arxiv.org/abs/2605.05758): biomedical tool-call supervision; do not import its data here.
- [CAREAgent, May 2026](https://arxiv.org/abs/2606.01094): verified trajectories and SFT followed by RL for a more complex task; this does not imply RL is required here.
- [AbstentionBench](https://arxiv.org/abs/2506.09038): motivates measuring abstention and over-refusal separately.
- [BFCL multi-step evaluation](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html): motivates dependency/execution-aware scoring.
- [SentryLine, September 2026](https://arxiv.org/abs/2609.08364): versioned guideline evidence; relevant to Stretch B provenance, not a Core RAG requirement.
- [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B), [TRL SFT](https://huggingface.co/docs/trl/en/sft_trainer), [PEFT quantization](https://huggingface.co/docs/peft/en/developer_guides/quantization).

Model scores in these sources are not predictions of this project's performance. Training/runtime/results must be measured locally on the selected GPU.
