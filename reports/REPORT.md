# Clinical QA Fine-Tuning — Report

Status: **pipeline implemented and CPU-verified; GPU training and model results pending.** Sections marked
_PENDING_ are filled in only from measured outputs in `outputs/`. Nothing here is a predicted score.

## 1. Setup

```bash
# Python 3.11 + uv (https://docs.astral.sh/uv/)
make setup      # uv sync --frozen --extra train; exact versions in uv.lock
make all        # data copy + checksums, analysis, views, tests (CPU)
```

Pinned stack (uv.lock): torch 2.14.0, transformers 5.17.0, peft 0.21.0, accelerate 1.15.0,
bitsandbytes 0.50.2 (Linux only). Base model `Qwen/Qwen3-4B-Instruct-2507@cdbee75f`; smoke model
`Qwen/Qwen3-0.6B@c1899de2` (shares the Qwen3 vocabulary; the 4B chat template is used for both).

## 2. How to run end to end

```bash
make audit-masks   # format + masks -> reports/formatted_examples.md, reports/mask_audit.md, reports/token_lengths.json
make smoke         # tiny overfit + adapter reload + rollout; all checks must be true
make core          # train R1, eval R0 + each R1 epoch on val, select checkpoint, paired compare
# set the selected checkpoint in configs/final_eval.yaml, then once:
make final-eval
```

RunPod runbook: [README](../README.md#running-on-runpod-24gb-gpu).

## 3. Hardware and runtime

Target: one 24GB NVIDIA GPU (RTX 4090 / L4 / A5000), QLoRA NF4.

| Stage | Expected | Measured |
|---|---|---|
| R1 training (2 epochs, 250 optimizer steps, ~2.75M tokens/epoch) | 1-2 h | _PENDING_ |
| Val rollout per model (250 examples, <=2 turns) | minutes | _PENDING_ |
| Peak VRAM | < 24 GB | _PENDING_ |

## 4. Artifacts

See the [artifact table in the README](../README.md#artifacts).

## 5. Approach and trade-offs

**Data formatting.** Native Qwen chat template with a `tools` column. User message = note verbatim, the full
table as Markdown (empty units shown as `(no unit)`), and the question. Tool examples are three assistant-side
turns: a pure `<tool_call>` (no text), a masked tool response `{"result": x}` produced by the real executor
(checked equal to gold for all 662 calls), and the gold final answer. `answer_type`, gold and quality
metadata live in sidecar files and never enter prompts. See `reports/formatted_examples.md`.

**Loss masks.** TRL's `assistant_only_loss` needs generation markers the stock Qwen3 template lacks, so labels are
built by prefix-diff rendering with exact token-boundary assertions. Only assistant content and its
`<|im_end|>` are supervised (5.5% of tokens). Mask audit: 2,000/2,000 conversations clean.

**Lengths.** Measured with the pinned tokenizer, tool schemas included: train p50 1,375 / p95 1,477 /
max 1,614 tokens; val max 1,562. `max_length` 2048, no truncation, no packing.

**Base model.** Qwen3-4B-Instruct-2507: non-thinking, native tool-calling template, fits QLoRA on 24GB with
room for 1.6k-token conversations; a single model keeps the budget on evaluation rather than model sweeps.

**Training.** Plain `transformers.Trainer` + PEFT over pre-tokenized labels (transparent, version-stable).
QLoRA NF4 + double quant, bf16 compute, LoRA r16/alpha32/dropout 0.05 on all attention and MLP projections,
LR 1e-4 cosine, 3% warmup, 2 epochs, effective batch 16, grad-norm 1.0, gradient checkpointing, seed 42.
No hyperparameter sweep (time budget); checkpoint per epoch, selected on val by a predeclared rule.

**Inference.** Greedy HF `generate`, NF4 for base and adapter alike. State machine: generate -> strict parse
-> schema validation -> real execution -> tool response -> final answer; at most one call, two assistant turns,
256 tokens per turn. Malformed or schema-invalid calls are recorded and end the rollout; nothing is repaired
or retried. Per-token log-probs and the first-token log-probability of opening `<tool_call>` are logged.

**Evaluation.** Deterministic rules (no LLM judge):
- extractive: every question-relevant number in the gold (reference ranges/thresholds excluded) must appear at
  gold precision, and high/low/normal direction must not conflict; token-F1 >= 0.5 fallback when the gold has no
  grounded number (118/800 train extractive);
- numeric: derived numbers (not present in the input) must match at gold precision, plus direction;
- tool selection / arguments on the unconditional tool denominator; imperial-derived arguments accepted within
  +/-0.05 of gold or of the exact conversion rounded to 1 dp; tool end-to-end also requires the executed result
  in the final answer and no extra calls;
- uncertainty: abstention phrase + the missing field named + no fabricated value of that field;
- over-call and over-refusal rates, Wilson 95% CIs, paired bootstrap between runs;
- subsets: full, Q5-grounded (heuristically ungrounded BMI gold removed), Q5-only.

Scorer validation: 51 fixture, regression and gold-copy tests, plus an adversarial probe on train/val (never
test) whose failure families were each fixed and pinned by a fixture (DECISIONS D-035). Gold answers score
correct on 1,995/2,000 train and 249/250 val; the 6 exceptions are gold answers that state a value absent from
the input (e.g. a height), which the fabrication rule correctly flags. The abstention rule fires on 2 of 1,912
non-uncertain train/val gold answers (over-refusal false-positive floor).

## 6. Data quality findings that shaped the design

See [data_analysis.md](data_analysis.md) and [docs/FINDINGS.md](../docs/FINDINGS.md). Most important: in 95 of 534
BMI tool examples the gold arguments are not in the input (78/7/10 train/val/test), which trains argument
hallucination and conflicts with abstention. R1 keeps them (assignment-compatible baseline); R2 removes the 78
train rows; evaluation reports the Q5-grounded subset alongside the full set. Test labels were inspected during
the data audit, so test is described as audited-but-unused, not blind.

## 7. Results

_PENDING GPU RUN._ Will contain, for R0 and the selected R1 checkpoint on val and (once) test: the five required
metrics with k/n and CIs, tool E2E, over-call, over-refusal, the Q5-grounded and Q5-only subsets, paired
differences, and the error-category tables from `outputs/<label>/<split>/error_analysis.md`.

## 8. What worked / what did not

_PENDING GPU RUN._

## 9. What RL or DPO would target

_PENDING GPU RUN_ — to be derived from the measured error categories and the call-vs-answer decision
log-probabilities in `trajectories.jsonl`, not assumed. Gates for considering RL are in PLAN section 11.

## 10. Limitations

- Text scoring is heuristic: paraphrases that drop a number or use unusual direction wording can be scored wrong,
  and a correct number with wrong reasoning can be scored right. ~40 val predictions will be audited by hand.
- The Q5 subset is itself heuristic (regex measurement extraction), not human-adjudicated.
- Imperial BMI arguments require the model to convert units mentally in R1 (single-call targets); an lb/2.2
  approximation fails the argument rule even when the BMI rounds identically.
- Tool E2E checks that the executed result appears in the answer, not that the stated BMI category is right.
- Numeric answers are accepted without restating the patient value; derived ratios/percentages must match at the
  gold's precision (one decimal coarser fails).
- Only 3 of 9 conversions appear in the data; generalisation to the others is untested.
- No hyperparameter sweep; one seed.

## 11. Next steps

R2 (Q5-filtered) vs R1; explicit unit_convert -> calculate_bmi chains (R3); LR/rank sweep; paired input
perturbation diagnostics; Stretch B (reference_lookup); DPO/RL only if gated conditions in PLAN section 11 hold.
