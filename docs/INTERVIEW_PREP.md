# Interview preparation: gaps, likely questions, conclusions and evidence

Session started 2026-10-01. This is a grill-me review from the viewpoint of a strict AI model engineer interviewer, organised by the same four layers as [ROUND4_FINDINGS.md](ROUND4_FINDINGS.md). Each conclusion lists the evidence for it and says whether the evidence is **measured**, **estimated** or from the **literature**. The file is appended as the session continues.

## Review notice (2026-10-01)

This imported narrative contains hypotheses and stale assertions. [INTERVIEW_REVIEW.md](INTERVIEW_REVIEW.md) governs their interpretation and [EXPERIMENTS_WAVE3.md](EXPERIMENTS_WAVE3.md) is the active planning document. In particular: Git commits exist; P1 has 40 grounded BMI candidates rather than 55; raw token logprob is not a correctness probability; v2.1 is not a validated selector; no clean-test or quantified contamination-bias claim is supported. Statements below that an experiment was "added" express proposals, not implemented code or user-authorized GPU execution. The original is archived unchanged.

## Summary of this session's decisions

| # | Question | Decision | Evidence type | Section |
|---|---|---|---|---|
| 1 | QLoRA vs bf16 LoRA | QLoRA kept; bf16 cost to be measured (R0-bf16, A-BF16) | estimated memory budget + measured 7.8 GB peak | 1.1 |
| 2 | Micro-batch 1 | kept; mb4 is 32% slower with identical loss | measured | 1.2 |
| 3 | LoRA rank | r = 16 kept; D-TRAINFIT decides whether A-R64 runs | literature + indirect measurements | 1.3 |
| 4 | Model size | A-8B added (own template, `enable_thinking=False`, mask re-audit) | prior; estimated memory | 1.4 |
| 5 | SFT vs prompting | R0-v1, R0-v3 and **R0-v3-FS** (few-shot); accuracy, cost and controllability axes | planned measurement | 2.1 |
| 6 | Tool-result faithfulness | not tested; documented limitation | reasoning | 2.2 |
| 7 | Table serialisation | no ablation; not a bottleneck | measured (extractive 0.99–1.00, base 0.92–0.98) | 2.3 |
| 8 | Filter vs wrong gold vs relabel | raw is worst (89% fabrication); filtered at lr ≥1e-4 is 45%; relabel; **P1-expanded (~55 items) becomes the primary Q5 gate** | measured, n = 7 caveat | 3.1 |
| 9 | Commentary hallucination | method documented (MiniCheck), not run | reasoning + data flags | 3.2 |
| 10 | Data size | targeted data, not more data; no curve | measured + literature | 3.3 |
| 11 | Validating the validators | LLM adjudicators with safeguards; informal human check of about 20 items, not recorded | measured κ; informal | 4.1 |
| 12 | Calibration | **calibration diagnostics** added (C10) | stored logprobs | 4.2 |
| 13 | Statistical power | MDE table + discordant counts added (C2) | computed | 4.3 |

## Status of the gap list after this session

The gaps that remain without any plan:

1. **Git history**: no commits; `requirements.txt`; `Clinical/` vs `ClinicalQA/` divergence. This is the highest priority before any interview.
2. `REPORT.md` pending sections and a written failure taxonomy.
3. Exact per-type token shares with the real tokenizer.
4. Learning curve from the intermediate checkpoints.
5. Inference latency and throughput table (from existing `run.json`).

Everything else is either in [EXPERIMENTS_WAVE3.md](EXPERIMENTS_WAVE3.md) or documented as a deliberate limitation above.

## Hard questions to rehearse (one-line answers)

- **"What did SFT actually buy you?"** Under a validated scorer: tool use (base 0.57–0.74 → 0.87–1.00), abstention and format. Extractive was already 0.92–0.98 zero-shot. Most of v1's extractive gap was the model learning the gold style.
- **"Your biggest mistake?"** Selecting with a scorer and a subset that could not see fabrication. Wave 1 picked the model that invents BMI inputs.
- **"Why should I trust your evaluation?"** Answer keys are computed from the input; v2.0 failed a clean holdout (76.7%) and I reported it; the fixed v2.1 scored 92.6% (κ 0.72) on a second clean holdout. The limits: LLM adjudicators, a template-bound parser.
- **"What would you do with one more week?"** Recorded human labels; LLM-parser scorer v3; the tool-result tampering probe; MiniCheck groundedness; the data-size curve; 8B with a tuned lr.
- **"Is test still clean?"** No. This is the second model evaluation on test, and scorer rules were adjusted on 120 test items (≈0.7pp bias, disclosed). Ablations never touch test.

## 0. Gap inventory (status on 2026-10-01)

| Layer | Missing result / experiment / report | Severity | Cost to close | Status |
|---|---|---|---|---|
| L1 | Seed variance | high | in wave 3 (F-s42/43/44) | planned |
| L1 | Exact per-type supervised token shares (current numbers are a chars/4 estimate) | medium | CPU, about 10 min | open |
| L1 | Cost of 4-bit quantisation: QLoRA vs bf16 LoRA | high | R0-bf16 vs R0-NF4 (inference, about 20 min) + 1 bf16 LoRA run | **added to wave 3** (§1.1) |
| L1 | Model size choice (4B) never compared | medium | 1 run at 1.7B or 8B | open |
| L1 | Learning curve: only epoch-end checkpoints were evaluated (step 50/100 exist) | low | val eval per checkpoint | open |
| L1 | Train-set fit, needed to decide whether LoRA rank matters | medium | inference on 200 train records | **added to wave 3** (§1.3) |
| L2 | SFT vs a strong prompt (R0-v3) | **critical** | inference, about 15 min | planned |
| L2 | Visible-reasoning ablation (A-VIS) | medium | 1 run | planned |
| L3 | Q5 relabel fix not yet verified | **critical** | F-s42 | planned |
| L3 | Q5 evidence is only n = 7 per run | medium | P1 probe and/or a held-out train slice | partly planned (P1) |
| L3 | Data-size curve (25 / 50 / 100%) | medium | 2 runs | open |
| L3 | `reports/DATA_QUALITY.md` | medium | writing | planned |
| L4 | Scorer validated only by LLM adjudicators, with no human check | high | the user blind-labels about 30 items, about 1 h | open |
| L4 | No calibration metric (the assignment asks for "calibrated responses"); logprobs exist but no ECE/AUROC | high | CPU on existing trajectories | open |
| L4 | REPORT.md sections still pending; failure taxonomy not written | high | writing | open |
| L4 | Inference latency / throughput not reported | low | read existing `run.json` | open |
| Eng | **No git commits**; `requirements.txt` lacks the train extra; `Clinical/` and `ClinicalQA/` have diverged | **critical** (20% code-quality weight) | about 30 min | open |

## 1. Layer 1: loss and optimisation

### 1.1 "Why QLoRA when bf16 LoRA fits on 24 GB? What does 4-bit cost?"

**Conclusion.** bf16 LoRA fits, with an estimated peak of about 13 GB at micro-batch 1 with gradient checkpointing. QLoRA was chosen for headroom: a later 8B model or a larger micro-batch. NF4 is used for training and for every inference run, R0 included, so comparisons are internally fair. The accuracy cost of 4-bit has not been measured.

**Evidence.**

- *Estimated.* Memory budget, using the Qwen3-4B config: about 4.0B parameters, 36 layers, hidden 2560, FFN 9728, 8 KV heads × head_dim 128, vocabulary 151,936, tied embeddings. Check these against `config.json`.

  | Component | bf16 LoRA | QLoRA (NF4) |
  |---|---|---|
  | Base weights: embedding about 0.39B in bf16 (bnb does not quantise it) + about 3.6B other | 0.78 + 7.2 ≈ 8.0 GB | 0.78 + 1.8 ≈ 2.6 GB |
  | LoRA parameters, r = 16 on 7 projections: about 0.92M per layer × 36 ≈ 33M, fp32 | 0.13 GB | 0.13 GB |
  | LoRA gradients + Adam m and v | ~0.4 GB | ~0.4 GB |
  | Activations with gradient checkpointing (36 × 1600 × 2560 × 2 B plus one layer recomputed) | ~0.5 GB | ~0.5 GB |
  | Logits + loss: 1600 × 151,936 values, in bf16 plus an fp32 copy, plus backward | 2–3 GB | 2–3 GB |
  | CUDA context + fragmentation | 1–1.5 GB | 1–1.5 GB |
  | **Total** | **≈ 12.5–13.5 GB** | ≈ 7–8 GB |

- *Measured.* QLoRA peak 7.8 GB (`outputs/w2_filtered_lr1e4_mb1/train_log.jsonl`, `peak_vram_gb`).
  - This matches the estimate.
  - The only bf16/NF4 difference is the non-embedding weights: 7.2 − 1.8 = 5.4 GB. So bf16 should peak at about 7.8 + 5.4 ≈ 13.2 GB.
  - At mb4 the QLoRA peak was 17.75 GB, so a bf16 mb4 run would need about 23 GB: too tight for 24 GB.
- *Estimated.* Full fine-tuning: bf16 weights 8 + bf16 gradients 8 + fp32 master weights 16 + Adam m/v 32 ≈ 64 GB before activations. This rules it out on one 24 GB GPU without offload. It is the quantitative reason for choosing LoRA.
- *Estimated.* bf16 inference: 8 GB weights + KV cache. The cache is about 147 KB per token (36 layers × 2 tensors (K, V) × 8 heads × 128 dims × 2 B), which is about 4.7 GB at batch 16 × 2k tokens.

**Action (added to wave 3).**

1. R0 inference in bf16 vs NF4: isolates the effect of quantisation on accuracy.
2. One bf16 LoRA training run (seed 42, mb1), paired with F-s42: isolates the effect on training.
3. Before reporting the bf16 numbers, measure the peak on GPU (`peak_vram_gb`). The table above is an estimate.

**Interview line.** "bf16 LoRA on 4B needs about 13 GB, consistent with the measured 7.8 GB QLoRA peak plus the 5.4 GB weight difference. I chose QLoRA for headroom, and I quantify its cost with a paired bf16 control."

### 1.2 "Why micro-batch 1?"

**Conclusion.** mb1 × accumulation 16 started as a conservative default (P-008, before any GPU measurement). It is now **supported by measurement**: mb4 × accumulation 4, the same effective batch, was slower and used much more memory, with no change in optimisation.

**Evidence (measured, wave 2, `outputs/w2_filtered_lr1e4_mb{1,4}/train_log.jsonl`).**

| Metric | mb1 × 16 | mb4 × 4 |
|---|---|---|
| seconds per optimiser step | 10.2 | 13.5 (+32%) |
| peak VRAM | 7.8 GB | 17.75 GB |
| loss at steps 1 / 10 / 20 / 25 | 1.918 / 0.867 / 0.583 / 0.480 | 1.915 / 0.858 / 0.582 / 0.479 |

- **The loss trajectories are effectively identical.** This is direct evidence that the loss normalisation is correct: with `num_items_in_batch`, how micro-batches are split does not change the token-mean loss (D-053).
- *Hypothesis, not verified.* The slowdown probably comes from padding:
  - mb1 needs no padding mask, so SDPA can use its Flash kernel;
  - padded mb4 batches need an attention mask and fall back to a slower kernel;
  - and the logits of 4 sequences are held at once.

  Padding-free packing (`DataCollatorWithFlattening` with position_ids) is the right throughput fix. It is listed as a next step, not part of this round.

### 1.3 "Did you tune the LoRA rank? Would a larger r help?"

**Conclusion.** r has never changed: every run used r = 16, α = 32, dropout 0.05, all 7 linear projections (`outputs/*/run_contract.json`). Neither the literature nor our indirect evidence suggests rank is the bottleneck. The decision rests on a direct diagnostic, train-set fit, added to wave 3.

**Evidence.**

- *Literature.*
  - LoRA (Hu et al., 2021): very low ranks (1–8) already suffice for adaptation; returns saturate quickly.
  - QLoRA (Dettmers et al., 2023): when LoRA covers all linear layers, r has little effect; coverage of layers matters more. We cover all 7.
  - "LoRA Learns Less and Forgets Less" (Biderman et al., 2024): higher rank matters for knowledge-heavy continued pre-training; for instruction tuning, low-rank LoRA is close to full fine-tuning and forgets less. Our task (2k examples teaching format and behaviour) is the second kind.
- *Measured, indirect.*
  - Final train loss is about 0.23 against val loss 0.33.
  - Val loss is still falling at epoch 2 (`outputs/raw_lr1e4/train_log.jsonl`).
  - The remaining errors are reasoning and data coverage, not failures to fit.
- *Missing, direct.* Generation accuracy on the training set, by answer type:

  | Numeric accuracy on train | Interpretation | Action |
  |---|---|---|
  | ≈ val (about 0.85) | Train examples are not learned: a capacity or representation problem | r = 64 ablation; larger model; A-VIS |
  | ≈ 1.0 | Train examples are learned: a generalisation problem | Do not raise r (it risks more overfitting); fix data or format |

**Action (added to wave 3).**

- **D-TRAINFIT**: evaluate F-s42 ep2 on 200 stratified train records (inference only, `--split train`) and score them with v1 and v2.1.
- A conditional **A-R64** run (r = 64, α = 128, seed 42) only if train numeric accuracy is not clearly above val accuracy.

### 1.4 "Why Qwen3-4B? Did you compare model sizes?"

**Conclusion.** Until wave 3, only Qwen3-4B-Instruct-2507 was trained. The choice rested on priors:

- native tool-call format (`<tool_call>`, which the model saw in training);
- non-thinking variant, so the loss masks are simple;
- VRAM headroom;
- the base model already separates call/no-call well: first-token AUROC 0.966 (wave 1, measured).

Wave 3 adds **A-8B**, a one-run size comparison.

**Evidence and design points.**

- *Estimated.* QLoRA peak for Qwen3-8B, from its config: about 8.2B parameters, 36 layers, hidden 4096, FFN 12288, 8 KV heads × 128, vocabulary 151,936, untied embeddings. Verify against `config.json`.

  | Component | Estimate |
  |---|---|
  | Embedding + lm_head in bf16 (1.24B parameters) | 2.5 GB |
  | Other weights in NF4 (about 6.95B parameters) | 3.5 GB |
  | LoRA r = 16 (about 1.21M parameters per layer × 36 ≈ 44M), with gradients and Adam state | ~0.7 GB |
  | Activations with gradient checkpointing | ~0.8 GB |
  | Logits + loss | 2–3 GB |
  | Overhead | ~1.5 GB |
  | **Total** | **≈ 12–14 GB** |

  Compute is about 2× the 4B model, so expect roughly 1.5 h for 2 epochs on a 4090.
- **Representation parity risk** (an interviewer would catch this). To our knowledge there is no "Instruct-2507" release at 8B; Qwen3-8B is the hybrid thinking model. Verify on the Hub before running. Its chat template behaves differently:
  - with `enable_thinking=False`, the generation prompt gets an empty `<think>\n\n</think>\n\n` block;
  - thinking content is stripped from earlier assistant turns.

  So A-8B must use **its own template** for both training and inference, and the mask audit must be rerun:
  - the empty think block should stay in the masked prefix;
  - the call turn's `</tool_call><|im_end|>` must remain supervised;
  - the per-token logprob of the call decision must be read at the correct position.

  Reusing the 4B-2507 template, which works for the 0.6B smoke model only because it shares the vocabulary, would put the 8B model in an input format it never saw.
- **Confounds to disclose.**
  - The learning rate is not retuned for 8B (1e-4 kept).
  - There is a single seed.
  - Base models differ in post-training (hybrid vs instruct-only), not only in size.

  A difference is evidence about "this 8B model under our recipe", not a clean scaling law.
- **Decision rule.**
  - If A-8B improves numeric beyond the F-seed spread, together with D-TRAINFIT showing poor train fit for 4B: capacity is the bottleneck; recommend 8B.
  - If it does not: the bottleneck is data or format (A-VIS).

## 2. Layer 2: input/output representation

### 2.1 "Your base model is zero-shot. With a good prompt and a few examples, how much of the gap remains? Why fine-tune at all?"

**Conclusion.** The baseline is reported at three levels, all inference-only, so that "SFT beats prompting" is tested rather than assumed:

| Baseline | Prompt |
|---|---|
| R0-v1 | zero-shot, the same prompt as SFT (parity) |
| R0-v3 | zero-shot, strong instructions |
| **R0-v3-FS** | strong instructions plus 4 few-shot demonstrations |

The case for SFT is argued on three axes, not on accuracy alone.

**Design of R0-v3-FS.**

- One demonstration per answer type, taken from train and fixed. The IDs are recorded, and selection is deterministic: the first record of each type that is metric-unit, not Q5, and passes the gold self-check.
- The demonstrations are earlier conversation turns in the native chat template. The tool demonstration therefore includes the real `<tool_call>`, `<tool_response>` and final answer; the format is not described in prose.
- Prompt length is about 4 × 1.4k + the query, roughly 7k tokens, within the context window. Expected val time is about 20–30 min.

**How to argue it.**

1. **Accuracy gap**: measured, paired against F-s42.
2. **Cost**: prompt tokens per request rise about 4–5× and latency rises too. Both are measured, from `run.json` token counts and latency.
3. **Controllability**: behaviour on data gaps, such as Q5 (abstain vs call when both measurements are missing) and the implicit-allergy shortcut, is a weak point for few-shot. Four examples cannot cover every missing-field combination. This is checked with the Q5 metrics and probes P1/P5 on R0-v3-FS.

**What a bad answer would be.** "Fine-tuning is always better." If R0-v3-FS comes close to SFT, the honest conclusion is that SFT's advantage is mainly cost and robustness, not raw accuracy. That is still a valid engineering reason.

### 2.2 "How do you know the final answer uses the tool result rather than the model recomputing it?"

**Conclusion.** It is not established, and this is acknowledged as a limitation. No tool-result tampering probe is run this round, by the user's decision on 2026-10-01.

**Evidence.**

- `result_in_answer` only checks that the executed value appears in the final answer.
- In training, the tool result always equals the correct value, so the model is never forced to tell "copy the tool" apart from "recompute". A correct mental calculation and a copied result look the same to this metric.
- What we do have:
  - tool selection, argument grounding and outcome are scored separately (v2.1);
  - the call decision is near perfect, with first-token AUROC 1.0.

**Interview line.** "result_in_answer cannot separate copying from recomputing, because training tool outputs are always correct. The test I would run next is a counterfactual executor that returns a perturbed value, reporting copy / recompute / flag-inconsistency rates. Ideally the model flags an implausible result rather than blindly copying it."

### 2.3 "Why Markdown for the table? Did you try JSON or key-value?"

**Conclusion.** No serialisation ablation was run, and none is planned. Table reading is not a measured bottleneck.

**Evidence (measured, v2.1, val).**

- Extractive table lookups score 0.99–1.00 for every SFT run.
- The zero-shot base model already reaches 0.92–0.98 on extractive, so it reads the Markdown table natively.
- The remaining numeric errors are reasoning errors (wrong analyte for "most abnormal", reversed direction, arithmetic outside tolerance), not misread cells.

**Design rationale.**

- Markdown tables are common in pre-training text and are token-efficient.
- Each row carries analyte, value, unit and range aligned; pipes are escaped; an empty unit is shown as `(no unit)` (P-002).

**When to revisit.** Larger or mixed-panel tables, where JSON or one-fact-per-line formats might help.

## 3. Layer 3: data design, sampling and quality

### 3.1 "Does filtering the 78 bad records stop fabrication? Or did keeping the wrong gold make it worse?"

**Conclusion.**

- Keeping the wrong gold (raw) is by far the worst: it fabricates via a tool call.
- Filtering helps, but at the locked lr (1e-4) about 45% fabrication remains, now **in the text**. At lr 5e-5 filtering almost works.
- So filtering removes the wrong signal without teaching the right behaviour. The coverage gap is real: after filtering, 1 train example has "both measurements missing". This motivates relabelling (D-059).

**Evidence (measured, v2.1 rescoring of wave 1 and wave 2, the 7 val Q5 items; `reports/scorer_v2/val/`, `reports/w2_round/v21/`).**

| Model | Evaluations | Abstains correctly | Fabricates in call | Fabricates in text | Fabrication rate |
|---|---|---|---|---|---|
| base | 4 | 28/28 | 0 | 0 | 0% |
| raw (lr 5e-5 and 1e-4) | 8 | 6/56 | 50 | 0 | 89% |
| filtered, lr 5e-5 | 4 | 27/28 | 0 | 1 | 4% |
| filtered, lr ≥1e-4 (wave 2) | 6 | 23/42 | 1 | 18 | 45% |

- **The raw model is confident.** Its first-token probability of calling is about 0.99, and it repeats a memorised 178.5 cm.
- **The filtered model fabricates without calling.** It writes "The patient's weight is documented at 68.5 kg …", copying the opening of the 108 single-missing uncertain gold answers.
- **The two failure forms differ in risk.**
  - The raw call-fabrication produces an executed tool result. That looks authoritative and is harder to catch downstream.
  - The text fabrication is a plain hallucinated claim.

**Weakness an interviewer will point out.** All rows reuse the **same 7 items**, so the effective sample size is about 7, not 21 or 42. Per-configuration Wilson intervals are wide: 3/7 gives about [16%, 75%]. The ordering raw ≫ filtered-1e-4 > filtered-5e-5 ≈ base is consistent across every evaluation, but the rates themselves are not reliable.

**Fix (decided 2026-10-01).** Expand probe P1 into the **primary Q5 test**:

- take every grounded BMI tool item in val (about 55);
- remove both weight and height naturally (drop the sentence, or use neutral wording, with no visible editing artefacts);
- the input-derived key expects abstention.

This gives about 55 independent Q5-type items per model, classified as abstain / call-fabrication / text-fabrication. It runs on F-s42/43/44 and on the existing filtered lr 1e-4 adapters (inference only), for a filter-vs-relabel comparison on n ≈ 55. The D-059 gate becomes "fabrication ≤5% on P1-expanded, with the Wilson upper bound reported". The 7 natural Q5 items remain supporting evidence.

### 3.2 "Gold answers are LLM-generated with long clinical commentary. Does SFT teach the model to invent clinical context?"

**Conclusion.** This is not measured, and it is acknowledged as an open risk. The groundedness check below is documented as the method but not run this round, by the user's decision on 2026-10-01.

**Why it is a real risk (evidence).**

- *Measured.* The explanation text dominates the long targets. Tool and uncertain gold average about 113 and 93 supervised tokens, against about 28 for extractive. Explanation tokens make up a large share of the gradient (estimated, §L1 of ROUND4_FINDINGS).
- *Measured.* Data-quality flags show the gold contains unsupported or synthetic content:
  - Q16: 219 plausibility artefacts;
  - Q11: 46 answers cite standard ranges not in the input, correctly when the input has none;
  - Q7: 119 notes whose stated BMI disagrees with their own weight and height.
- *Measured.* Neither scorer checks the commentary. v2.1 scores only the requested items; open explanatory clauses are unscored (SCORER_V2 §3.2).
- *Observed, anecdotal.* SFT answers mirror the gold style. For example, "may reflect chronic disease-related cachexia" in the val tool answers.

**Method, if asked (documented, not run).**

1. Split each final answer into sentences. Check each sentence against the note and table with **MiniCheck** (Liyan06/MiniCheck, flan-t5-large, about 770M parameters, offline, deterministic, document-sentence → supported 0/1).
2. Report the **unsupported-statement rate** for base, SFT (F-s42), R0-v3-FS and **gold itself**.
3. Interpretation: SFT above base and close to gold means the behaviour was learned from the gold style.
4. Validate MiniCheck by hand on about 30 sentences before trusting it. Clinical paraphrase and general medical knowledge ("obesity is a risk factor for DVT") are borderline cases: they are true, but not supported by the note.

**Fixes this would justify.**

- Data: shorten the commentary or restrict it to note-grounded explanation.
- Loss: down-weight explanation segments. This is the L1 segment-weighting ablation, which would then be evidence-driven rather than speculative.

**Interview line.** "Our scorers deliberately do not grade the commentary, so hallucinated context there is unmeasured. I would quantify it with a sentence-level groundedness checker (MiniCheck) against the note, comparing base, SFT and the gold itself. If SFT copies the gold's unsupported commentary, I would shorten the targets or down-weight those segments."

### 3.3 "Is 2,000 examples enough? Would you get more data or better data?"

**Conclusion.** Better-targeted data, not more data. No data-size curve is run this round, by the user's decision on 2026-10-01; it is listed as a next step.

**Evidence.**

- *Measured.* Format is learned early. Extractive and tool behaviour are already at their final level at epoch 1 (125 optimiser steps); ep1 and ep2 differ within noise (wave 1 and wave 2 tables, v2.1).
- *Measured.* Every remaining failure traces to a specific under-covered subgroup, not to overall volume:
  - Q5 both-missing: 1 train example after filtering;
  - implicit allergy questions: 5 train examples (8/18 correct on val vs 36/36 for explicit);
  - "most abnormal" with same units and conflicting scales: 5 train examples.
- *Literature.*
  - Llama 3 filtered about 30% of its tool data for quality.
  - Hammer found an optimal irrelevance (no-call) ratio of about 10%, with call accuracy and abstention trading off.
  - APIGen and ToolACE spend their effort on verification, not volume.

**How it would be tested.**

- A data-size curve at 25 / 50 / 100%, sampled stratified by answer type, with the same recipe.
- Prediction: extractive and tool saturate early; numeric and missing-information behaviour keep improving or depend on specific subgroups.
- A-IMPL in wave 3 already tests the more specific claim: "filling a coverage gap fixes the failure".

## 4. Layer 4: evaluation

### 4.1 "Your adjudicators are LLM agents. Who validated the validators?"

**Conclusion.** The scorer was validated against LLM adjudicators with safeguards. The human check is only an **informal spot check**: the user reports reviewing about 20 items, broadly consistent with the adjudicator labels. These labels were not recorded, so no agreement statistic exists for them. This is a known limitation.

**Evidence and safeguards (measured; `reports/scorer_v2/VALIDATION.md`).**

- Blinding: adjudicators saw the input, the gold answer (marked "may be wrong") and the model output, but no scorer output and no model identity.
- Two independent rater groups per set. Agreement between groups: dev κ 0.92, holdout 1 κ 1.00, holdout 2 κ 0.93.
- Strict raters had to cite concrete values in every reason. One rater (B), who used a single templated reason on 71 items, was detected and excluded.
- The scorer was frozen by sha256 before each clean holdout.

**Residual risk.** The adjudicators and v2.1 share one written rubric. A rubric-level bias would move both together, so high agreement does not prove correctness. An example is "the final assertion wins on self-correction". Clinician labels would be needed to rule this out.

**Interview line.** "Agreement is against LLM adjudicators, with blinding, two independent groups and a detected-and-excluded lazy rater. I spot-checked about 20 items myself and they matched, but I did not record those labels, so I won't quote a number. The proper next step is a recorded, blinded human set of about 40 stratified items, reporting human-vs-adjudicator and human-vs-scorer κ."

**Cheap improvement (optional).** Re-label the about 20 items blind and record them in `reports/scorer_v2/adjudication/human_spotcheck.jsonl`. That turns the informal claim into a number.

### 4.2 "The assignment says 'calibrated responses'. You only measure whether abstention is correct. Is the model actually calibrated?"

**Conclusion.** Abstention correctness is a behaviour metric, not calibration. **Calibration diagnostics** are added (decided 2026-10-01). They are computed from logprobs already stored in every trajectory: CPU only, an evaluation-script change, no training change. They are reported as diagnostics, not as a claim that the model is calibrated.

**Data available (measured fields).** Each trajectory turn stores:

- `token_logprobs`: per generated token, greedy decoding;
- `call_logprob`: log p(the first token is `<tool_call>`).

**Diagnostics.**

1. **Answer-correctness calibration.**
   - Confidence = mean token logprob of the final answer; report the first-sentence mean as well.
   - Report AUROC (confidence separates v2.1-correct from incorrect answers), 10-bin ECE and a reliability curve, per answer type.
2. **Call-decision calibration.**
   - p = exp(`call_logprob`) against whether the item should call (Q5 items count as should-not-call).
   - Report ECE, a reliability curve and AUROC.
   - Known values (wave 1, measured): AUROC 0.966 for base, 1.000 after SFT. The raw SFT model gives p ≈ 0.99 on Q5 items that should not call: confident and wrong. This is a concrete overconfidence case.
3. **Abstention-confidence check.** On uncertain items, compare the confidence of correct abstentions with that of fabricating answers. Fabrications are expected to look as confident as correct answers.

**Limitations (always stated).**

- Token logprob is a probability of the text, not of correctness.
- Long explanatory answers dilute it with many low-information tokens.
- Greedy decoding gives one sample. Sampling-based self-consistency would be a stronger estimate, and is a next step.
- So these numbers diagnose whether the confidence signal tracks correctness. They do not certify calibration.

**Interview line.** "I separate behavioural calibration (does it abstain when information is missing? measured by the uncertainty metrics and the P1 probe) from probabilistic calibration (does its confidence track correctness?). For the second I report ECE and AUROC from stored logprobs as diagnostics. The call decision is perfectly ranked after SFT, but the raw model was about 99% confident when it invented BMI inputs. Confident and wrong is exactly the failure that matters clinically."

**Where it runs.** All existing wave-1 and wave-2 trajectories immediately, and every wave-3 model. Code item C10 in EXPERIMENTS_WAVE3.

### 4.3 "Val has 250 items (numeric 50, uncertain 38). Can you tell which of two models is better from a 4pp difference?"

**Conclusion.** No. Per-type differences below about 15–20pp cannot be resolved on val, so no decision is made from per-type deltas. A minimum-detectable-effect (MDE) table and discordant-pair counts are added to the report (decided 2026-10-01; part of C2).

**Evidence (computed).** Two-proportion test, α = 0.05 two-sided, power 0.8, p ≈ 0.85, independent samples: MDE ≈ (1.96 + 0.84) · √(2p(1−p)/n).

| Slice | n | MDE |
|---|---|---|
| extractive | 100 | ≈14pp |
| numeric | 50 | ≈20pp |
| tool | 62 | ≈18pp |
| uncertain | 38 | ≈23pp |
| macro, val | 250 | ≈9pp |
| macro, test | 400 | ≈7pp (√(250/400) ≈ 0.79× val) |

- Paired comparisons do better than this. Two models agree on most items, so the information is in the discordant items. The effective MDE depends on how many items the models disagree on. That count is reported (the existing `compare` output lists fixed/broken IDs).
- Example from wave 2, prompt v1 vs v2 on the same adapter (measured, `outputs/compare_w2p_*`): 0–2 discordant items per type, with every CI including 0. This is why "v2 looks higher" was rejected.

**How the design copes.**

- Configuration-level decisions use the mean ± sd over 3 seeds, with epoch 2 fixed in advance (D-063).
- Same-seed paired bootstrap; a CI that includes 0 means no difference.
- The key Q5 claim gets its own larger test set: P1-expanded, about 55 items per model (§3.1).
- Test (400) is for confirmation, not selection.

**Reporting rule.** Write deltas as discordant counts: "numeric +4pp = 3 fixed, 1 broken out of 50", not as bare percentages.
