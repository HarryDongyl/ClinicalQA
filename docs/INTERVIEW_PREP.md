# Interview preparation: gaps, likely questions, conclusions and evidence

Session started 2026-10-01. This is a grill-me review from the viewpoint of a strict AI model engineer interviewer, organised by the same four layers as [ROUND4_FINDINGS.md](ROUND4_FINDINGS.md). Each conclusion lists the evidence for it and says whether the evidence is **measured**, **estimated** or from the **literature**. The file is appended as the session continues.

## Review notice (2026-10-01)

This imported narrative contains hypotheses and stale assertions. [INTERVIEW_REVIEW.md](INTERVIEW_REVIEW.md) governs their interpretation and [EXPERIMENTS_WAVE3.md](EXPERIMENTS_WAVE3.md) is the active planning document. In particular: Git commits exist; P1 has 40 grounded BMI candidates rather than 55; raw token logprob is not a correctness probability; v2.1 is not a validated selector; no clean-test or quantified contamination-bias claim is supported. Statements below that an experiment was "added" express proposals, not implemented code or user-authorized GPU execution. The original is archived unchanged.

> **Update 2026-10-02:** wave-3 results are in section 5, which corrects several claims below (P1 n = 34, v2.1 numeric false positives, latency, calibration). Where section 5 conflicts with sections 0–4, section 5 governs.

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

---

## 5. Wave-3 results update (2026-10-02)

Sources: [WAVE3_RESULTS_REVIEW_8349.md](WAVE3_RESULTS_REVIEW_8349.md) and `reports/w3/results_2026-10-02/` (`results_overview.json`, `paired_diagnostics.json`, `p1/`, `v21/`, `bounded_v2/`, `c10_*`, `trainfit/`, `training/`).

Scope: validation, P1 and train-fit only; RTX 4090; seed 42 only. No test results. Qwen3.5 was still running.

### 5.1 Corrections to sections 1–4 (stale or overstated claims)

| Earlier claim | Corrected statement | Evidence |
|---|---|---|
| P1-expanded has about 55 items | **34 approved probes**, from 40 grounded BMI candidates with 6 documented exclusions (D-085) | `reports/w3/p1_review_packet.md` |
| Gate "≤5% on P1" | 0/34 has a Wilson upper bound of **10.15%**, so it cannot establish a rate below 5% | `results_overview.json` p1.ci95 |
| v2.1 is a validated selector | v2.1 is a **diagnostic**: it passes numeric answers that are contradictory or wrong in arithmetic (val_062, val_239, val_054) | W3 review §2 |
| Few-shot costs 4–5× latency | Prompt tokens are 4,484 vs 1,294 (3.5×), but measured wall time was **not slower** (846 s vs 1,001–1,084 s). No latency advantage for SFT can be claimed from token counts | W3 review §6 |
| Calibration via token logprob | Raw mean logprob is not a probability of correctness. C10b call-prefix AUROC is 1.0 even though C fabricated on 25 P1 answers: predicting a call prefix is not predicting truthfulness. No ECE was computed | W3 review §5 |
| "No git commits" | The repository has commits (30 at clone time) | `git log` |
| Test bias of about 0.7pp | Not a quantified, supported claim. Disclose only that test is not an untouched holdout | INTERVIEW_REVIEW.md |

### 5.2 Layer 3, data: the headline result

| | C-filtered-s42 (1,922 rows) | F-s42 relabel, epoch 2 (2,000 rows) |
|---|---|---|
| P1 numeric fabrication (missing measurements) | **25/34** | **0/34** (Wilson upper bound 10.15%) |
| Actual tool calls on P1 | 0 | 0 |
| Intact partner, valid call | 33/34 | 33/34 (the same failure, val_105) |
| Grounded tool tasks | 54/55 (BMI 39/40, conversion 15/15) | 54/55 |
| Natural val Q5 fabrication | 4/7 | 0/7 |

- **Interpretation.** Filtering removed the wrong targets but did not teach the "both inputs missing" response. Explicit negative examples teach when an answer is unsupported, not just whether to call. The control fabricates **in text, with zero calls**, so a tool-routing metric alone would miss it.
- **Caveats.**
  - This is one seed.
  - C vs F is a policy package, not a pure label change: row count, class mix and 242 vs 250 optimiser steps differ.
  - P1 items are paired perturbations of val, not an external cohort.
  - val_135 still makes a qualitative unsupported statement.
  - The numeric grounding screen is not a general hallucination guarantee.
- **Interview line.** "Relabelling the 78 input-deficient records as abstentions took missing-measurement fabrication from 25/34 to 0/34 at no cost to intact-input tool success (33/34 both). Filtering alone had moved the hallucination from the tool arguments into the prose."

### 5.3 Layer 2, representation and baselines (Qwen3-4B, val)

| Arm | Grounded tool tasks /55 | P1 fabrication /34 | Intact partner valid call /34 | v2.1 macro (diagnostic) |
|---|---|---|---|---|
| R0-v1 (zero-shot) | 29 | 0 | 20 | 73.8% |
| R0-v3 (strong prompt) | **9** | 0 | **5** | 67.0% |
| R0-v3-FS4 (strong prompt + 4 shots) | 29 | 2 flagged | — | 77.6% |
| C-filtered SFT | 54 | 25 | 33 | 91.4% |
| F relabel SFT, epoch 2 | 54 | 0 | 33 | 96.7% |
| R0-8B (Qwen3-8B zero-shot) | 46 | **28** | — | 68.8% |

**Findings.**

- **The strong prompt caused a tool-compliance regression.** v3 made no call on 32/40 grounded BMI and 11/15 conversion questions. A longer, stricter prompt is not automatically better. It is a composite change, so no single sentence is isolated as the cause.
- **Few-shot recovers BMI selection only partly.** BMI selection reached 33/40, but 6 of those calls fail the argument checks, and conversion selection stays at 4/15. Its only tool demonstration is one BMI call, which plausibly explains the uneven coverage (a hypothesis).
- **"Zero fabrication" alone is misleading.** Base v1 and v3 both score 0/34 on P1 fabrication, but complete only 20/34 and 5/34 intact partners. Always pair missing-input behaviour with useful behaviour on intact inputs.
- **A larger base model is not sufficient.** R0-8B completes more tool tasks (46) but fabricates on 28/34 P1 probes, and hits the generation budget in 26/250 trajectories. The problem is grounding, not only capacity. No 8B SFT exists, so this is no verdict on 8B capacity.
- **Decision.** v1 stays the comparator; v3 and FS4 are not promoted. A future prompt experiment must separate "you must call on supported tool tasks" from "you must not invent missing arguments", and demonstrate both tool schemas.
- **Interview line (SFT vs prompting).** "Prompting does not get there. The best prompted 4B completes 29/55 grounded tool tasks; SFT completes 54/55. A stricter prompt made it worse (9/55). An 8B base model completes more tasks but invents measurements on 28/34 probes."

### 5.4 Layer 4, evaluation: numeric is where the scorer is weakest

- **Headline.** v2.1 numeric is C 40/50 → F 46/50, but **only 3 of the 6 apparent gains are credible**:
  - credible: val_028 (TIBC), val_114 (haematocrit), val_186 (bilirubin ranking);
  - ambiguous: val_054 (ambiguous criterion, plus a wrong 194% claim);
  - false positive: val_062 (self-contradictory);
  - scorer preference: val_124.
- **More false positives under v2.1.**
  - val_239: a false claim about ALT that v2.1 passes;
  - val_075: the scorer misses a real repair;
  - val_001, val_085, val_185: range-membership, sign and subtraction errors remain;
  - val_202: wrong ranking, and 19.3 written for 119.3.
- **Bounded scorer.** Numeric is C 18 pass / 4 fail / 28 review, F 21 / 2 / 27. That is not a verified ranking either.
- **Do not say** "92% numeric accuracy" or "96.7% clinical macro accuracy". These are scorer outcomes.
- **Paired diagnostics** (exploratory, unadjusted, seed 42 only), F epoch 2 vs C:
  - v2.1 macro +5.28pp, CI [+2.22, +8.67];
  - v1 macro +3.07pp, CI [+0.25, +6.23].

  Resampling neither fixes scorer bias nor measures seed variance.
- **Other type-level changes.**
  - Extractive 100 → 99: val_191 omits a timing detail.
  - Uncertain 35 → 37 of 38.
- **Calibration diagnostics (C10).**
  - Final-answer mean-logprob AUROC is about 0.750 for C and 0.640 for F. The error sets differ, so this is not "calibration got worse".
  - Call-prefix AUROC is 1.0 for both.
- **Next measurement.** A claim-level audit of all 50 numeric answers for the finalists, with a frozen rubric: reference range taken from the input, analyte selection, direction, arithmetic, units, ranking criterion, extra false claims, plus an ambiguous/disputed category. Do not feed validation-specific repairs into training labels.
- **Interview line.** "My own scorer over-credits numeric answers. On inspection only 3 of 6 apparent numeric gains were real. So I report the missing-input result, which is robust, and treat numeric as unresolved pending a claim-level audit."

### 5.5 Layer 1, training (measured)

| | C-filtered | F relabel |
|---|---|---|
| Training time | 42.7 min | 43.9 min |
| Median s/step | 10.21 | 10.13 |
| Peak allocated | 7.81 GiB | 7.81 GiB |
| Teacher-forced val loss, epoch 2 | 0.3414 | 0.3470 |
| Train-fit numeric (v1, 200 train items) | 32/50 | 34/50 |

- No divergence over two epochs.
- **Val loss is slightly worse for F while behaviour improves.** Teacher-forced loss is unsuitable for checkpoint selection here: the canonical val targets include unsupported Q5 calls, and the objective differs from the behaviour being measured. This is a good interview point.
- **Train-fit diagnostic (D-TRAINFIT).** Numeric on train under v1 is only 32–34/50, i.e. not near 1.0. But it is scorer-dependent (v1 under-credits numeric), so it **cannot establish a capacity ceiling**. A-R64 stays unjustified until the claim-level audit says whether train numeric errors are real.

### 5.6 Qwen3.5 round (running; D-088, D-089)

- **Arms.** R0-Q35-4B, R0-Q35-9B (zero-shot), and A-Q35-4B SFT on the **filtered** recipe. Comparisons: A-Q35 vs C-filtered (same data), A-Q35 vs R0-Q35 (SFT effect). Do not compare A-Q35 filtered against F relabel, which would confound family with data policy.
- **Integration facts.**
  - Hybrid Gated DeltaNet / full attention (3:1).
  - XML tool-call format.
  - Empty think block in every assistant turn.
  - Max length 1,781 tokens.
  - 30.5M LoRA parameters after adding `in_proj_qkv/in_proj_z/out_proj`. The original seven module names would adapt only the 8 full-attention layers.
- **Why it is slower (analysis, 2026-10-02).** The vocabulary grows from 151,936 to about 248k. That adds only about 3 points to the lm_head share of per-token compute (about 5% → 8%), plus about 1.6 GB of logit memory. The larger factors:
  - the linear-attention kernels (torch fallback, later fla);
  - longer sequences (XML calls plus the think block);
  - A100 with `PARALLEL=1` contention, which makes timings non-comparable by construction.
- **Value.** None of the Q3.5 comparisons isolates capacity (D-073). The arm most likely to be informative is "does a different backbone also fabricate under the filtered policy?". Cross-family numeric comparison is blocked by the numeric scorer weakness (§5.4).
- **Status (2026-10-02).** A-Q35-4B SFT had already started and was close to finishing, so it is completed as frozen. No expansion.
- **Recommendation.** Finish the frozen arms. Do not expand (no Q3.5 relabel, seeds or 9B SFT) unless A-Q35 filtered shows a qualitatively different missing-input behaviour. Spend GPU on F-s43/44 first.

### 5.7 Updated priority list (as of 2026-10-02)

1. Record the reviewed F-s42 engineering gate, then F-s43/44 with the same recipe; report per-seed P1 and partner outcomes.
2. Claim-level numeric audit (all 50) for the finalists.
3. Finish Qwen3.5, then read every P1 and natural-Q5 response.
4. Only then decide on a bounded numeric intervention or a prompt/tool-demonstration experiment. RL is not required by current evidence.
5. REPORT.md: lead with the missing-input result and its caveats; report numeric as unresolved.

### 5.8 "C vs F changes row count, class mix and steps, with one seed. How do you know it is the relabel?"

**Conclusion (decided 2026-10-02).** Use the existing wave-1 **raw_lr1e4 step 250** adapter as a label-only control. It has the same 2,000 rows, 250 optimiser steps, lr 1e-4, 2 epochs, mb1×16 and seed 42 as F-s42. The only difference is that its 78 Q5 records carry the wrong call targets instead of abstention targets. It runs on the 34 frozen P1 probes and the 7 natural Q5 items. Inference only, about 15 min; the user has kept the adapter.

**What the three-way comparison answers.**

| Arm | 78 Q5 records | Rows / steps | Isolates |
|---|---|---|---|
| raw_lr1e4 (wave 1) | wrong call targets | 2,000 / 250 | effect of keeping wrong labels |
| C-filtered-s42 | removed | 1,922 / 242 | effect of removing them |
| F-s42 | relabelled as abstain | 2,000 / 250 | effect of correct negative labels |

raw vs F is the clean "labels only" contrast; C vs F is the policy package.

**Known prior evidence.** raw lr1e-4 fabricated on 6/7 natural Q5 items via a tool call (v2.1 rescoring, wave 1). The expected P1 failure mode for raw is therefore call fabrication, while C's is text fabrication.

**Caveats.**

- raw was trained in the wave-1 code state. Run it with the wave-3 inference protocol and the same `p1-2` classifier, and record and compare the source hashes and inference protocol objects with F before claiming comparability.
- Still one seed. F-s43/44 remain necessary.

**Interview line.** "To separate the label effect from the extra rows and steps, I compare against a wave-1 run with identical rows, steps and seed in which those 78 records still carry the wrong labels. Only the labels differ."

### 5.9 "Your numeric scorer is unreliable. How will you establish numeric accuracy?"

**Decision (2026-10-02).** A claim-level numeric audit.

| Item | Design |
|---|---|
| Scope | All 50 val numeric answers for C-filtered-s42 and F-s42 epoch 2 (100 answers); A-Q35-4B's 50 once available |
| Rubric (frozen before labelling) | Per answer: reference range taken from the input; analyte selection; direction; arithmetic; units; ranking criterion as asked; extra false claims. Each scored correct / wrong / n.a. Under-specified questions or disputed gold are marked **disputed** instead of being forced into pass/fail |
| Labelling | Two independent LLM adjudicators, blinded to model identity and scorer output (existing protocol). **The user adjudicates their disagreements** (expected 10–20 items, about 30 min) |
| Output | Correct / wrong / disputed counts per model; distribution of error types; agreement between the two adjudicators |
| Leakage control | Audit labels are used for evaluation only: never as training labels and not to tune v2.1. Any later scorer revision uses independent synthetic regression fixtures and is labelled post-inspection |

**Interview line.** "I don't trust a regex scorer on multi-step numeric answers. So I audit every numeric answer claim by claim with a frozen rubric, keep an explicit disputed category, and personally adjudicate the cases the two blinded raters disagree on."

### 5.10 P1 probe results, full table (val, 34 pairs; `reports/w3/results_2026-10-02/p1/summary.md`)

Each pair is one probe input with both weight and height removed (correct behaviour: abstain) and its intact partner (correct behaviour: call the tool).

| Arm | Fabrication | in call | in text | No call + states missing | Intact partner valid call |
|---|---|---|---|---|---|
| R0-v1 | 0/34 | 0 | 0 | 34/34 | 20/34 |
| R0-v3 | 0/34 | 0 | 0 | 34/34 | 5/34 |
| R0-v3-FS4 | 2/34 | 0 | 2 | 31/34 | 21/34 |
| C-filtered SFT | **25/34** | 0 | **25** | 14/34 | 33/34 |
| F relabel SFT, epoch 2 | **0/34** | 0 | 0 | **34/34** | **33/34** |
| R0-8B | **28/34** | **28** | 6 | 0/34 | 25/34 |

F epoch 1 has no P1 run. Text fabrication flags are regex candidates and were inspected individually in the W3 review.

**Reading.**

- Only F is correct in both columns.
- Base "zero fabrication" comes from under-calling: it rarely calls even on intact inputs.
- C hallucinates only in prose (0 calls), so a call-only metric would miss all 25 cases. This confirms the wave-2 observation, now on n = 34 instead of 7.
- 8B calls eagerly and invents the arguments.
- F's 0/34 has a Wilson upper bound of 10.15%, from one seed. val_135 shows a qualitative unsupported statement that is outside the numeric screen.

**Interview line.** "On 34 paired probes, only the relabelled SFT model both abstains when measurements are removed (34/34) and still calls the tool when they're present (33/34). The filtered model invents measurements in prose on 25/34 without ever calling a tool, and the 8B base invents them as tool arguments on 28/34. The base models' zero fabrication is because they mostly don't call at all."

### 5.11 Test-set use for wave 3 (decided 2026-10-02)

- **Option B.** One confirmatory test run on a frozen, committed model list (F-s42/43/44, C-filtered-s42, R0-v1), reporting only pre-specified metrics:
  - natural Q5 fabrication on test (n = 10);
  - grounded tool-task completion;
  - the five assignment metrics under v1 and v2.1 (diagnostic).
- **No test-side probes** (user decision). The confirmation of the core missing-input claim on test therefore rests on the 10 natural Q5 items and is underpowered. Say so.
- Numeric on test: v1/v2.1 diagnostic scores only; no audited accuracy (the audit is val-only).
- Disclosure: test was used once in wave 1 and to validate the scorer, and val has been used adaptively. This run is a pre-specified confirmation, not a pristine holdout estimate.

### 5.12 All evaluation dimensions, starting from the assignment (status 2026-10-02)

Values are val, seed 42, F = F-s42 epoch 2, C = C-filtered-s42. Sources: `reports/w3/results_2026-10-02/{v21,p1,c10_*,training}`, the W3 review.

**A. What the assignment says the model must do (Overview)**

| # | Required behaviour | Our measurement | F | C |
|---|---|---|---|---|
| A1 | Extract facts from the note or table | extractive accuracy (B1) | 99/100 | 100/100 |
| A2 | Simple numeric reasoning | numeric accuracy (B2) | 46/50 v2.1 (**unverified**; audit pending) | 40/50 |
| A3 | Tool use: when to call, valid arguments | B3, B4, C1–C4 | see below | see below |
| A4 | Calibrated responses: state uncertainty, no hallucination | B5 + D1–D3 + E4 | see below | see below |

**B. Minimum evaluation metrics (assignment section 4), under spec names**

| # | Metric | Implementation | F (v2.1 / v1) | C (v2.1 / v1) | Limitation |
|---|---|---|---|---|---|
| B1 | Extractive accuracy | v2.1: input-derived keys; v1: gold-word match | 99 / 98 of 100 | 100 / 99 | v1 under-credits style (about 58/61 base false negatives in wave 1) |
| B2 | Numeric reasoning accuracy | v2.1 typed tolerance checks; v1 | 46 / 29 of 50 | 40 / 25 | v2.1 false positives (only 3 of 6 apparent gains are credible); v1 false negatives |
| B3 | Tool selection accuracy | correct tool name / grounded tool records | 55/55 | 55/55 | — |
| B4 | Tool argument accuracy | strict ±0.05 (v1 rule) / outcome band | 54/55 / 54/55 | 54/55 / 54/55 | strict penalises clinically negligible imperial rounding |
| B5 | Uncertainty detection rate | abstention phrase + missing field named + no fabricated value | 37/38 | 35/38 | regex-bound; "safe" conclusions need an extra rule |

**C. The four tool-call questions the assignment lists**

| # | Question | Measurement | F | C |
|---|---|---|---|---|
| C1 | Did it decide to call (vs answer directly)? | relevance over all 62 tool records, including Q5 = should not call | 62/62 | 62/62 |
| C2 | Correct tool? | = B3 | 55/55 | 55/55 |
| C3 | Valid arguments that match the note (including imperial → metric)? | schema-valid + grounded + = B4; parse errors | 54/55; parse errors 0 | 54/55; 0 |
| C4 | Final answer incorporates the result, with brief clinical context? | result reported in the answer (part of tool E2E: 54/55 for both) | incorporation measured; **clinical context not scored** | same |

**D. Our additions for "no hallucination"**

| # | Dimension | Why it was added | F | C |
|---|---|---|---|---|
| D1 | Natural Q5 abstention (gold calls BMI without inputs) | label-noise records; v1 could not see them | 7/7 | 4/7 |
| D2 | P1 fabrication, in call / in text (34 probes) | n = 7 too small; text fabrication is invisible to call metrics | 0/34 (0 / 0) | 25/34 (0 / 25) |
| D3 | P1 intact-partner valid call | prevents "zero fabrication by never calling" | 33/34 | 33/34 |
| D4 | Over-call (calls on non-tool items) | abstention ↔ call trade-off | 0/188 | 1/188 |
| D5 | Over-refusal (abstains on answerable items) | the same trade-off | 0/150 | 0/150 |
| D6 | Unsupported clinical commentary | gold commentary may be learned | **not measured** (MiniCheck method documented, §3.2) | — |
| D7 | Tool-result faithfulness (copy vs recompute) | `result_in_answer` cannot tell them apart | **not measured** (§2.2) | — |

**E. Diagnostics and slices**

| # | Dimension | Status |
|---|---|---|
| E1 | Implicit vs explicit allergy question slice | baseline 8/18 vs 36/36 (wave 2); A-IMPL ablation proposed |
| E2 | Same-unit "most abnormal" questions | v2.1 accepts both scales; covered by the numeric audit |
| E3 | Imperial vs metric tool records | outcome vs strict argument scores |
| E4 | Calibration diagnostics (C10) | answer mean-logprob AUROC: F 0.640, C 0.750 (different error sets; not a calibration claim); call-prefix AUROC 1.0 for both; no ECE |
| E5 | Self-correction rate | v2.1 flags `self_corrected` |
| E6 | Train-fit (200 train records) | v1 numeric F 34/50, C 32/50; scorer-dependent, no capacity conclusion |
| E7 | Teacher-forced val loss per type | F 0.3470 vs C 0.3414: worse loss, better behaviour, so unsuitable for selection |

**F. Statistical validity**

| # | Item | Status |
|---|---|---|
| F1 | Wilson 95% CI per metric | reported (e.g. 0/34 has an upper bound of 10.15%) |
| F2 | Paired bootstrap, F vs C | v2.1 +5.28pp [2.22, 8.67]; v1 +3.07pp [0.25, 6.23]; exploratory, unadjusted |
| F3 | Minimum detectable effect / discordant counts | per-type MDE about 14–23pp on val (§4.3) |
| F4 | Seed variance | **missing** (only s42); F-s43/44 pending |
| F5 | Label-only control (raw vs F) | P1-RAW proposed (§5.8) |

**G. Evaluator validity**

| # | Item | Status |
|---|---|---|
| G1 | v2.1 vs blind adjudicators | 92.6%, κ 0.72 on a clean holdout (v1: 65.7%) |
| G2 | Inter-adjudicator agreement | κ 0.92 / 1.00 / 0.93 |
| G3 | Human check | informal, about 20 items, not recorded |
| G4 | Numeric claim-level audit | decided, pending (§5.9) |

**H. Cost (assignment: hardware assumptions and runtime)**

| # | Item | F | C | Note |
|---|---|---|---|---|
| H1 | Training time | 43.9 min | 42.7 min | RTX 4090 |
| H2 | Peak VRAM | 7.81 GiB | 7.81 GiB | |
| H3 | Val generation time (250) | 1,000.9 s | 1,083.6 s | FS4 846 s, so no latency advantage claimed for SFT |
| H4 | First-turn prompt tokens | 1,294 | 1,294 | FS4 4,484 |

**I. Project-level grading criteria (assignment "Evaluation Criteria"; not model metrics)**

| Weight | Criterion | Main evidence in the repository |
|---|---|---|
| 10% | Data formatting | native template, assistant-only masks audited, Markdown tables, the 4 answer types, Q5 relabel |
| 40% | Fine-tuning setup | QLoRA recipe with measured rationale, locked hyperparameters, provenance, gates |
| 10% | Evaluation | B + C + D + statistics + validated scorer, with an honest limitations list |
| 20% | Code quality | tests, pinned dependencies, git history (verify the README matches the current tree) |
| 20% | Stretch / beyond | scorer research and validation, probes, label-only control, calibration diagnostics, model-family extension (Qwen3.5) |

**Coverage gaps against the assignment**

- C4 "brief clinical context" is not scored (D6).
- A4 "calibrated" is only behavioural, plus diagnostic AUROC (E4).
- A2 numeric accuracy is unverified (G4).
- F4 has no seed variance.

### 5.13 Plan for the four coverage gaps (decided 2026-10-02)

| Gap | Can gold help? | Plan | Compute |
|---|---|---|---|
| **C4: "brief clinical context" not scored** | **Yes, as the convention.** In train, gold states a BMI category on 410/410 BMI tool answers (407 match WHO cut-offs) and a status on 89/90 conversion answers; in val, 47/47 (46 match) and 15/15 | **Clinical-context check (CC).** BMI: the category implied by the *executed* result under WHO cut-offs (<18.5, <25, <30, ≥30) must be stated, with no conflicting category. Conversion: the status of the converted analyte must be stated and agree with the input range. The truth comes from the tool result and input, not from gold text (about 1% of gold categories are wrong) | CPU; **done**, results in §5.14 |
| **A2: numeric unverified** | **Partly.** v2.1 keys record `gold_agrees`: where the input-derived key and gold agree, use it as a high-confidence reference; where they disagree, pre-mark the item disputed. Gold cannot catch extra false claims (e.g. val_239), so the claim-level audit is still needed | Stratify the 50 numeric val items by key-vs-gold agreement and coverage before the audit (§5.9) | CPU; **stratification done** (§5.14); audit pending |
| **A4: calibration only behavioural** | **No.** Gold has no probabilities | Add **ECE** for the call-prefix probability (C10b: p(first token = `<tool_call>`) is a genuine model probability), next to the existing Brier and bins. Per D-071, answer-token log-prob stays ranking-only (AUROC, risk-coverage): no ECE without an independently assessed probability mapping. Sampling-based self-consistency confidence is a GPU next step | CPU; **done** (§5.14) |
| **F4: one seed** | **No** | P1-RAW label-only control (about 15 min), then F-s43/44 (each about 44 min training + about 20 min val/P1) | GPU; pending |

### 5.14 Results of the CPU-only gap work (2026-10-02)

All three are post-hoc diagnostics on existing val outputs. No GPU, no test data, no change to the frozen scorer (`src/clinqa/scorer_v2.py`). Tests: `tests/test_gap_diagnostics.py`.

**C4 clinical-context check** (`scripts/clinical_context.py`; output `reports/w3/results_2026-10-02/clinical_context/`).

- Rules were developed on **train gold only**. Rule self-check on train gold: 418/422 (99.1%). Of the 4 failures, 3 are gold errors: BMI 18.7–18.9 called "underweight", which is normal under WHO.
- Two patterns ("Despite falling within …", "near the upper limit of normal") were added after they appeared in a train gold failure. They were also seen in val outputs, so the final version is post-inspection.
- Val gold: 52/55. The misses are 1 gold BMI error and 2 conversion phrasings ("seemingly normal", "standard laboratory reference range") that the rule does not parse. These are known rule false negatives.

| Arm | BMI | Conversion | All | No executed call |
|---|---|---|---|---|
| R0-v1 | 19/24 | 4/7 | 23/31 | 24 |
| R0-v3 | 1/8 | 2/4 | 3/12 | 43 |
| R0-v3-FS4 | 27/32 | 3/4 | 30/36 | 19 |
| C-filtered | 39/40 | 15/15 | 54/55 | 0 |
| F relabel, epoch 1 | 39/40 | 15/15 | 54/55 | 0 |
| F relabel, epoch 2 | 39/40 | 15/15 | 54/55 | 0 |
| R0-8B | 26/31 | 9/15 | 35/46 | 9 |

- The only SFT failure is val_125: BMI 18.7 called "underweight". The val gold makes the same mistake, so this is plausibly learned from gold; train gold has 3 such cases.
- Base models fail C4 mainly by not calling at all, and when they do call they omit or misstate the category more often.

**Interview line.** "The assignment asks for brief clinical context. I scored it as 'states the WHO category of the executed BMI, or the correct status of the converted value'. SFT gets 54/55, and its one miss copies a gold labelling error at the 18.5 cut-off."

**A4 calibration: call-prefix ECE added to C10b** (`reports/w3/results_2026-10-02/c10_v21_ece/`; the earlier `c10_v21/` is preserved).

| Arm | C10a AUROC (answer, ranking only) | C10b AUROC (grounded) | Brier | **ECE** |
|---|---|---|---|---|
| R0-v1 | 0.585 | 0.964 | 0.098 | 0.104 |
| R0-v3 | 0.500 | 0.955 | 0.183 | 0.188 |
| R0-v3-FS4 | 0.670 | 0.987 | 0.074 | 0.074 |
| C-filtered | 0.750 | 1.000 | 0.0007 | 0.0025 |
| F epoch 1 | 0.748 | 1.000 | 0.0032 | 0.0047 |
| F epoch 2 | 0.640 | 1.000 | 0.0004 | 0.0018 |
| R0-8B | 0.666 | 0.964 | 0.128 | 0.145 |

- SFT makes the call-prefix probability nearly perfectly calibrated on val (ECE about 0.002 vs 0.07–0.19 for base).
- **But C has ECE 0.0025 and still fabricates on 25/34 P1 probes.** Its fabrication is in prose with no call, so call calibration says nothing about answer truthfulness.
- Answer-token log-prob stays ranking-only (D-071): no ECE.
- P1 probe trajectories are not part of C10b.

**Interview line.** "SFT's call decision is well calibrated (ECE 0.002), but that is the wrong quantity for hallucination: the filtered model is just as well calibrated on calls while inventing measurements in text."

**A2 numeric audit preparation** (`scripts/numeric_audit_prep.py`; output `reports/w3/numeric_audit_prep/`).

| Stratum (from input-derived keys, no predictions) | n | v2.1 pass, C | v2.1 pass, F |
|---|---|---|---|
| key agrees with gold | 34 | 26 | 31 |
| key disputes gold | 0 | — | — |
| open / partial coverage | 16 | 14 | 15 |

- No key–gold disputes remain on val. That is expected, because v2.1 was tuned until val gold passed 150/150.
- **The known v2.1 false positives are mostly in the "agrees" stratum** (val_239, val_054, val_124, val_075, val_202, val_085, val_185). Their errors are extra false claims or self-contradictions, which a correct key cannot catch. **Gold therefore cannot replace the claim-level audit.** It only prioritises it.
- The 16 open/partial items, where v2.1 checks less, pass at 88–94% and should be audited first.

**Remaining gap.** F4 (seed variance) needs GPU: P1-RAW, then F-s43/44.

### 5.15 "What else did you do?" and Stretch A (decided 2026-10-02)

- **Stretch A chosen** (third tool `calculate_egfr`); full plan in [STRETCH_A_PLAN.md](STRETCH_A_PLAN.md). Stretch B is not done.
- **Rationale.** SFT reliably learned the *in-distribution* two-tool policy, including when not to call. Stretch A tests whether that is a transferable tool-use policy. It runs three zero-shot arms (schema only, one-line prompt, base) and one SFT arm with 52 template rows, paired with age-removal probes and a core regression check.
- **Do not claim** "tool use is fully solved by SFT": the evidence covers 2 tools, one call, templated questions and one seed.
- **New data finding.** Table eGFR is not derivable from creatinine, age and sex (median |Δ| 26–37 vs CKD-EPI 2021). eGFR examples therefore use only records without a table eGFR.

**Answers kept for "with more time" (user framing, 2026-10-02).**

- **Seeds.** More seeds, to rule out a lucky draw. Note: bnb/QLoRA is not bit-deterministic, so "re-running seed 42" reproduces approximately. Strongest current answer: the effect size (25/34 vs 0/34) and the fact that the filtered failure appears across the 3 wave-2 learning rates on natural Q5.
- **8B.** The 8B base calls more readily (46/55 grounded tool tasks vs 29 for 4B; 25/34 intact partners vs 20/34), so it fabricates as tool arguments (28/34) where 4B avoids fabrication by not calling. More capability needs explicit negatives. That a larger model helps with many tools is a hypothesis, not a finding.
- **Numeric false positives under v2.1.** Known cases from targeted review (not an exhaustive rate):
  - val_062: self-contradiction; "any number in band" matching accepts it;
  - val_239: an extra false ALT claim; v2.1 checks only the requested items;
  - val_054: an ambiguous ranking criterion plus an unchecked wrong 194%;
  - val_124: wording preference ("closest").

  Root causes: permissive number matching, no checking of unrequested claims, and regex parsing that cannot judge whole-answer consistency. With more time: human annotators for claim-level labels, then a calibrated LLM judge or a reward model. A usable reward model needs on the order of a thousand labels, so a small human set plus a calibrated judge comes first.
- **Scorer vs v1.** On the same 110 clean holdout items, v1 is 65.7% (κ 0.26, 35 false fails) and v2.1 is 92.6% (κ 0.72). The reference is LLM adjudicators, and numeric false positives remain.
