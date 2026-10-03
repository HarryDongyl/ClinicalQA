> **Archived 2026-10-03.** Superseded by `docs/PROJECT_WALKTHROUGH.md`. Kept unchanged below as a historical record; relative links and some script paths (now `scripts/legacy/`) may be stale.

# Round 4: first-principles review — findings and decisions

This document is for explaining the project, for example in an interview. For each layer it gives:

- the question asked;
- the evidence that answered it;
- the decision taken;
- why the alternatives were rejected;
- the follow-up questions to expect.

The formal decision entries are D-053 to D-068 in [DECISIONS.md](DECISIONS.md). The runs they lead to are in [EXPERIMENTS_WAVE3.md](EXPERIMENTS_WAVE3.md).

## Review status (2026-09-30)

This is the supplied explanatory narrative with decision IDs reconciled to D-053..D-068 and material review corrections added. Exact original wording is preserved in `history/wave3_import_2026-09-30/ROUND4_FINDINGS.incoming.md`. The qualifications here and in [WAVE3_REVIEW.md](WAVE3_REVIEW.md) govern interpretation; the document does not establish a scorer release or authorize test-driven experiments.

Verified from training records with the existing measurement parser: 109 BMI-uncertain rows comprise 65 with weight only, 43 with height only, and one with neither. The 78 existing Q5 proposals are all classified as missing both. This is parser-level evidence, not independent per-record adjudication. The opening claim that no training example remains is therefore incorrect: one remains.

The supplied findings give mechanisms and aggregate claims but not all run rosters, item IDs, extraction code or external citations. AUROC 1.0, token-share percentages, 17/42 fabrication, allergy slices and comparative external-paper gains remain unverified in this review. A high AUROC measures ranking against a specified label set, not perfect calls at the deployed threshold; unsupported Q5 gold labels also change what that ranking means. Supervised-token share is not measured gradient magnitude or influence. The coverage-gap and lexical-shortcut explanations are hypotheses to test, not identified causal mechanisms.

The heldout scorer agreement was measured against LLM-agent labels on test-derived packets, with disputed items excluded and earlier test exposure. Call it packet-specific model agreement, not clean clinical validation. The known false passes remain relevant. No new raw test records were read for this follow-up.

**Evidence base**

| Source | What it contains |
|---|---|
| Wave 1 | 3 runs × 2 epochs, plus the test run of the selected model (RunPod RTX 4090) |
| Wave 2 | lr sweep on the Q5-filtered view, plus an inference-only prompt v1/v2 comparison |
| Rescoring | All wave 1 and wave 2 runs rescored with scorer v2.1: reported 92.6% agreement, κ 0.72 on the agreed subset of a test-derived LLM-adjudication packet; not independent clinical validation; v1 scored 65.7%, κ 0.26 on the same items |

---

## 0. The story in one paragraph

- **Scorer.** The first evaluation used a simple string/regex scorer, which is what the assignment allows. It understated both the base model and the SFT model. Most of the apparent SFT gain on extractive questions was the SFT model learning the gold answer style that the scorer rewarded. After building and blind-validating a MedCalc-style scorer, the durable SFT gains are in tool use, abstention and format; the remaining weakness is numeric reasoning.
- **Label noise.** 78 training records (3.9%) have no weight or height in the note, yet gold still calls the BMI tool. They taught the model to invent measurements confidently.
- **Filtering was not enough.** Removing these records only moved the fabrication from the tool arguments into the answer text, and more so at higher learning rates. A plausible contributor is sparse coverage: one training example remains for both measurements missing. This does not establish causation.
- **The general lesson:** the observations motivate hypotheses about data coverage and scoring. They do not rule out effects of loss weighting, prompts or interactions. The next round changes the data and adds diagnostics; the loss and the hyperparameters stay fixed.

---

## Layer 1: loss and optimisation

### 1.1 Should different answer types have different loss weights?

**Facts:**

- **How the loss is computed.** HF Trainer with `num_items_in_batch` computes a token-mean cross-entropy over all supervised tokens in the accumulated batch of 16. This weights token losses rather than examples; token counts alone do not determine gradient norms or causal influence. Verify normalization with the actual Trainer/PEFT implementation.

  | Type | Share of examples | Mean supervised tokens | Share of supervised tokens (not gradient influence) |
  |---|---|---|---|
  | extractive | 40% | ~28 | ~16% |
  | numeric | 20% | ~79 | ~23% |
  | uncertain | 15% | ~93 | ~20% |
  | tool_call | 25% | ~113 | ~41%, of which the call turn is ~10% |

- The call/no-call decision is a single token (`<tool_call>`).
- **Which parts are under-weighted, and does it matter?** Extractive and the call decision get the smallest shares. They are also the parts that are already solved:
  - extractive scores 0.99–1.00 under v2.1;
  - the call decision has AUROC 1.0 on first-token log-probability.
- **What actually fails.** Numeric reasoning (0.80–0.90) and missing-information coverage.

**Decision (D-053):** keep token-mean cross-entropy. No per-type, per-example or per-segment weights.

**Why:**

- Re-weighting should follow failure analysis, not a sense that the types ought to be balanced.
- Per-example or per-type averaging would move gradient away from numeric, the weakest type, towards extractive, which is already solved.
- Numeric is reported to account for about 23% of supervised tokens. Error patterns alone do not establish whether signal weighting would help; retaining unweighted loss is a conservative control choice.

**External evidence (verified primary sources):**

- LLaMA-Factory, Axolotl and TRL all train on the call turn and mask the tool response. We do the same.
- The global token mean is the standard fix for the late-2024 gradient-accumulation bug.
- No verified study shows that up-weighting call tokens helps tool SFT. Hammer's "function masking" is a data augmentation, not a loss weight. Rho-1 selective token loss was only validated for pre-training.

**Likely follow-ups:**

- *"Isn't the long uncertain explanation dominating the gradient?"* Its share (20%) is close to its share of examples (15%). Its val loss is the highest (0.59), but the cause could involve answer entropy, label quality or learning; cross-type loss alone cannot distinguish these.
- *"Would you ever weight?"* Yes, if a type failed because of too little signal. The test would be a segment-weighting ablation: down-weight explanation tokens and check numeric accuracy and fabrication.

### 1.2 Hyperparameters

**Facts:** wave-2 lr sweep on the Q5-filtered view, v2.1 macro on val.

| lr | ep1 | ep2 |
|---|---|---|
| 1e-4 | 0.936 | 0.914 |
| 1.5e-4 | 0.931 | 0.921 |
| 2e-4 | 0.908 | 0.941 |

- The per-epoch LR spans are below 0.03 macro, but this does not establish equivalence. The reviewed paired epoch-two 2e-4 minus 1e-4 interval is approximately [+0.35, +5.53] percentage points under v2.1, unadjusted and single-seed. See WAVE3_REVIEW.md.
- Wave-1 lr 1e-4 beat 5e-5 with a paired CI of [0.007, 0.066], but that was measured with v1.
- The val loss was still falling at epoch 2. Epoch 1 vs 2 shows no consistent direction.

**Decision (D-054):** lock QLoRA r16/α32 on all linear layers, lr 1e-4, cosine, 3% warmup, 2 epochs, effective batch 16. Run 3 seeds. Stop sweeping.

**Why:** more sweeping inside the noise band produces a winner's curse, i.e. picking whichever run happened to score high. Seed variance is a more informative number to report than a "best lr". 2e-4 ep2 scored highest, but the same lr scored lowest at ep1, so its checkpoint ranking changes; this is not evidence of numerical training instability. The mb4 throughput test was stopped on purpose (peak VRAM 17.75 GB).

**Interaction found later (Layer 3):** the learning rate interacts with the data. At lr ≥1e-4 the filtered model fabricates in text on Q5 items; at 5e-5 it almost never does. So a hyperparameter and a data decision cannot be judged independently.

---

## Layer 2: input/output representation

### 2.1 Do training and inference see the same input?

These were checked in the code; all hold:

- There is one `render()` path, used for both training and inference. The tool call is re-rendered through the same template during rollout.
- The input order is fixed: `## Encounter note` → `## Table (type)` (Markdown table; empty units shown as `(no unit)`) → `## Question`.
- Every sample carries the same `TOOL_SCHEMAS`, including non-tool types. Its hash is recorded.
- answer_type only lives in the sidecar files, and a test guards against label leakage.
- Qwen's native special tokens are used. The call turn's `</tool_call><|im_end|>` is supervised, which a mask audit confirmed. TRL flags stop-token loss as a known risk for Qwen templates; it does not affect us.
- Tool responses are masked and rendered as `<tool_response>` in a user turn. Instruct-2507 has no `<think>`.

### 2.2 Which system prompt?

**Facts:**

- Prompt v2 is v1 plus one line: "use only reference ranges supplied in the input".
- Changing to v2 **at inference only**, on adapters trained with v1, flipped 0–2 items per type in paired comparisons. All CIs include 0.
- Numeric, the type the new line targets, gained one item. The selected model got slightly worse overall (0.924 → 0.917).
- The base model's tool score rose from 0.63 to 0.74, but that is 7 of 62 items, and the added line has nothing to do with tools.
- 36 training gold answers correctly use a standard range when the input gives none (Q11). v2's new line contradicts them.

**Decision (D-055, D-056):**

- Train and infer with v1.
- Report R0 twice: R0-v1 for parity, and R0-v3, a strong prompt, as the stronger prompted baseline, not an upper bound.

**Why:**

- The SFT models are insensitive to small prompt changes; their behaviour comes from the weights.
- An inference-only prompt change is an input shift that can improve or degrade performance; matched validation is required. The original prompt remains the parity control.
- A prompt rule that contradicts the training labels would teach inconsistency.
- The base model has not received this project's SFT, but it is already instruction-trained; use the same prompt for the parity comparison. It should get its best prompt, so that the claim "SFT beats prompt engineering" is fair.

**Likely follow-ups:**

- *"The v2 numbers look higher, why not use it?"* Failure to exclude zero is not equivalence. Net count changes differ from item flips; base has 13 tool-label flips with a net gain of seven under v2.1.
- *"Why not put the v3 rules in the training prompt?"* The model already learns those rules from the training data. Putting them in the prompt too would make it harder to say what SFT itself learned.

### 2.3 Should the call turn contain reasoning?

**Facts:**

- External: ToolACE gained +12pp pass rate with thinking before the call; Llama 3 interleaves reasoning with calls in multi-step use.
- Ours: the call decision is already perfect.
- The only remaining tool error is imperial mental-arithmetic drift of 0.1–0.2 (6 test items, 0 val items).

**Decision (D-058):** the core configuration keeps the call turn empty. One ablation, **A-VIS**:

- numeric targets become `Working:` (rendered deterministically from the input-derived key, and added only when the key agrees with gold) followed by `Answer:` (gold, unchanged);
- imperial call turns get one conversion line using "≈".

**Why:**

- The problem it targets only appears on test, so it should not drive the core design.
- It changes both the output format and one evaluation metric (`call_logprob`), so it must be isolated.
- The `Answer:` anchor also gives the scorer a MedCalc-style final-answer field.

---

**Development-boundary correction:** the proposed imperial call-line intervention is motivated here by test-only errors. An ablation is still a development decision, so it cannot be justified under the validation-only policy on that basis. Defer this component. A numeric-only explanation ablation can be motivated by permitted validation errors. Score all visible claims, including Working text, not only Answer completeness.

## Layer 3: data design, sampling and quality

### 3.1 Q5: gold calls BMI when the note has no weight or height

**Facts:** 78 train, 7 val and 10 test records. The behaviour on the 7 val items:

| Model | Abstains correctly | Fabricates arguments in the call | Fabricates values in the text |
|---|---|---|---|
| base | 7/7 | 0 | 0 |
| raw SFT (all lrs) | 0–1 | 6–7 | 0 |
| filtered, lr 5e-5 | 6–7 | 0 | 0–1 |
| filtered, lr ≥1e-4 | 2–5 | 0–1 | 2–4 |

- The raw model's first-token probability of calling on these items is about 0.99, and it repeats a memorised 178.5 cm.
- The filtered model writes "The patient's weight is documented at 68.5 kg … using the calculate_bmi tool …" without making any call.
- **Candidate mechanism (not causally established).** After filtering, the training set has **1** example of "both weight and height missing → abstain". The 108 other BMI-uncertain examples each miss one measurement, and their gold opens with "The patient's weight is documented at X kg…". The model copies that opening and fills in both numbers.

**Decision (D-059):** relabel the 78 records as uncertain, with a two-part answer: what is documented, and what is missing. Deleting them is no longer the chosen approach.

**Why:**

- Filtering removes the wrong signal but does not add the right one. Relabelling fills the coverage gap.
- Uncertain rises from 15% to 19% of training, a small shift.
- Hammer reports that call accuracy and abstention accuracy trade off against each other, so the acceptance criteria include "the no-call rate on grounded tool items does not rise".
- APIGen and Llama 3 both add explicit "insufficient information → do not call" examples.

**Likely follow-ups:**

- *"Isn't relabelling editing gold?"* Only on train, only for records whose gold is provably unsupported by its own input, with the reason documented. Val and test are untouched.
- *"Why did the first analysis say filtering was enough?"* Because it was measured at lr 5e-5. At the locked lr it isn't. The lesson is that data decisions have to be checked at the learning rate that will actually be used.

### 3.2 Coverage gaps and shortcuts

**Allergy (D-061):**

| Allergy questions | Train examples | Val accuracy (6 runs) |
|---|---|---|
| The question mentions allergy | 51 | 36/36 = 100% |
| The question does not mention allergy (the model has to notice the missing allergy section itself) | 5 | 8/18 = 44% |

- Every "is it safe…" question in train is of the uncertain type.
- This is consistent with a surface-cue shortcut, but also with differences in ambiguity or difficulty; matched context-preserving probes are needed to distinguish them.
- Decision: the core configuration is unchanged, and the finding is reported as a slice. Ablation **A-IMPL** rewrites 25 of the 51 explicit questions into the implicit form, with gold unchanged and the total count unchanged, so only the question form varies.

**"Most abnormal" scale (D-060):**

- Gold picks the relative deviation in 13 of the 15 cases where relative and absolute deviation disagree. Only 5 of those 15 have matching units.
- Some remaining errors are genuine; for example val_202 is wrong under both scales.
- Decision: no data change, since editing gold or upsampling would not address a reasoning weakness. A-VIS makes the comparison explicit instead, and the scorer accepts both scales when units match.

**Q11 external reference ranges (D-062):** kept. When the input gives no range, answering requires the standard range, so this is part of the task design, not noise. Correction: val_085 has a vitals-only table. Its input supplies BUN 52.1 and creatinine 1.0 but no BUN reference bound of 20; the reference answer and scorer supply that bound. Whether external ranges are allowed must be settled as a task policy rather than inferred from gold.

### 3.3 Other flags (kept; handling documented in DATA_QUALITY.md, D-068)

| Flag | Count | Handling |
|---|---|---|
| Q6: imperial rounding | 51 | Handled by tolerances |
| Q7: stated BMI inconsistent with weight/height | 119 notes | Gold uses the computed value |
| Q13: µ vs μ | 56 | Units normalised |
| Q16: synthetic-data plausibility artefacts | 219 | Kept |
| Q17: long answers | 35 | Kept |
| Q14 | 3 | Kept |

Leakage checks Q3 and Q18 pass.

---

## Layer 4: evaluation

### 4.1 Why the evaluation was rebuilt

**v1 scorer.** Gold-word matching with same-sentence direction words and token-F1. On blind adjudication it agreed with the adjudicators only 65–68% of the time:

- About 58 of 61 base-model extractive failures were false negatives.
- About half of the SFT numeric failures were false negatives.

**v2 scorer**, designed after MedCalc-Bench and validated in stages:

- Answer keys are computed from the inputs before any prediction is read.
- Checks are typed, each with its own tolerance. Every prediction gets pass or fail, with no review bucket.
- Tool calls are scored on the executed outcome (τ-bench).
- Abstention uses deterministic gates (AbstentionBench).
- Ambiguous questions accept several answers (BFCL).

| Stage | Result |
|---|---|
| Gold self-check | train 98.6%, val 100% |
| v2.0 on its development set | 99.2% |
| v2.0 on a frozen scorer's test-derived adjudication packet | **76.7%**, showing it had overfit its development set |
| v2.1, after general fixes | **92.6%, κ 0.72** on a second test-derived adjudication packet (v1: 65.7%) |

Details: [candidate v2.1 audit](SCORER_V2_1_AUDIT.md), (SCORER_V2.md documents the different bounded implementation), [../reports/scorer_v2/VALIDATION.md](../reports/scorer_v2/VALIDATION.md).

**Honest limitations of v2:**

- Both parsing layers are regex rules fitted to this dataset's templates, so it will not transfer to new phrasings for free.
- The adjudicators were LLM agents, not clinicians.
- The next step, an LLM parser with a deterministic verifier, is deferred (D-064).

### 4.2 Model selection (D-063)

- **What went wrong in wave 1:** it picked the best of 6 candidates using v1 on the Q5-grounded subset. That rule could not see Q5 fabrication, and it selected the model that fabricates.
- **Metric:** v2.1 macro on the full val set.
- **Hard gates:**
  - Q5 fabrication ≤1/7;
  - the no-call rate on grounded tool items does not rise;
  - zero parse errors.
- **Unit of comparison:** a configuration, as the mean ± sd over 3 seeds, with epoch 2 fixed in advance.
- **Ablation vs control:** a same-seed paired bootstrap; a CI that includes 0 is insufficient evidence of a difference, not proof of equivalence.
- v1 is always reported alongside (the assignment accepts simple implementations), but it does not decide.

### 4.3 Test protocol (D-065)

- Full test (400 records). Only the final configuration (3 seeds), R0-v1 and R0-v3 are run on it; ablations stay on val.
- Disclosed:
  - this is the second model evaluation on test;
  - test outputs were used to validate the scorer, and v2.1 rules were adjusted on 120 of the test items. The arithmetic 2.4pp × 120/400 does not estimate contamination bias: scorer changes can affect all items and model rankings, and no counterfactual unbiased score is available. The magnitude is unknown.
- The model list is frozen and committed before the run.

### 4.4 Counterfactual probes (D-066)

About 40 pairs derived from val. Automatic scoring is a proposed implementation, not guaranteed by input-derived keys. Every mutation needs consistent edits, recomputed expectations and contract coverage checks:

| Probe | Change | Expected behaviour |
|---|---|---|
| P1 | Remove all mentions of a required measurement and any answer-revealing derived value | Name the missing input when the question is no longer answerable |
| P2 | Add the missing required measurement | Call only when all required inputs are now sufficient and a calculation is requested |
| P3 | Switch metric to imperial | Same BMI |
| P4 | Change a value and update related facts | Recompute state/ranking; they need not change unless the relevant boundary is crossed |
| P5 | Remove "allergy" from the question | Still flag the missing allergy section |

This answers "does the model read the note, or follow a template?" with validated expected behavior; it does not by itself identify the model's internal mechanism.

### 4.5 Reporting (D-067)

- The five assignment metrics, under their spec names and with the spec denominators, v1 and v2.1 side by side.
- A diagnostics table:
  - tool end-to-end;
  - Q5 abstention and fabrication (call and text);
  - over-call and over-refusal;
  - self-correction;
  - slices: explicit vs implicit missing field, same-unit "most", imperial vs metric;
  - probes.
- Wilson CIs, seed sd, paired bootstrap.

---

## Numbers to remember

| Fact | Value |
|---|---|
| Training set | 2,000 records: 40% extractive, 20% numeric, 25% tool, 15% uncertain |
| Training cost | 4B QLoRA: about 44 minutes for 2 epochs on an RTX 4090, 10 s/step, peak 7.8 GB |
| Base → SFT (wave-1 selected model, test), v1 | macro 0.44 → 0.82 |
| Base → SFT (wave-1 selected model, test), v2.1 | macro 0.79 → 0.93 |
| Extractive under v2.1 | base 0.92–0.98, SFT 0.99–1.00 |
| Tool under v2.1 | base 0.57–0.74, SFT 0.87–1.00 |
| Numeric (weakest type) | SFT 0.80–0.90 |
| Call-decision AUROC (first token) | 1.000 after SFT (base 0.966) |
| Q5 fabrication | raw 6–7/7 in the call; filtered at lr ≥1e-4: 17/42 in text |
| Allergy shortcut | 36/36 explicit vs 8/18 implicit |
| Scorer agreement with adjudicators | v1 about 66%; v2.1 92.6% (κ 0.72) on the agreed test-derived adjudication subset |

## Open items and next steps

1. Wave 3 per [EXPERIMENTS_WAVE3.md](EXPERIMENTS_WAVE3.md).
2. Write `reports/DATA_QUALITY.md` (D-068).
3. Update `reports/REPORT.md` with the tables from D-067.
4. Scorer v3: an LLM parser with a deterministic verifier; a HealthBench-style rubric judge for the open explanatory parts.
5. If there is time: relabel/augment for the explicit/implicit balance beyond A-IMPL; negatives for "safe" questions where the allergy history is complete; upgrade to an 8B model for numeric reasoning.
