# Project walkthrough: from the assignment to the current solution

This document retells the project in order, starting from nothing but [ASSIGNMENT.md](ASSIGNMENT.md). For each stage it gives:

- **Goal**: what the stage had to answer;
- **What was done**;
- **Evidence**: concrete numbers and IDs;
- **Decisions**: the choice and why;
- **Hindsight**: what was wrong or would be done differently.

Status: 2026-10-03, final. The final models are F′ (core) and A-sft2 epoch 2 (Stretch A); the frozen test run is complete (stage 19).

Numbers below are validation unless marked test. Scores are labelled by scorer:

- **v1** is the original simple scorer;
- **v2.1** is the validated MedCalc-style scorer, a diagnostic and not a verified clinical accuracy.

Detailed evidence lives in:

- [DECISIONS.md](DECISIONS.md) and [EXPERIMENT_JOURNAL.md](EXPERIMENT_JOURNAL.md);
- [SCORER.md](SCORER.md), [SCORER_V2_1_KNOWN_ISSUES.md](SCORER_V2_1_KNOWN_ISSUES.md) and [../reports/scorer_v2/VALIDATION.md](../reports/scorer_v2/VALIDATION.md);
- [STRETCH_A.md](STRETCH_A.md);
- [../reports/REPORT.md](../reports/REPORT.md), [../reports/DATA_QUALITY.md](../reports/DATA_QUALITY.md) and [../reports/w4/test/TEST_REPORT.md](../reports/w4/test/TEST_REPORT.md);
- archived round reviews in [history/](history/).

---

## The chain at a glance

| # | Stage | One-line outcome |
|---|---|---|
| 1 | Requirements | Four behaviours. The central tension is *call tools reliably* vs *never invent inputs* |
| 2 | Constraints & success definition | One 24 GB GPU, train/val/test discipline, pre-registered selection, behavioural metrics beside the five required ones |
| 3 | Data analysis | 19 quality checks. 3.9% of training tool labels call BMI with invented measurements (Q5) |
| 4 | Representation | Native Qwen chat template and tool-call format, assistant-only loss masks, Markdown table, one call |
| 5 | Model & training | Qwen3-4B-Instruct-2507, QLoRA r16, lr 1e-4, 2 epochs, effective batch 16, token-mean CE |
| 6 | Evaluation v1 | Rollout with real tool execution; five metrics plus diagnostics; simple regex scorer |
| 7 | Wave 1 | SFT works (v1 macro 0.44 → 0.82 on test), but the selected model invents BMI inputs, and v1 under-credits answers |
| 8 | Scorer rebuild | Research → MedCalc-style v2.0 → failed a clean holdout (76.7%) → v2.1 passes a second one (92.6% vs v1 65.7%) |
| 9 | Wave 2 | The LR sweep sits inside noise; prompt tweaks don't matter for SFT; **filtering moves fabrication into prose** |
| 10 | First-principles review | Lock loss and hyperparameters, keep prompt v1, relabel Q5, pre-register selection, add probes |
| 11 | Wave 3 | Relabel takes missing-input fabrication from 25/34 to 0/34 with no tool loss. Prompting and an 8B base do not solve it |
| 12 | Qwen3.5 | The filtered failure replicates on a second family (29/34); Qwen3.5 looks stronger on numeric |
| 13 | CPU gap work | Clinical-context check, call-decision ECE, numeric audit strata |
| 14 | Stretch A | Third tool `calculate_egfr`: data generated, evaluation set reviewed and frozen |
| 15 | Wave 4 design | H1 cross-family replication, H2 numeric superiority (sign test), H3 tool transfer. Gates and stop rules frozen |
| 16 | Wave 4 round 1 | **H1 holds** (Qwen3.5 P1 29/34 → 0/34); P1-RAW 32/34 rules out rows/steps; the families tie on relabel data (+0.25 pp), so **Qwen3 is kept** for later experiments, on cost (sequences 11% longer on Qwen3.5); gate needs review (val_104) |
| 17 | Scorer finding | v1 over-fails numeric (51% agreement on the clean holdout); v2.1 under-fails false *extra* claims. A guarded v2.2 was prototyped, **not adopted** |
| 18 | F′ refit and Stretch A | F′ retrained (adapter lost): P1 0/34 again, but only 136/250 val items identical and macro 1.7 pp lower (one refit, not a variance estimate). A-sft fabricated ages (16/19); the diagnosed fix, A-sft2, reaches 17/19 end-to-end with 0/19 fabrication at about 1 pp core cost |
| 19 | Test run | Frozen five-model list, run once: F′ 97.5 v2.1 macro and 0/10 natural-Q5 fabrication; filter-only 6/10; Qwen3.5 relabel ties F′ |
| 20 | Limitations & next steps | Single seed, test reused, LLM adjudicators, regex parsing, template-bound data |

---

## 1. Requirements analysis

**Goal.** Turn the assignment into measurable behaviours and deliverables.

**What the assignment asks.** SFT a model to answer one clinical question from one encounter note plus one structured table. It must:

1. extract facts from the note or table;
2. do simple numeric reasoning (compare, subtract, thresholds);
3. use tools: decide *when* to call `unit_convert` / `calculate_bmi`, produce *valid* arguments (including an implicit imperial → metric conversion; gold arguments are always metric), and use the result;
4. state uncertainty when information is missing, without hallucinating.

**Deliverables.** A formatting pipeline, a training script, a data-quality analysis (≥ 5 formatted examples plus statistics), an evaluation script with five named metrics, and a report: setup, commands, hardware, artefacts, what was tried, results, limitations.

**Grading weights.** Data formatting 10%, **fine-tuning setup 40%**, evaluation 10%, code quality 20%, stretch / going beyond 20%.

**Decisions.**

| Decision | Logic |
|---|---|
| "Do not train on test" extended to **"no selection on test"** | A checkpoint, threshold or hyperparameter chosen on test leaks test information as surely as training on it |
| Tool use decomposed into the assignment's four questions: decide to call; correct tool; valid, grounded arguments; result incorporated with brief clinical context | Failures must be attributable to a step |
| "Calibrated" read first as **behavioural**: abstain when data is missing, don't over-refuse. Probability calibration is only diagnostic | The text says "explicitly stating uncertainty", which is a behaviour |
| One tool call per answer | Every gold `tool_call` record has exactly one call |
| Core first, Stretch later | "Complete the Core track first" |

**Central tension**, which became the spine of the project. Pushing the model to call tools reliably encourages it to call when inputs are absent and to invent the arguments. Making it cautious suppresses correct calls. So both sides are measured together, every time: abstention on missing inputs **and** correct calls on intact inputs.

---

## 2. Constraints and the definition of success

**Goal.** Fix the boundaries and the success criteria before any experiment.

| Area | Choice | Evidence / reason |
|---|---|---|
| Hardware | One 24 GB GPU → a 4B model with QLoRA | Measured peak 7.8 GB on an RTX 4090 |
| Data discipline | Train only for training, val only for selection, test once at the end from a frozen, committed list | Prevents leakage and adaptive overfitting |
| Reproducibility | uv lockfile; hashes of code, data, prompt, template and tool schemas in every run; seed 42; greedy decoding | Every comparison is attributable |
| Success | The five required metrics **plus** behavioural metrics (over-call, over-refusal, fabrication) | The five alone would score a never-calling model as non-hallucinating |
| Selection | Rules written before results: which checkpoint, gates, comparator | Avoids winner's curse and post-hoc choice |
| Statistical humility | val n = 250; per-type minimum detectable effect about 14–23 pp (numeric n = 50 → about 20 pp) | No decisions from small per-type deltas |

**Why greedy decoding.**

- Paired, noise-free comparisons.
- The tasks have one correct answer, so the mode is what is wanted.
- One sample per item keeps evaluation cheap.
- The first-token call probability is a valid diagnostic only under greedy decoding.

Costs: only the most likely trajectory is seen, so a sampled deployment could expose fabrication that greedy hides; there is no self-consistency confidence; outputs are not bit-identical across GPU kernels.

**Hindsight.** The pre-registered wave-1 rule turned out to be wrong (stage 7). It was replaced openly, with the reason documented, and applied only to later waves. A rule may change, but only before seeing the results it judges.

---

## 3. Data analysis

**Goal.** Know the data and its defects before formatting or training.

**What was done.**

1. Byte-identical copies of the provided files, with SHA-256 recorded and re-verified before use.
2. Schema validation (Q1).
3. Statistics per split.
4. 19 targeted quality checks, Q1–Q19.

Generated report: `reports/data_analysis.md`. Every flag: `reports/quality_flags.jsonl` (624 flags).

**Evidence: statistics.**

- Splits 2,000 / 250 / 400. Answer-type mix 40 / 20 / 25 / 15 in every split (val 40 / 20 / 24.8 / 15.2).
- Train tool calls: 410 BMI and 90 unit conversions.
- Of the 410 train BMI calls, 112 come from imperial notes.
- Tokenised conversations: p50 1,375, max 1,614 tokens.

**Evidence: quality checks** (train / val / test).

| Check | Finding | Handling |
|---|---|---|
| **Q5: argument grounding** | **95 of 534 BMI calls (78 / 7 / 10) have weight and height nowhere in the note, table or question.** The gold arguments are invented | Most consequential. Training on them teaches argument hallucination (stages 9–11) |
| Q5 corollary | 67 BMI examples give the measurements only in the question. Disjoint from the 95, because the question counts as input | Kept. The prompt says to ground in "explicit values in the question". P1 probes exclude such items |
| Q6: imperial rounding | 51 heights don't round-trip to the gold cm. Exact conversion changes the BMI by ±0.1 in 20 of 138 imperial cases | Handled with scorer tolerances; no relabel |
| Q7: stated BMI vs measurements | 119 notes state a BMI inconsistent with their own weight and height | Kept; gold uses the computed value |
| Q10: arithmetic | The initial flags were parser false positives (decimal backtracking) | Parser fixed, regression tests added, rows kept |
| Q11: external reference ranges | 46 extractive answers cite a standard range not in the input | **Task design, not noise**: when the input gives no range, the standard range is required |
| Q13: µ vs μ | 56 | Units normalised |
| Q16: synthetic plausibility | 219 | Kept and documented |
| Q3 / Q18: overlap between splits | None | — |

**Principle.** Heuristics produce *candidates*, not proof that gold is wrong. Each check has an explicit keep / flag / exclude policy.

**Hindsight.** Some data problems only appeared through model behaviour, after training. Data analysis is therefore revisited after training:

| Problem found later | Stage |
|---|---|
| After filtering Q5, only **1** train example covers "both measurements missing" | 10 |
| Allergy questions that mention allergy: 51 train examples; implicit ones: **5**. Val accuracy 36/36 vs 8/18 | 10 |
| The table eGFR is not derivable from creatinine, age and sex (median deviation 26–37) | 14 |

---

## 4. Input/output representation

**Goal.** Decide exactly what the model sees and emits, and keep training and inference identical.

| Decision | Logic / evidence |
|---|---|
| Qwen's **native chat template** and native `<tool_call>{"name","arguments"}</tool_call>`, rendered with `apply_chat_template(tools=...)` | Matches the model's post-training format, which is the strongest prior for tool calling |
| A fixed user message: `## Encounter note` → `## Table (type)` as Markdown (pipes escaped, empty units shown as `(no unit)`) → `## Question` | Consistent and token-efficient. Table reading is not a bottleneck: SFT extractive scores 0.99–1.00 and zero-shot base 0.92–0.98 under v2.1 |
| Identical tool schemas in every sample, including non-tool types | The model learns *when* to call, not *whether tools exist* |
| answer_type lives only in sidecar files; a test guards against label leakage | The model must not see the label |
| The call turn has empty content; the tool result goes into a `tool` message (rendered as `<tool_response>`); then the final answer | Native trajectory |
| **Assistant-only loss masks** built by prefix-diff rendering. TRL's `assistant_only_loss` is not used, because the stock Qwen3 template has no generation markers (TRL lists Qwen3 as needing a patched template) | Mask audit: 2,000/2,000 conversations clean; 5.5% of tokens supervised; the call turn's `</tool_call><|im_end|>` is supervised |
| One `render()` path for training and inference; calls are re-rendered through the template during rollout | Train/inference parity by construction |
| System prompt v1: grounding, tool use, missing information, "missing allergy documentation ≠ no allergies", concise final answer | Frozen and hashed into every run |
| Gold metric arguments copied as targets; imperial conversion is learned implicitly | Follows the dataset contract. An explicit conversion chain is a separate ablation |

---

## 5. Model and training setup

**Goal.** A reasonable, justified, reproducible fine-tuning recipe that fits one GPU.

| Choice | Value | Logic / evidence |
|---|---|---|
| Base model | Qwen3-4B-Instruct-2507 at a pinned revision | Native tool calling; non-thinking variant, so masks are simple; fits 24 GB; zero-shot call-decision AUROC already 0.966 |
| Method | **QLoRA** NF4, bf16 compute, LoRA r16 / α32 / dropout 0.05 on all 7 linear projections (about 33M parameters) | Full fine-tuning needs about 64 GB (bf16 weights 8 + gradients 8 + fp32 master 16 + Adam 32). bf16 LoRA needs about 13 GB, so QLoRA was chosen for headroom (larger model or batch). The literature (QLoRA paper) says rank matters little when all linear layers are covered |
| Optimisation | lr 1e-4 cosine, 3% warmup, 2 epochs, micro-batch 1 × accumulation 16 (250 steps), weight decay 0, grad-norm 1.0, gradient checkpointing | A conventional QLoRA range |
| Loss | **Token-mean cross-entropy** over supervised tokens, normalised with `num_items_in_batch` across accumulation | The standard after the HF gradient-accumulation fix |
| Seed / provenance | 42; every run records git commit, packages, hardware and hashes | Reproducibility requirement |

**Evidence: gradient share by type** (supervised-token estimate):

| Type | Share of examples | Share of gradient |
|---|---|---|
| extractive | 40% | about 16% |
| numeric | 20% | about 23% |
| uncertain | 15% | about 20% |
| tool | 25% | about 41%, of which the call turn is about 10% |

Re-weighting was rejected: the under-weighted parts (extractive and the call decision) turned out to be already solved, and numeric errors are reasoning errors, not lack of gradient. No verified study supports up-weighting call tokens for tool SFT.

**Evidence: micro-batch.** Micro-batch 4 × accumulation 4 (wave 2) was **32% slower** per optimiser step (13.5 vs 10.2 s) and used 17.75 vs 7.8 GB. Its loss curve was identical to micro-batch 1 (1.915 / 0.858 / 0.582 vs 1.918 / 0.867 / 0.583 at steps 1 / 10 / 20). This is direct evidence that the loss normalisation is correct. The slowdown is probably padding forcing a slower SDPA kernel (a hypothesis). The fix would be padding-free packing.

---

## 6. Evaluation design (v1)

**Goal.** A runnable evaluation that reports the five metrics and can attribute failures.

**What was built.**

- A rollout state machine: generate → parse (strict; malformed JSON is never repaired) → validate against the schema → **really execute the tool** → feed the result back → final answer.
- Budget: one call, two assistant turns, 256 tokens per turn, 512 total. Greedy decoding, left padding.
- Logged per trajectory: raw turns, parsed calls, executor input and output, stop reason, token log-probs, and the first-token `<tool_call>` probability.
- **Scorer v1** (regex): extractive = gold numbers at gold precision, plus direction words in the same sentence, with a token-F1 fallback; numeric = gold numbers; tool funnel = selected → valid → arguments within ±0.05 → executed → result in the answer; uncertainty = abstention phrase + missing field named + no fabricated value.
- Extra diagnostics: tool end-to-end, over-call, over-refusal, unsupported-argument rate, Wilson CIs, paired bootstrap.
- The test set is guarded: `generate` refuses test; `final` requires a clean tree and a frozen adapter.

---

## 7. Wave 1 (RunPod RTX 4090)

**Goal.** First end-to-end result. Does SFT help?

**What was run.** Three runs × 2 epochs: raw lr 1e-4, raw 5e-5, Q5-filtered 5e-5. Selection by the pre-registered rule: v1 macro on the Q5-grounded val subset. Selected: **raw lr 1e-4, epoch 1** (macro 0.878). That model then went to test once.

**Evidence.**

| Item | Result |
|---|---|
| Test, v1 macro | base **0.442** → SFT **0.824** |
| Training cost | about 44 min, about 10 s/step, peak 7.8 GB |
| Call decision | first-token AUROC 0.966 (base) → **1.000** (SFT); tool vs uncertain 0.859 → 1.000 |
| **Q5 val items (7), raw SFT** | 6/7 **invent arguments** with p(call) ≈ 0.99 and a memorised height of **178.5 cm** |
| Q5 val items, base and filtered | 7/7 abstain |
| Test residual tool errors | 6 imperial drifts of 0.1–0.2 (e.g. 241.8 lb → 110.0 kg vs gold 109.7); 0 such errors on val |
| v1 scorer audit | Of base's 61 extractive failures, about **58 were false negatives**: correct answers in a different style. About half of SFT's numeric failures were false negatives |

**Decisions.**

- The selection rule was declared defective. It scored only the Q5-grounded subset, so it could not see that the selected model fabricates, and it picked exactly that model.
- The scorer could not be trusted for per-type claims, so it had to be rebuilt.

**Hindsight.** The 0.44 → 0.82 headline mostly reflected the v1 scorer rewarding the gold answer style. It was not a pure capability gain.

---

## 8. Rebuilding the scorer

**Goal.** An evaluation that measures correctness rather than resemblance to the gold wording, validated against independent judgement.

**Research** (primary sources; file paths verified):

| Source | Borrowed idea |
|---|---|
| MedCalc-Bench `evaluate.py` | Answer keys computed from the inputs; lower/upper tolerance bands; exact match for integer and categorical answers |
| DROP `drop_eval.py` | A number gate |
| BFCL `ast_checker.py` | Several acceptable answers; relevance (call / no call) and parse errors kept separate |
| τ-bench | Score the executed outcome |
| AbstentionBench | Interchangeable keyword detector and judge |
| HealthBench | Binary per-criterion rubric; judge-vs-physician meta-evaluation |

The user's bounded v2 scorer left 25–42% of items unresolved ("review"), so it could not rank models.

**v2.0.**

- MedCalc-style answer keys built from the table, question and note **before reading any prediction**.
- Typed checks: value, status, diff, ratio, percentage, BMI, entity, set, yes/no, with a text fallback. Each has a tolerance policy: exact for integers; ±max(0.051, 0.5%) for differences; ±5% for ratios and percentages; ±0.15 for BMI.
- A negation-aware state reader, binding across sentences.
- Tools scored on outcome; Q5 items expect abstention.
- Gold self-check rose from 90.2% → 98% during development.

**Validation protocol.**

- Blinded LLM adjudicators with a written rubric; they saw no scorer output and no model identity.
- Two independent rater groups per set. Strict raters had to cite values. One lazy rater (identical reason on 71 items) was detected and excluded.
- The scorer was frozen by sha256 before each clean holdout.

| Scorer | Dev set (tuned) | Holdout 1 (test) | Holdout 2 (test, unseen) |
|---|---|---|---|
| v1 | 65.8% (κ 0.17) | 67.5% (κ 0.34) | **65.7% (κ 0.26)**, 35 false fails |
| v2.0 | 99.2% | **76.7% (κ 0.48)**: overfit to the dev set | — |
| v2.1 | 100% | 95.0% (tuned on holdout 1) | **92.6% (κ 0.72)** |

Rater agreement: 0.92 / 1.00 / 0.93.

**Evidence: what v2.1 changes about wave 1.**

- Base extractive 0.39 → 0.92–0.98, so base already reads tables.
- The durable SFT gains are **tool use** (base 0.57–0.74 vs SFT 0.87–1.00), abstention and format.

**Hindsight.**

- Reporting the honest 76.7% holdout result, then fixing by question type and re-validating on a second unseen holdout, is the methodological highlight.
- **Remaining weakness:** both parsing layers are regex rules fitted to this dataset's templates. Numeric answers are still over-credited (stage 11); stage 17 pins down the mechanism.

---

## 9. Wave 2 (learning rate, batch, prompt)

**Goal.** Tune within the recipe and test a prompt fix for range errors.

| Experiment | Evidence | Reading |
|---|---|---|
| LR ladder on the filtered view: 1e-4 / 1.5e-4 / 2e-4 (v2.1 macro, epoch 1 / epoch 2) | 0.936 / 0.914, 0.931 / 0.921, 0.908 / 0.941 | All differences ≤ 0.03: **noise** |
| Micro-batch 4 | See stage 5 | Slower, same optimisation; stopped on purpose |
| Prompt v2 (v1 + "use only reference ranges supplied in the input"), inference only | Paired: 0–2 items flip per type; every CI includes 0. The prompt targeted numeric, which gained 1 item | SFT models are insensitive to small prompt changes. The new line also contradicts 36 correct train golds (Q11) |
| **Natural Q5 behaviour across all runs** | Raw: **50/56 fabricate via a call (89%)**. Filtered at lr 5e-5: 1/28. **Filtered at lr ≥ 1e-4: 17/42 fabricate in text** ("The patient's weight is documented at 68.5 kg…", with no call) | Filtering removes the wrong signal but teaches nothing; the fabrication moves into prose, worse at higher lr |

**Hindsight.** The wave-1 impression that "filtering fixes it" was measured at lr 5e-5. Data decisions must be checked at the learning rate that will actually be used.

---

## 10. First-principles review (four layers)

**Goal.** Re-derive every design choice from the requirements, using wave 1–2 evidence.

| Layer | Decision | Logic / evidence |
|---|---|---|
| Loss | Keep token-mean CE; no type or segment weights | Stage 5 gradient-share evidence; no supporting literature |
| Hyperparameters | Lock lr 1e-4, 2 epochs, effective batch 16; run seeds instead of more sweeps | Stage 9 noise band |
| Prompt | **v1** for training and inference | v2 effect is 0–2 items; it contradicts Q11 golds |
| Baselines | R0-v1 (parity prompt), R0-v3 (strong prompt), R0-v3-FS4 (+4 shots) | Tests "SFT vs prompting" fairly |
| Call-turn reasoning | Core keeps the empty call turn. R-VIS ablation: `Working:` + `Answer:` numeric targets built from input-derived keys | ToolACE reports +12 pp from thinking, but our only tool error was on test, so it is an ablation, not core |
| **Q5 data** | **Relabel the 78 as uncertain** with a two-part answer (what is documented / what is missing) | After filtering, only 1 "both missing" example remains (BMI questions: 332 tool both-present; 43 uncertain height-only; 65 weight-only; **1** both-missing). Hammer and APIGen use explicit no-call data. Gate on the abstain-vs-call trade-off |
| Allergy shortcut | Reported as a slice; R-IMPL ablation proposed | Explicit allergy questions 36/36, implicit 8/18; train has 51 explicit vs 5 implicit; every "is it safe" question is uncertain |
| "Most abnormal" | No gold edits | Gold uses relative deviation in 13/15 ambiguous cases; only 5 same-unit cases; val_202 is wrong under both scales |
| Q11 | Kept | Task design |
| Selection | v2.1 macro on full val; hard gates (Q5 fabrication, no rise in no-call, zero parse errors); configuration level, mean ± sd over seeds; epoch 2 fixed in advance; paired bootstrap | Fixes stage 7's blind spot and winner's curse |
| Scorer | v2.1 frozen; v1 always reported | Stage 8 |
| Test | Second use disclosed; frozen list | Stage 2 discipline |
| Probes | Counterfactual P1–P5 | Asks "does the model read the note?" without hand-written answers |

---

## 11. Wave 3 (RTX 4090; one seed)

**Goal.** Test the relabel policy and the baselines.

**Setup.**

- **P1 probes**: 34 frozen pairs, from 40 grounded BMI val candidates minus 6 whose question states the measurement. Each pair is weight and height removed (correct behaviour: abstain) plus the intact partner (correct behaviour: call).
- **Arms**:
  - C-filtered: 1,922 rows, 242 steps;
  - **F-relabel**: 2,000 rows, 250 steps;
  - R0-v1 / R0-v3 / R0-v3-FS4;
  - R0-8B zero-shot.

**Evidence: missing-input behaviour (headline).**

| Arm | P1 fabrication (in call / in text) | Intact partner valid call | Grounded tool tasks /55 | Natural Q5 fabrication /7 |
|---|---|---|---|---|
| R0-v1 | 0/34 | 20/34 | 29 | 0 |
| R0-v3 | 0/34 | **5/34** | **9** | 0 |
| R0-v3-FS4 | 2/34 | 21/34 | 29 | — |
| **C-filtered** | **25/34 (0 / 25)** | 33/34 | 54 | 4 |
| **F-relabel, epoch 2** | **0/34** (Wilson upper bound 10.15%) | **33/34** (same failure, val_105) | 54 | **0** |
| R0-8B | **28/34 (28 / 6)** | 25/34 | 46 | — |

What the table shows:

1. **Relabel solves the missing-input case without costing tool use.**
2. **Filter-only hallucinates in prose**; a call-only metric would miss all 25 cases.
3. **Prompting doesn't get there.** The strict prompt regressed tool compliance (no call on 32/40 BMI and 11/15 conversions). Few-shot recovered only BMI selection (33/40, of which 6 calls had wrong arguments; conversion 4/15; only one BMI demonstration).
4. **The base model's "zero fabrication" is under-calling.**
5. **The 8B base calls eagerly and invents arguments**, so scale alone is not grounding.

**Evidence: other metrics.**

| Metric | Result |
|---|---|
| v2.1 macro | R0-v1 73.8%, v3 67.0%, FS4 77.6%, C 91.4%, F ep1 94.4%, **F ep2 96.7%**, R0-8B 68.8% |
| F vs C, paired (exploratory) | v2.1 +5.28 pp [2.22, 8.67]; v1 +3.07 pp [0.25, 6.23] |
| **Numeric** (v2.1, C 40/50 → F 46/50) | Only **3 of the 6 apparent gains are credible**: val_028 (TIBC), val_114 (haematocrit), val_186 (bilirubin ranking). The rest are ambiguous (val_054), a false positive (val_062, self-contradiction), or a scorer preference (val_124). Further false passes: val_239 (an extra false ALT claim). Real errors remain: val_001, 085, 185, 202 (19.3 written for 119.3). **Numeric is unresolved** |
| Training | 42.7 / 43.9 min; 7.81 GiB |
| Teacher-forced val loss | **F worse (0.3470) than C (0.3414) while behaviour is better**, so CE is unsuitable for selection here |
| Train-fit numeric (v1) | C 32/50, F 34/50: scorer-dependent; no capacity conclusion |
| C10a answer-confidence AUROC | C 0.750, F 0.640: different error sets, so not "calibration got worse" |
| C10b call-prefix AUROC | 1.0 for both |
| Cost | FS4 uses 4,484 prompt tokens vs 1,294, yet ran **no slower** (846 s vs 1,001 s). No latency advantage is claimed for SFT |

**Decisions.**

- F epoch 2 is the best single-seed candidate.
- Keep v1 as the comparator prompt.
- Do not promote v3 or FS4.
- Numeric needs a claim-level audit.
- Do not claim "92% numeric" or "96.7% clinical accuracy".

---

## 12. Qwen3.5 round (A100; one seed)

**Goal.** Does the finding transfer across model families, and is a newer backbone better?

**Integration facts.**

- Hybrid Gated DeltaNet / full attention (3:1).
- XML tool calls; an empty think block in every assistant turn.
- LoRA extended to the linear-attention projections: 30.5M parameters (the original seven names would adapt only 8 of 32 layers).
- flash-linear-attention kernels.
- Vocabulary about 248k vs 152k. That adds only about 3 points to the lm_head share of per-token compute; the slowdown comes mainly from linear attention, longer sequences (XML plus the think block) and contention under `PARALLEL=1`.

**Evidence.**

| Arm | P1 fabrication (call / text) | Partner | Grounded tool | Natural Q5 abstain | Over-call | v1 numeric | Clinical context |
|---|---|---|---|---|---|---|---|
| R0-Q35-4B | 3/34 (3 / 0) | 30/34 | 53/55 | 5/7 | **50/188** | 17/50 | **12/55** |
| R0-Q35-9B | 14/34 | 29/34 | 48/55 | 3/7 | 38/188 | 23/50 | 25/49 |
| **A-Q35-4B filtered, epoch 2** | **29/34 (1 / 28)** | 33/34 | 54/55 | **2/7** | 0 | **34/50** | 53/55 |
| A-Q35-4B filtered, epoch 1 | (P1 not run) | — | 53/55 | **7/7** | 2 | 35/50 | 52/55 |

**Readings.**

1. **The filtered failure replicates on a different family, template, call format and GPU.** The model even writes "Using calculate_bmi with these values (weight_kg=108.5, height_cm=162.0)… BMI 41.4" **without calling**: a hallucinated tool call in prose. The pre-registered trigger for a Qwen3.5 relabel run was met.
2. **Fabrication grows with training**: natural Q5 abstention 7/7 at epoch 1 vs 2/7 at epoch 2.
3. **Zero-shot Qwen3.5 is a stronger tool caller** (53/55) but over-calls (50/188), abstains less, and rarely states the BMI category.
4. **Numeric looks better after SFT.** v1 34 vs 25/50, v2.1 0.96 vs 0.80, train-fit 38 vs 32. Both scorers agree here, but on the matched relabel comparison they disagree (stage 17); H2 is reported as unresolved (D-101).
5. **Cost**: 63.3 min and 11.3 GiB (A100, contention), so not a clean comparison.

Smoke-test note: the tool-argument smoke check failed on one imperial example (train_006) and was overridden; the override is recorded.

**Decision.** Focus on Qwen3.5 from here. F is frozen as the comparator; no more Qwen3 training.

---

## 13. CPU-only gap work (existing outputs; post-hoc diagnostics)

**Goal.** Close assignment coverage gaps without the GPU.

| Gap | What was done | Evidence |
|---|---|---|
| **C4: "brief clinical context"** | Deterministic check. The BMI's WHO category (from the executed result) or the converted value's status must be stated and agree with the input. Rules developed on **train gold** | Train gold passes 418/422 (3 of the failures are gold errors: 18.7–18.9 called "underweight"). SFT 54/55 (C and F); base R0-v1 23/31; R0-Q35-4B 12/55. The single SFT miss (val_125, 18.7 "underweight") copies the val gold's error |
| **A4: calibration** | Added ECE for the **call-prefix probability**, a genuine probability. Answer-token log-prob stays ranking-only (D-071) | SFT ECE about 0.002 vs base 0.07–0.19. **But C has ECE 0.0025 and still fabricates 25/34 in prose**: call calibration is not truthfulness |
| **A2: numeric** | Stratified the 50 items by whether the input-derived key agrees with gold | 34 agree / 0 dispute / 16 open or partial. **Known v2.1 false passes sit in the "agree" stratum** (extra false claims, contradictions), so gold cannot replace the claim-level audit |
| Seed variance | Needs GPU | Deferred, framed as "with more time" |

---

## 14. Stretch A: a third tool, `calculate_egfr`

**Why A and not B.** The evidence shows SFT learned the **in-distribution** two-tool policy, including when not to call. Stretch A tests whether that is a *transferable* tool-use policy, using the same grounding and abstention machinery. Stretch B (reference lookup) tests retrieval, a different capability.

**Data facts.**

- Age and sex are extractable from all notes.
- Creatinine is in the table for 350 / 40 / 82 records (train / val / test).
- **The table eGFR is inconsistent with CKD-EPI 2021** (median |Δ| 26–37; about 9% within ±5), so only records without a table eGFR are used: 164 / 19 / 31.

**Design.**

| Item | Choice |
|---|---|
| Tool | `calculate_egfr(creatinine_mg_dl, age, sex)`, CKD-EPI 2021, integer output |
| Prompt | v1e: v1 plus one sentence naming the tool, for symmetry, since v1 names the other tools |
| Arms | Zero-shot (schema only, schema + prompt, base) and A-sft (+52 template rows: 40 positives, 6 age-removed and 6 sex-removed abstentions) |
| Evaluation | 19 positives + 19 age-removed probes, plus a core regression check (including eGFR over-call where the table already has eGFR) |
| Pre-registered criteria | A-sft end-to-end ≥ 15/19; probe fabrication ≤ 2/19; core tool ≥ 53/55; P1 ≤ 1/34; over-call ≤ 1/21 |

**Review of the 19 + 19 items** (delegated AI review, not a clinician):

- independent re-extraction and an independent CKD-EPI implementation: 0/19 mechanical issues;
- example: val_000 → 2.5 mg/dL, 58, female → 21.75 → 22 → G4.

Fixes made during review:

1. The answer template was changed from "consistent with CKD stage" to "KDIGO GFR category …, assuming stable kidney function". Four notes describe acute or possibly acute kidney injury; val_009 is explicit AKI.
2. Grammatical repair after removing the age.

Tags: 4 non-steady-state, 1 near cut-off (44.91), 3 value-free age mentions. All 19 approved; hashes frozen.

---

## 15. Wave 4 design

**Hypotheses.**

| | Hypothesis | Test |
|---|---|---|
| **H1** | Relabel replicates on Qwen3.5 | A-Q35-relabel vs A-Q35-filter: P1, partner calls, natural Q5 |
| **H2** | With both on relabel data, Qwen3.5 beats Qwen3 on numeric | Claim-level audit of F vs Q35-relabel (2 × 50) |
| **H3** | Tool transfer (Stretch A) | 19 + 19 eGFR set |

**Decisions (Wave4-Q1 to Q5).**

1. **Epoch 2 is the fixed primary endpoint** (same rule as F). P1 is run on both epochs as a diagnostic. A failing epoch 2 is reported, never swapped for epoch 1.
2. **Gate**: F's frozen thresholds, but the control is the same-family filtered model, so the screen is isomorphic to F vs C. The cross-family comparison with F is report-only. Pass → Q35-relabel becomes the final candidate and the Stretch A backbone; fail → fall back to F.
3. **Hardware**: A100 with the same kernel as A-Q35-filter, `PARALLEL=0` for clean training timing. P1-RAW also on the A100, with the hardware difference disclosed.
4. **H2 rule**:
   - primary: Qwen3.5-only correct items outnumber F-only correct items, with an exact two-sided sign test p < 0.05 (e.g. 10 vs 2 discordant items gives p = 0.039);
   - non-inferiority: P1 ≤ 5%; grounded tool ≥ 52/55; partner calls ≥ 31/34; the other assignment metrics not all worse;
   - outcomes: "stronger" / "possibly stronger, insufficient evidence" / "no evidence".
5. **Order and stop rules**:
   - round 1 → score and review → audit → round 2 (Stretch A) → frozen test list → one test run;
   - a failed gate is reported as is;
   - a parity mismatch stops Stretch A;
   - if A-sft harms core, the final model stays tool-less;
   - test runs once.

**Supporting controls.**

- **P1-RAW**: the wave-1 raw model (same 2,000 rows, 250 steps, seed 42 as F; only the 78 labels differ) on the P1 probes. A label-only control.
- **Parity regeneration**: the Stretch A code changes `src/`, so A-Q35-filter epoch 2 is regenerated under the new code, with item-by-item equality required. The new code keeps the core tool schemas byte-identical (hash `9eb098db…` unchanged).

**Test (option B).** Frozen list: Q35-relabel, F, Q35-filter, R0-Q35-4B. Pre-specified metrics only. The natural-Q5 test confirmation has n = 10 and is underpowered; this is disclosed.

---

## 16. Wave 4 round 1 (A100; one seed)

**What was run.** A-Q35-relabel training (2,000 rows, 250 steps, `PARALLEL=0`), both epochs on val and P1, epoch-2 train-fit; P1 on A-Q35-filter epoch 1; P1-RAW; the parity regeneration of A-Q35-filter epoch 2.

**Operational note.** The first attempt stopped at step 2: the `w4` branch had been cut from a commit that lacked the wave-3 Qwen3.5 outputs, so `w3_epochs.py ckpt` found no manifest and `set -e` ended the script before the upload. Merging `master` into `w4` and re-running resumed every finished step.

**Evidence: H1 and controls.**

| Arm | P1 fabrication (call / text) | Partner valid call | Natural Q5 abstain | Grounded tool /55 | v2.1 macro |
|---|---|---|---|---|---|
| **Q35-relabel ep2** (primary) | **0/34** | 33/34 | **7/7** | 54 | **96.9** |
| Q35-relabel ep1 | 0/34 | 34/34 | 7/7 | 53 | 97.0 |
| Q35-filter ep2 (control) | 29/34 (1 / 28) | 33/34 | 2/7 | 54 | 95.3 |
| Q35-filter ep1 | **21/34 (1 / 20)** | 32/34 | 7/7 | 53 | 95.9 |
| **P1-RAW** (wave-1 raw, label-only control) | **32/34** (mostly in the call) | 33/34 | — | — | — |
| F-relabel ep2 (Qwen3) | 0/34 | 33/34 | 7/7 | 54 | 96.7 |

Readings:

1. **H1 holds.** The relabel effect replicates on Qwen3.5: 29/34 → 0/34, mirroring Qwen3's 25/34 → 0/34.
2. **P1-RAW removes the "more rows and steps" explanation.** It has the same 2,000 rows, 250 steps and seed as F, only the 78 labels differ, and it fabricates on 32/34 probes, almost all inside the tool call. The three Qwen3 arms span the full policy: raw labels fabricate in the call, filtered data fabricates in prose, relabelled data abstains.
3. **Filter fabrication is already present at epoch 1** (21/34), while natural Q5 still shows 7/7. Seven natural items cannot see it; the paired probes can.
4. **Parity passed:** 250/250 identical (raw outputs, calls, final answers, stop reasons). The Stretch A code did not change core behaviour.
5. Clinical context 54/55; call ECE 0.0016; train-fit tool and uncertain 50/50; 56.6 min training, 11.3 GiB.

**Gate (reviewed: pass, Harry, 2026-10-02).** Four of five screen checks pass. "No new valid-call failures on P1 partners" fails on **val_104**:

- the note gives 68.9 in (175.006 cm); the model passed 174.9 cm, 0.106 off, outside the ±0.05 argument tolerance;
- the BMI (21.1), category and context are correct, and v2.1 scores the item correct on outcome;
- it is an imperial-conversion drift, the class seen on test in wave 1, not a suppressed call;
- epoch 1 passes it (175.0); totals are equal (33/34 each; the control's failure is val_105, which the candidate passes).

The pre-registered rule says a failed screen goes to review, never to a moved threshold or another epoch. The review passed it, disclosing the strict-check failure and its cause (`configs/w4/gate_q35.json`).

**Evidence: full family comparison** (`reports/w4/r1/family_compare/TABLE.md`).

| | Qwen3 R0 | Qwen3 F ep2 | Qwen3.5 R0 | Qwen3.5 relabel ep2 |
|---|---|---|---|---|
| v2.1 extractive / numeric / tool / uncertain | 93 / 41 / 37 / 23 | 99 / 46 / 61 / 37 | 100 / 37 / 58 / 18 | 100 / 46 / 61 / 37 |
| v2.1 macro | 73.8 | 96.7 | 78.7 | 96.9 |
| v1 numeric | 40% | 58% | 34% | **72%** |
| Grounded tool /55 | 30 | 54 | **53** | 54 |
| Over-call /188 | 6 | 0 | **51** | 0 |
| Clinical context | 23/31 | 54/55 | **12/55** | 54/55 |
| Call ECE | 0.104 | 0.0018 | 0.203 | 0.0016 |

Paired, v2.1 macro (type-stratified bootstrap, 2,000 resamples):

| Comparison | Δ (pp) [95% CI] |
|---|---|
| F → Q35-relabel (same data, across families) | **+0.25 [−2.15, +3.10]** |
| C → Q35-filter | +3.85 [+0.25, +7.94], driven by numeric while Q5 fabrication is equally bad |
| Q3 R0 → Q35 R0 (zero-shot) | +4.93 [−1.86, +11.57] |
| Q35-filter → Q35-relabel | +1.67 [−0.69, +4.35] |

Readings:

1. **On the same relabel data, the families tie.** Safety metrics are identical. The data policy moves P1 by 25–29 items; the backbone moves it by 0.
2. **Zero-shot Qwen3.5 is the stronger tool caller but reckless**: over-calls 51/188, abstains less, rarely states the category, and its call ECE is 0.20. SFT removes all of it.
3. **Answer-confidence AUROC falls from epoch 1 to epoch 2 in both families** (0.748 → 0.640; 0.795 → 0.682): possible late-training overconfidence; diagnostic only (few errors).
4. **Best training per family:** Qwen3 F-relabel ep2 is best on every Qwen3 metric. Qwen3.5 relabel ep1 and ep2 are indistinguishable (97.0 vs 96.9); ep2 is reported because it is the pre-registered endpoint, not because it is better.

**Decision: later experiments use Qwen3 (F-s42), D-095.**

- **Why.** Qwen3.5 costs more at equal performance:
  - the XML tool-call format and the empty think block in every assistant turn make each conversation about **11% longer** (p50 1,525 vs 1,375 tokens; max 1,781 vs 1,614);
  - measured throughput is lower (1,793 vs 2,081 tokens/s);
  - peak memory is higher (11.3 vs 7.8 GiB);
  - it adds the flash-linear-attention dependency.

  The metrics show no material gap: +0.25 pp v2.1 macro, identical safety metrics, and H2 unresolved.
- **Caveats, disclosed.**
  - The speed comparison is confounded by hardware (Qwen3 on an RTX 4090, Qwen3.5 on an A100). The longer sequence is the confirmed cause; the linear-attention kernels are unprofiled.
  - The choice deviates from the pre-registered Wave4-Q2 rule ("gate pass → Qwen3.5") and was made after seeing round 1. The bias risk is low because it is a cost choice between metric-tied candidates, not selection on a favourable score.
- **Consequences.**
  - Stretch A runs on F: `stretch_a_backbone: q3` in `configs/w4/gate_q35.json`, read by `run_w4_round.sh`.
  - F is the final model in the frozen test list, with Q35-relabel kept as the cross-family comparator.
  - The Qwen3.5 results stay in the report as a replication of H1.
- **RTX 4090 note** (if Qwen3.5 is revisited): 11.3 GiB fits 24 GB; flash-linear-attention on Ada (sm_89) is expected to work but is unverified; run `make gpu-smoke` first and never compare timings across GPUs.

---

## 17. Scorer finding: where v1 and v2.1 misjudge numeric answers

**Goal.** Explain why v1 and v2.1 disagree so much on numeric (F vs Q35-relabel: v1 58% vs 72%, v2.1 46 vs 46 of 50).

**How each scores numeric.**

- **v1** (`metrics.py:235`) is **gold-driven**: every number the gold answer derives (not present in the input) must appear in the prediction at the gold's precision, plus direction words in the same sentence.
- **v2.1** (`scorer_v2.py`, `build_key` / `run_check`) is **question-driven**. It builds checks from the question's clauses and input-derived values (val numeric: 83 checks, 39 of them differences). A check passes if *any* number in the answer falls in its band. Only asked parts are checked; 15 of 50 val numeric keys are partial.

**Evidence: errors against the adjudicated reference** (numeric items where both rater groups agree):

| Set | v1 false fail / false pass / agree | v2.1 false fail / false pass / agree |
|---|---|---|
| Dev (tuned) | 20 / 0 / 25 | 0 / 0 / 45 |
| Holdout 1 (tuned) | 19 / 2 / 24 | 3 / 2 / 40 |
| **Holdout 2 (clean)** | **17 / 0 / 18 (51%)** | **0 / 1 / 34 (97%)** |

So **v1 is not adequate for numeric**: its errors are almost all false fails on correct answers worded differently from gold, for example fold instead of percent (val_047), or a gold's incidental comparison number (val_186). Qwen3.5's SFT outputs copy the gold template more closely, which inflates its v1 lead.

**Why v2.1 still over-credits.** It never checks the answer's *extra* claims, and contradicting numbers do not count against it. F ep2 has three false passes of this kind:

- val_117: "ferritin … below by 4.2"; the true difference is 5.2;
- val_194: "Creatinine is also elevated at 1.2", which is normal;
- val_062: "exceeds by 9 bpm (100 − 99 = 1)".

The adjudication rubric also did not require unrequested claims to be true, and the holdouts came from wave-1 models, whose errors sat in the asked part. **The scorer's error profile depends on the model being scored:** as models improve, the remaining errors move into extra claims that v2.1 cannot see.

An informal item-by-item reading of the 16 v1-discordant numeric items between F and Q35-relabel:

- about 7 real Qwen3.5 wins: val_001, 062, 085, 117, 185, 194, 202;
- 1 real F win: val_114;
- the rest are wording differences.

**Prototype (not adopted).** `scripts/scorer_v22.py` adds two guards to the frozen v2.1, numeric only and pass → fail only:

1. an abnormal-state claim must match the input;
2. a "by X" or "X above/below the limit" amount must equal |value − bound|.

Results:

- gold self-check 0/450 fires;
- 4 new disagreements with the adjudicated reference, all verified real errors the lenient rubric missed;
- on the validation arms, 6 flips, all verified real; no Qwen3.5 arm changed;
- F ep2 numeric 46 → 43; the Q35-relabel vs F sign test is 5 vs 2, p = 0.45.

**Decision.** Keep **v2.1 as the reported scorer**. Report this as a known limitation, with v2.2 as documented future work. Reasons: there is no clean holdout for v2.2, the guards were tuned on the same sets, and numeric conclusions rest on the claim-level audit anyway. Details: `reports/scorer_v2/VALIDATION.md` §5.

---

## 18. F′ refit and Stretch A (rounds 2 and 2b, RTX 4090)

**F′.** The wave-3 F-s42 adapter had never been uploaded, so F was retrained with an identical recipe as F′ (D-097).

| | F (wave 3) | F′ |
|---|---|---|
| P1 fabrication | 0/34 | 0/34 |
| Partner valid calls | 33/34 | 34/34 |
| Natural Q5 | 7/7 | 7/7 |
| Grounded tool | 54/55 | 54/55 |
| v2.1 macro | 96.7 | 95.0 |
| Identical validation items | — | 136/250 |

**Reading.** The safety findings reproduce, but this one same-recipe refit moved the macro by 1.7 pp (one observation, not a variance estimate; kernel nondeterminism is plausible but not isolated). Differences of 1–2 pp (including the +0.25 pp family comparison) are therefore not interpreted.

**First A-sft failed** its pre-registered criteria: end-to-end 9/19, age-probe fabrication 16/19, grounded tool 51/55. The diagnosis (D-100):

- **A threshold shift, not lost discrimination.** Probe call probability sat around 0.5 while the positives-vs-probes AUROC was 0.983. The causes were 40 positives against 6 age negatives, and question templates that presuppose age.
- **An unlearned KDIGO mapping.** G3a and G5 had 3 and 1 examples.
- **Core regression overlapping the refit differences**: three of four tool failures also occur in F or F′.

**A-sft2.** Prompt v1e2 (with the KDIGO table), 120 stage-balanced positives, 60 age negatives (40 paired), 20 sex negatives and neutral templates. Result: end-to-end 17/19, probe fabrication 0/19, P1 1/34, natural Q5 7/7.

The two core criteria still fail: grounded tool 51/55 (two v2.1 reader false fails, S-04) and discordance 9 against ≤ 5. Following the pre-registered rule, **F′ stays the core final model**, and A-sft2 is delivered as the Stretch A model with its core cost disclosed (D-101). Details: [STRETCH_A.md](STRETCH_A.md).

**Hindsight.** The first A-sft repeated the project's central lesson on a new tool: tool SFT with too few no-call examples teaches fabrication. The fix was the same kind of change as the Q5 relabel.

---

## 19. Confirmatory test run (RTX 4090; run once)

The five-model list was frozen and committed before any output (`configs/w4/final_test.json`; D-098, D-101), then run once at commit `7ae0034`. Report: [../reports/w4/test/TEST_REPORT.md](../reports/w4/test/TEST_REPORT.md).

| | F′ | A-sft2 | C-filtered | R0-v1 | Q35-relabel |
|---|---|---|---|---|---|
| Natural Q5 fabrication | **0/10** | 0/10 | **6/10** | 1/10 | 0/10 |
| Grounded tool tasks | **89/90** | 87/90 | 87/90 | 48/90 | 88/90 |
| v2.1 macro | **97.5** | 96.2 | 95.8 | 79.3 | 97.4 |

**Readings.**

1. **The headline replicates on test.** C-filtered fabricates on 6/10: four in prose (three of them claiming a calculate_bmi call that was never made), one through the call with the memorised height 178.5 cm, and one (test_288) that invents 50 kg and 150 cm and then says BMI cannot be calculated. v2.1 credits that last one as an abstention.
2. **F′ is the strongest core model.** Q35-relabel ties it (−0.09 pp), and A-sft2 is −1.28 pp [−2.66, −0.06]; two of its three tool failures are scorer false fails.

**Caveats.** Test was used before, so this is an exploratory confirmation, and n = 10 natural-Q5 items is underpowered. Fabrication is reduced, not eliminated: on test_020, an uncertain record outside the Q5 set, F′ invents a weight of 145.5 lb.

---

## 20. Current claims, limitations and next steps

**What can be claimed** (single seed; validation, and confirmed on a reused test split where noted):

1. Relabelling input-deficient tool records as explicit abstentions removes missing-measurement fabrication (25/34 → 0/34) without hurting tool use (54/55; partners 33/34).
2. Filtering alone moves the hallucination from tool arguments into prose, and the same happens on a second model family (29/34).
3. SFT beats prompting decisively on tool use (54/55 vs at best 29/55). Neither a stricter prompt nor an 8B base fixes grounding.
4. The validated scorer is far more reliable than the simple one (92.6% vs 65.7% on an unseen holdout).
5. The relabel effect replicates across model families (Qwen3.5: 29/34 → 0/34), and a label-only control (P1-RAW, 32/34) rules out row count and training length.
6. On the same data the two backbones tie (+0.25 pp [−2.15, +3.10] on val; −0.09 pp on test): the data policy matters far more than the newer base.
7. On test, relabel models abstain on 10/10 natural-Q5 items, while filter-only fabricates on 6/10.
8. A third tool can be added by SFT (17/19 end-to-end, 0/19 fabrication) once no-call examples are sufficient, at about 1 pp core cost.

**What cannot be claimed.**

- Numeric accuracy: v1 over-fails and v2.1 misses false extra claims (stage 17); the claim-level audit was not run.
- Qwen3.5 superiority on numeric (H2 unresolved).
- Population-level fabrication rates (0/34 has an upper bound of about 10%).
- Seed robustness.
- A clean test estimate (test is reused).

**Limitations.**

- One seed.
- Test reused (wave 1 plus scorer validation); val used adaptively.
- Adjudicators are LLM agents; the human check is informal.
- Both scorer parsing layers are regex, fitted to the templates; v2.1 checks only the asked part of numeric answers (stage 17).
- Explanatory commentary is unscored (a hallucination risk inherited from the gold style).
- Tool-result faithfulness is untested (copy vs recompute).
- Synthetic, template-generated data.
- Hardware and code-state differences between rounds.
- One same-recipe refit (F vs F′): 114/250 outputs and 1.7 pp of macro differ; not a variance estimate.

**Next steps** (with more time, in order of value):

1. More seeds.
2. A recorded human labelled set, then an LLM-parser + deterministic-verifier scorer, then human labels → a calibrated judge or reward model. Start from the v2.2 contradiction guards (stage 17) and validate them on a fresh holdout of current-model outputs.
3. A tampered-tool-result probe.
4. MiniCheck groundedness of the commentary.
5. Implicit-question augmentation (R-IMPL).
6. Visible-reasoning ablation (R-VIS).
7. bf16 LoRA vs QLoRA.
8. 8B SFT with a tuned lr.
9. A data-size curve.
10. Multi-call support (µmol/L → eGFR).
11. A claim-level numeric audit (H2), and the paired v2.2 guards validated on a fresh holdout.
