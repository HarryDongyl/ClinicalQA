# Clinical QA fine-tuning: report

## 1. Summary and deliverables

This project fine-tunes a 4B instruction model to answer one clinical question from one encounter note plus one structured table. It covers the four required behaviours:

- extracting facts;
- simple numeric reasoning;
- deterministic tool use (`unit_convert`, `calculate_bmi`);
- stating what is missing instead of inventing it.

The final core configuration is assistant-only QLoRA on a pinned Qwen3-4B-Instruct-2507, with strict tool-call parsing, real tool execution and reproducible manifests.

**The decisive finding came from the data.** An audit found 78 training BMI targets whose weight and height appear nowhere in the input.

| Training labels | Fabrication on 34 paired missing-measurement probes | Calls kept on complete inputs |
|---|---|---|
| Unchanged | 32/34 | — |
| The 78 removed | 25/34, in prose | 33/34 |
| The 78 relabelled as reviewed abstentions | **0/34** | 33/34 or better |

On the frozen final test, the filter-only control fabricates on 6/10 natural missing-measurement items and the final model on 0/10. These sets are small and were exposed earlier, so this is not a deployment-safety guarantee.

**Stretch A** adds `calculate_egfr`. A revised model improves from 9/19 to 17/19 complete successes, and from 16/19 to 0/19 missing-age fabrications. It fails its pre-registered core-retention gate, so it is delivered separately.

**Deliverables:**

| Role | Run (checkpoint) | Tools / prompt / rows | Adapter sha256 |
|---|---|---|---|
| **Core final model, F′** | `w4_q3_refit_relabel_lr1e4_s42` (250) | 2 / v1 / 2,000 | `556eb573…3162c66` |
| **Stretch A model, A-sft2** | `w4_q3_relabel_egfr2_lr1e4` (276) | 3 / v1e2 / 2,200 | `bd88eb7a…6049a` |

The adapters are published on the Hugging Face Hub as `Harrydongyl/clinqa-<run>`. Pipeline code: `src/clinqa/` (formatting, training, evaluation, tools, scorers). Data-quality analysis: [DATA_QUALITY.md](DATA_QUALITY.md). Decision log: [docs/DECISIONS.md](../docs/DECISIONS.md) (D-001 to D-104).

"F" is the earlier wave-3 run with the same recipe. Its adapter was not preserved, so F′ retrains it. F's validation results remain historical evidence and are not attributed to F′.

## 2. Task, data and the supervision policy

**Contract.**

- The model may use the note, the complete table and the question. Some questions state measurements themselves, so grounding includes the question.
- The gold answer, answer type and gold tool calls are supervision only and are never shown at inference.
- Tool arguments are metric. Imperial notes require an implicit conversion.

**Data** (`reports/data_stats.json`):

| | Train | Val | Test |
|---|---|---|---|
| Records | 2,000 | 250 | 400 |
| Extractive / numeric / tool / uncertain | 800 / 400 / 500 / 300 | 100 / 50 / 62 / 38 | 160 / 80 / 100 / 60 |
| Mean note words | 274.5 | 273.7 | 274.9 |
| Labs / vitals tables | 1,393 / 607 | 184 / 66 | 288 / 112 |
| Tool records (BMI + conversion) | 500 (410 + 90) | 62 | 100 |

The provided splits are already stratified by answer type, and the original files are never modified (`_data/` → `data/`, sha256-checked).

**What the audit changed.** Nineteen checks ran over all splits ([DATA_QUALITY.md](DATA_QUALITY.md)).

- **The key flag is Q5.** 95 of 534 BMI calls (train 78, val 7, test 10) use measurements absent from the input. Training on them rewards fabrication.
- Three training views were compared:
  - **raw**: 2,000 records unchanged;
  - **filtered**: the 78 removed, 1,922 records;
  - **reviewed relabel**: the 78 kept, with an abstention target and no call (2,000 records: 422 tool, 378 uncertain).

Relabelling intentionally changes the training type mix. Example (`train_009`):

- Original gold: "Using calculate_bmi with weight 88.8 kg and height 169.1 cm, the patient's BMI is 31.1 …". No weight or height is documented anywhere.
- Reviewed target: "Weight and height are not documented, so BMI cannot be calculated from the available information."

The reviews were AI-assisted grounding reviews with recorded rationales (`configs/w3/q5_relabel_review.jsonl`), not clinician adjudication.

**Other findings that shape interpretation:**

- Imperial rounding: 51 heights do not round-trip to the gold centimetres.
- 119 notes state a BMI inconsistent with their own measurements.
- 46 extractive golds cite a standard reference range that the input does not supply.
- The table's eGFR is unrelated to creatinine (Spearman 0.008).
- After filtering, only one training example covers "both measurements missing". This is why filtering alone fails.

## 3. Fine-tuning and inference design

**Formatting** (`src/clinqa/formatting.py`):

- Each record is rendered with Qwen's native chat template and native `<tool_call>` format, and every sample offers the same tool schemas. The model learns *when* to call, not whether tools exist.
- The user message is `## Encounter note`, `## Table (<type>)` as Markdown (pipes escaped, empty units explicit), then `## Question`.
- Tool records are system, user, assistant call, the tool's *executed* result, then the final answer. The formatter executes every gold call and verifies its result.
- Six exact rendered examples, covering all four types plus a conversion and an imperial BMI: `reports/w3/relabel_mask_audit/formatted_examples.md`.

**Loss.** Token-mean causal cross-entropy over assistant tokens only: the call, the answer and the end-of-turn marker.

- System text, evidence, tool observations and padding are masked (label −100).
- Masks come from prefix-difference rendering. TRL's assistant-mask path does not support the stock Qwen3 template.
- The mask audit passed on 2,000/2,000 conversations, with 5.5% of tokens supervised.
- Sequences are measured after rendering (maximum 1,614 tokens against a 2,048 limit). An overlength record fails loudly instead of being truncated.

**Final recipe** (resolved F′ config and manifest):

```text
Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f17c01a7cc42f958dc650907174af0554, seed 42, max length 2048
NF4 4-bit frozen base with double quantisation, BF16 compute, SDPA attention
LoRA r16 / alpha 32 / dropout 0.05 on q,k,v,o,gate,up,down: 33,030,144 trainable parameters
LR 1e-4 cosine, 3% warmup (8 steps), 2 epochs = 250 steps, micro-batch 1 x accumulation 16
grad-norm clip 1.0, weight decay 0, gradient checkpointing; Transformers Trainer + PEFT (default optimiser,
adamw_torch_fused under the pinned versions); loss normalised by supervised tokens across accumulation
```

**Why these choices:**

- **QLoRA on a 4B instruction model** fits one 24 GB GPU with room for about 1.6k-token conversations (peak 7.8 GiB). Full fine-tuning would need about 64 GB. Rank, dropout and the target set were held fixed, not tuned.
- **LR 1e-4 and two epochs** is a conservative locked recipe, not a proven optimum (section 8). Epoch 2 is the pre-fixed endpoint, even when epoch 1 scores slightly higher on a new metric.
- **Micro-batch 1** was measured as the better choice. Micro-batch 4 × accumulation 4 had an identical loss curve but was 32% slower per step and used 17.8 vs 7.8 GB.

**Inference** (`src/clinqa/evaluate.py`) is a bounded state machine:

```text
note + table + question -> native template + tool schemas -> generate
  -> strict parse (malformed JSON is never repaired) -> schema validation -> real executor
  -> actual result returned to the model -> final answer -> separate behaviour checks
```

Decoding is greedy, batch 2, with at most one call, two assistant turns and 256 tokens per turn. Every trajectory logs raw turns, parsed calls, executor inputs and outputs, stop reasons and token log-probabilities.

Other backbones needed template work. Qwen3-8B needed segmented assistant turns. Qwen3.5 uses XML calls and needed LoRA targets on its linear-attention projections (30.5M parameters). Neither is on the final path.

## 4. Evaluation protocol

**The five assignment metrics come from scorer v1** (`src/clinqa/metrics.py`, legacy rules):

| Metric | v1 rule |
|---|---|
| Extractive accuracy | Gold numbers grounded in the input must reappear at gold precision, with matching direction words; otherwise token-F1 ≥ 0.5 |
| Numeric accuracy | The gold's derived numbers must reappear, with matching direction words |
| Tool selection | The gold tool is called |
| Tool arguments | Within ±0.05 of the metric gold |
| Uncertainty detection | An abstention phrase, the missing field named, and no fabricated value |

v1 is biased towards the gold's wording. Blinded adjudication agrees with it on only 65.7% of unseen items.

**Diagnostic v2.1** (`src/clinqa/scorer_v2.py`, frozen; [docs/SCORER.md](../docs/SCORER.md)):

- answer keys computed from the input before reading the prediction;
- typed checks with tolerances;
- negation-aware reading;
- tool calls scored on the executed outcome, with abstention expected where the input lacks the arguments.

It agrees 92.6% (κ 0.72) on an unseen adjudicated holdout. It still checks only the asked part of an answer and misses false extra claims ([known issues](../docs/SCORER_V2_1_KNOWN_ISSUES.md)).

**Behaviour diagnostics:**

| Diagnostic | Definition |
|---|---|
| Grounded tool tasks | Tool records whose inputs support the call: 55 val, 90 test |
| Natural Q5 | The Q5 records, where abstention is correct: 7 val, 10 test |
| P1 probes | 34 frozen pairs: a val BMI item with both measurements removed (expected: abstain), plus its intact partner (expected: call). Call and text fabrication are classified separately |
| Over-call / over-refusal | Calls on non-tool records; refusals on answerable records |

An abstention phrase does not prove grounding. Returning the executor's number does not prove the whole answer is correct.

**Selection discipline.**

- Train is used for training only. Every choice was made on validation, with criteria written before results (D-xxx).
- Test ran **once**, from a list frozen and committed before any output (`configs/w4/final_test.json`, commit `7ae0034`). Every adapter, protocol and config hash was checked, and there were no reruns.
- **The test split is not a fresh blind holdout.** It was evaluated in wave 1 and used for scorer validation. This is a frozen comparison on a reused test set.
- Paired intervals are descriptive bootstraps over items for fixed models (2,000 resamples, seed 42, stratified by type). They do not include training-seed or scorer uncertainty.

## 5. Core results

### Required metrics on test (v1, 400 items, frozen)

| Configuration | Extractive /160 | Numeric /80 | Tool selection /100 | Tool arguments /100 | Uncertainty /60 |
|---|---|---|---|---|---|
| Qwen3 base, prompt v1 | 56 | 20 | 52 | 50 | 46 |
| C-filtered (control) | 152 | 44 | 91 | 85 | 60 |
| **F′ (core final)** | 151 | 42 | 90 | 84 | 59 |
| Qwen3.5 relabel (comparator) | 153 | 47 | 90 | 88 | 59 |
| A-sft2 (Stretch A) | 151 | 44 | 90 | 84 | 59 |

- **Arguments conditional on an attempted call**: base 50/73, C 85/91, F′ 84/90, Qwen3.5 88/90, A-sft2 84/90.
- The 100 original tool labels include the 10 Q5 records. On those, abstention is correct, and the original selection metric penalises it. C's 91 versus F′'s 90 is therefore not better routing: C's extra call is the unsupported one.
- A-sft2 differs in prompt and tool list, so it is a configuration comparison, not an ablation.

### Diagnostic scores on test (v2.1)

| Configuration | Extractive /160 | Numeric /80 | Tool policy /100 | Uncertain /60 | Macro |
|---|---|---|---|---|---|
| Qwen3 base | 157 | 71 | 57 | 44 | 79.3% |
| C-filtered | 160 | 73 | 92 | 60 | 95.8% |
| **F′** | 160 | 74 | 99 | 59 | **97.5%** |
| Qwen3.5 relabel | 159 | 75 | 98 | 59 | 97.4% |
| A-sft2 | 159 | 72 | 97 | 59 | 96.2% |

The macro is an equal-weight diagnostic over the four types, not the fraction of fully correct clinical answers.

### By behaviour

- **Extraction.**
  - On the same outputs, base scores 56/160 under v1 and 157/160 under v2.1.
  - The large v1 gap is mostly a measurement artefact: the base model already reads tables, and F′ gains only three items under v2.1.
- **Numeric reasoning.**
  - F′ 74/80, against base 71 and C 73: a small gain. C and Qwen3.5 have higher v1 counts.
  - Real errors remain: a 12–300 midpoint computed as 150 (test_210), and mixed comparison scales (test_113).
  - Some v2.1 passes contain false extra claims. test_330 correctly gives a 48.2 deficit and then calls it a 48.2% reduction, where the true figure is about 80% against 60; test_191 calls platelets of 119.5 normal against 150–400.
  - Numeric accuracy is therefore not established. These cases are annotated, not used to rescore.
- **Tools.**
  - On the 90 supported requests, all three relabel-trained configurations select and execute the right tool on 90/90.
  - Grounded end-to-end success:

    | Scorer | Base | C | F′ | Qwen3.5 |
    |---|---|---|---|---|
    | Strict v1 (±0.05 arguments, exact displayed result) | 40 | 84 | 83 | 87 |
    | v2.1 (outcome tolerance) | 48 | 87 | 89 | 88 |

  - F′'s six strict-v1 misses are small imperial-conversion drifts. Its one v2.1 miss (test_026) is a reader false fail: "reduced renal perfusion" is bound to creatinine as low.
  - The gain over base is large and credible. Rankings among the SFT models depend on the scorer definition.
- **Uncertainty.**
  - Natural-Q5 fabrication: C 6/10, F′ 0/10. Exact McNemar on the paired items gives p = 0.031, unadjusted, on reused test.
  - C's sixth case (test_288) invents 50 kg and 150 cm, then says BMI cannot be calculated. v2.1 accepts the abstention phrase, which is why the fabrication check is reported separately.
  - Ordinary uncertainty is 59/60. F′'s one miss (test_020, an uncertain record outside Q5) invents a weight of 145.5 lb while correctly refusing BMI, so fabrication is reduced, not eliminated.

### Grounding versus utility, by cohort

| Cohort | Filtered | Relabelled | Intact-partner calls |
|---|---|---|---|
| Val P1, Qwen3 | C 25/34 (all in prose) | F 0/34; F′ 0/34 | 33 / 33 / 34 of 34 |
| Val P1, Qwen3.5 | 29/34 | 0/34 | 33 / 33 |
| Val P1, raw-label control (P1-RAW) | 32/34, mostly in the call | — | 33/34 |
| Test natural Q5 | C 6/10 | F′, Qwen3.5 relabel, A-sft2: 0/10 each | — |

These cohorts have different structures and are not pooled. Zero of 34 has a Wilson 95% upper bound of 10.2%, and zero of 10 has one of 27.8%.

### Selection and interpretation

| Comparison (v2.1 macro, test) | Δ (pp) [95% CI] |
|---|---|
| F′ − base | +18.2 [+14.2, +22.3]; per type: +3 extractive, +3 numeric, +42 tool, +15 uncertain items |
| F′ − C | +1.65 [−0.08, +3.40]; almost all from Q5 policy |
| Qwen3.5 relabel − F′ | −0.09 [−1.8, +1.8]; no separation, and no equivalence test |

F′ does not dominate every column. Its data policy and endpoint were fixed before test, from the validation grounding evidence; the test reports the consequences of that decision.

The Qwen3 backbone was chosen after round 1 for simplicity and integration. On identical data the two families tied on validation (+0.25 pp [−2.15, +3.10]), and Qwen3.5's XML calls and think blocks make sequences about 11% longer. This deviates from the earlier plan to adopt Qwen3.5 after its gate (D-095). It is not a measured speed advantage: the two families ran on different GPUs.

## 6. Stretch A: `calculate_egfr`

The tool computes CKD-EPI 2021 (race-free) from creatinine (mg/dL), age and sex. It is checked against the NKF equation, and four independent implementations agree on 21,258 inputs (`tests/test_egfr_formula.py`). It returns an integer, a disclosed deviation from the assignment's illustrative `-> float`.

**Data.** The evaluation set has 19 validation positives and the same 19 notes with the age removed, frozen before any output. Mechanical correctness was reviewed by AI with independent recomputation, not by a clinician. Training rows are template-generated from train notes; their sources are disjoint from the evaluation sources. Full detail: [docs/STRETCH_A.md](../docs/STRETCH_A.md).

| Configuration (validation) | Complete eGFR answer /19 | Missing-age fabrication /19 |
|---|---|---|
| F′ plus the new schema and prompt v1e (no training) | 3 | 3 |
| A-sft: core + 52 eGFR rows, prompt v1e | 9 | 16 |
| **A-sft2**: core + 200 rows, prompt v1e2 | **17** | **0** |

**Every arm already selects the tool, grounds the arguments, executes and reports the result on 19/19 positives.** The bottlenecks were interpreting the result (the KDIGO category) and abstaining when age is absent.

**The first A-sft failure.**

- It learned to call even without an age, inventing 65 on 12 of 16 fabrications. Call probability on probes moved to about 0.27–0.82, while separation from positives stayed high (AUROC 0.983).
- Its training data had 40 positives against 6 age negatives, with templates presupposing "age and sex".
- Its category mapping was poor; G3a and G5 had 3 and 1 examples.

**The revision (D-100)** changed several things at once:

- 60 age negatives (40 paired with a positive's note) and 20 sex negatives;
- coverage of rare categories (120 positives: G1 29, G2 29, G3a 13, G3b 15, G4 23, G5 11);
- neutral question templates;
- answers that state the range before the category;
- the KDIGO table in the prompt.

Without a zero-shot v1e2 control, the gain is attributed to this package as a whole, not to one component. Both remaining failures are the two G5 cases: eGFR 12 and 13 are written as "15–29 (G4)". This motivates deterministic category mapping in future work.

**The retention gate failed.** A-sft2 met the eGFR targets, P1 (1/34), natural Q5 (7/7) and over-call (0/21), but not:

- grounded core tools, 51/55 against the minimum of 52. Two of the four failures are reader false fails;
- discordance with F′ over all 250 core validation items, 9 against the maximum of 5 (A-sft2 2 better, F′ 7 better; 7 even after the two corrections).

The revised gate was fixed after the first attempt failed, but before A-sft2 ran. **F′ therefore stays the core model, and A-sft2 is the separately disclosed extension.** On test, A-sft2 − F′ = −1.28 pp [−2.66, −0.06]. The final test has no eGFR items, so it measures core retention, not new-tool generalisation.

## 7. Reproduction, artefacts and resources

**Setup.** Python 3.11 with uv; dependencies pinned in `uv.lock`, with `requirements.txt` exported from it.

- `make setup`: GPU environment (torch 2.14, transformers 5.17, peft 0.21, bitsandbytes 0.50).
- `make setup-cpu`: analysis only.
- `make test`: unit tests (273 pass on CPU; 4 need the GPU packages).

Pod setup and credentials: [docs/RUNBOOK.md](../docs/RUNBOOK.md).

```bash
make data analyze views-all            # copy + checksum the provided data, quality report, train views (CPU)
uv run --frozen python -m clinqa.formatting --config configs/format_w3.yaml   # SFT chat JSONL -> data/sft_w3/<view>/
make audit-masks                       # formatted examples + loss-mask audit (needs the tokenizer)
STAGES=refit UPLOAD=1 bash scripts/run_w4_round.sh   # GPU: train F′, evaluate both epochs on val, P1, train-fit
STAGES=r2b   UPLOAD=1 bash scripts/run_w4_round.sh   # GPU: train A-sft2, evaluate on val, P1, eGFR sets
uv run python scripts/score_v21_val.py --out reports/<new dir> --labels w4_q3_refit_relabel_lr1e4_s42_step000250
uv run python scripts/stretch_a_score.py score --out reports/<new dir> --labels w4_q3_relabel_egfr2_lr1e4_step000276
```

**Notes on the commands.**

- The single-step entry points are `make train RUN=<config>` (`python -m clinqa.train --config configs/train/<run>.yaml`) and `make eval`.
- The runners resume finished steps and refuse to overwrite completed run directories. For a fresh reproduction, use a new run name ([README](../README.md#add-a-new-experiment)) or evaluate the published adapters (README quickstart).
- Test was run through `scripts/w4_test.py freeze` and `run`. It must not be rerun.

**Measured resources** (RTX 4090 24 GB, NF4 / BF16, greedy, batch 2):

| Job | Wall time | Peak allocated |
|---|---|---|
| Train F′ (2,000 rows, 250 steps) | 42.5 min | 7.81 GiB |
| Train A-sft (2,052 rows, 258 steps) | 51.0 min | 8.21 GiB |
| Train A-sft2 (2,200 rows, 276 steps) | 59.7 min | 8.36 GiB |
| Generate 250 val items | About 20 min | — |
| Generate 400 test items | Base 23.2, C 33.6, F′ 32.2, A-sft2 32.8, Qwen3.5 31.5 min | — |

- Generation times are whole-job wall times, including different output lengths, not request latency.
- The generator does not reset its CUDA peak counter between sequential arms, so inference memory is not compared.
- Qwen3.5-4B training ran on an A100: 56.6 min, 11.3 GiB.

**Artefacts.**

| Path | Contents |
|---|---|
| `outputs/<run>/` | Training manifest (config, data, prompt, template and schema hashes, packages, hardware, adapter hashes) and training log |
| `outputs/<label>/<split>/` | `trajectories.jsonl`, `scored.jsonl` and `metrics.json` (v1), `run.json` (protocol and hashes) |
| `reports/` | Data quality, scorer validation (`scorer_v2/`), per-round scoring (`w3/`, `w4/`); final test: `reports/w4/test/TEST_REPORT.md` |
| `checkpoints/` | Adapters; git-ignored and published on the Hub |

**Provenance.** Run manifests record `git.dirty: true`, because generated files are written before the status is captured. Provenance therefore rests on the recorded source, config and artefact hashes, which match the frozen protocol.

## 8. What was tried, and what worked

| Tried | Result | Decision |
|---|---|---|
| Raw vs Q5-filtered data (wave 1) | SFT lifts test v1 macro from 0.44 to 0.82, but the pre-registered rule picked a model that invents BMI inputs (6/7 val Q5) | Selection rule replaced; data policy studied |
| Rebuilding the scorer (v2.0 → v2.1), blinded adjudication | v2.0 overfit (76.7% on a clean holdout); v2.1 92.6% | v2.1 reported beside v1 |
| LR ladder 1e-4 / 1.5e-4 / 2e-4 (filtered; v2.1 macro ep1 / ep2) | 0.936 / 0.914, 0.931 / 0.921, 0.908 / 0.941: inside noise | LR locked at 1e-4 |
| Inference-only prompt v2 ("use supplied ranges only") | 0–2 items changed; contradicts 36 correct golds | Not adopted |
| Strict prompt v3, and v3 with four shots (no SFT) | Grounded tool 9/55 and 29/55, against SFT 54/55 | SFT, not prompting |
| Qwen3-8B zero-shot | 46/55 grounded, but 28/34 P1 fabrication | Scale alone does not ground |
| **Reviewed relabel vs filter** | P1 0/34 vs 25/34; tools 54/55 each | **Adopted** |
| Qwen3.5-4B filter vs relabel | 29/34 vs 0/34; ties Qwen3 on identical data | Replication; Qwen3 kept |
| P1-RAW (raw labels; same rows and steps as F) | 32/34 | Rules out row count and step count |
| Validation loss as a selector | F has *higher* loss than C (0.347 vs 0.341) yet behaves better; A-sft2's loss falls while core macro falls | Behaviour-specific gates |
| Stretch A: A-sft, then A-sft2 | Section 6 | A-sft2 as a separate extension |

**Worked:**

- reviewed abstention targets;
- the native tool format with real execution;
- paired counterfactual probes, which caught prose fabrication that call-level metrics and the seven natural Q5 items missed;
- input-derived answer keys;
- pre-registered selection, parity regeneration after code changes, and a frozen single test run.

**Did not work:**

- filtering alone;
- prompting for tool use;
- learning-rate tuning;
- micro-batch 4;
- an 8B base without SFT;
- teacher-forced loss as a selector;
- the first Stretch A data mix;
- our own first rebuilt scorer.

## 9. Limitations and next steps

**Limitations.**

1. **Scorers.** v1 has a surface-form bias. v2.1 has incomplete claim coverage and reader errors. Neither macro is full-answer accuracy.
2. **Reuse.** Test was reused. The Stretch A evaluation items were inspected while designing A-sft2.
3. **Small cohorts.** Q5 has 10 items, P1 34 and Stretch A 19. Zero observed failures still allows large true rates.
4. **One seed.** There is one same-recipe refit: F vs F′ differ on 114/250 validation outputs and by 1.7 pp of macro. This is an observation, not a variance estimate. Seeds 43 and 44 were not run.
5. **Confounded comparisons.**
   - Relabel versus filter also changes the row count and steps (2,000 vs 1,922).
   - A-sft2 changes data, prompt, templates and steps together.
   - The family comparison changes template, call format and LoRA targets.
6. **Data.** It is synthetic, with remaining gold defects (val_125's BMI category; test_020's gold asserts a height the note lacks). There is no clinician validation.
7. **Residual failures:**
   - imperial-conversion drift;
   - a correct calculation with a wrong category;
   - false extra numeric claims;
   - residual weight fabrication (test_020);
   - implicit allergy questions (8/18 on val).

**Next steps:**

- seeds and broader missing-input probes;
- a claim-level scorer validated on a fresh development and evaluation protocol (`scripts/scorer_v22.py` is a prototype);
- isolating the Stretch A components;
- deterministic post-tool category mapping;
- multi-call support for µmol/L creatinine.

Reinforcement learning is premature. Target correction fixed the observed failure, and the current reward signals have known defects.

## 10. Assignment requirements and where they are met

| Requirement | Where |
|---|---|
| **Data formatting pipeline**: provided JSONL to SFT chat format; note and table serialised into the user message; tool-call examples; answer-type distribution preserved | `src/clinqa/formatting.py` (`python -m clinqa.formatting --config configs/format_w3.yaml` writes `data/sft_w3/<view>/train.jsonl` plus sidecars); section 3. The splits are stratified and every record is kept; the relabel view's change of type mix is disclosed in section 2 |
| **Fine-tuning script** (LoRA/QLoRA; base model choice explained) | `src/clinqa/train.py` (Transformers + PEFT; `make train RUN=…`); section 3 |
| **Data quality analysis**: ≥ 5 formatted examples showing exact input and output; counts per split, answer-type distribution, mean note length, tool-call frequency, table types; issues and their handling | [DATA_QUALITY.md](DATA_QUALITY.md) (six examples), [data_analysis.md](data_analysis.md), [quality_flags.jsonl](quality_flags.jsonl) |
| **Evaluation script**: extractive, numeric, tool-selection, tool-argument and uncertainty-detection metrics, with limitations explained | `src/clinqa/evaluate.py` and `src/clinqa/metrics.py` (`metrics.json` per run); sections 4–5; [docs/SCORER.md](../docs/SCORER.md) and [docs/SCORER_V2_1_KNOWN_ISSUES.md](../docs/SCORER_V2_1_KNOWN_ISSUES.md) |
| **Report**: setup, end-to-end commands, hardware and runtime, artefact locations, what was tried, what worked and why, results, limitations, next steps, ending with key findings | Sections 1–9 and 11 |
| **Reproducibility**: pinned versions, documented seeds, determinism where feasible | `uv.lock` and `requirements.txt`; seed 42; greedy decoding; per-run hashes in manifests (section 7) |
| **Test not used in fine-tuning** | Train-only views (`data/processed/`); evaluation formatting covers val only; test ran once from a frozen list (section 4) |
| **Runnable end to end** | Section 7 commands; [docs/RUNBOOK.md](../docs/RUNBOOK.md); `make help` |
| **Stretch A**: one additional tool; pipeline and evaluation expanded; 10–20 new annotated examples with results | `calculate_egfr` in `src/clinqa/tools.py` and `schemas.py`; 19 positives + 19 age-removed probes (`configs/w4/egfr_*.json`); section 6; [docs/STRETCH_A.md](../docs/STRETCH_A.md) |
| **Suggested project structure** (`data/`, `src/`, `configs/`, `reports/`) | Followed; see `README.md` |

## 11. Key findings

1. SFT on a 4B model, with native tool formatting and real execution, substantially improved tool use and missing-input behaviour (test grounded tool tasks 48 → 89 of 90; uncertainty 44 → 59 of 60).
2. Removing contradictory tool labels did not teach the model to abstain; it moved fabrication into prose. Explicit reviewed abstention targets reduced observed fabrication (val P1 25/34 → 0/34; test natural Q5 6/10 → 0/10) while keeping useful calls. The effect replicated on a second model family.
3. Evaluator design changes the apparent gains. Extraction and arithmetic improvements are much smaller than v1 suggests, and numeric accuracy remains unverified.
4. New-tool execution transferred immediately, but interpretation and missing-argument abstention needed targeted supervision. The revision fixed both (17/19 complete, 0/19 fabrication) yet failed core retention, so it is delivered separately from the core model.
5. The core and Stretch A configurations were frozen before a single reported test run, with all automatic failures and exposure caveats kept visible.
