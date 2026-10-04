# Clinical QA fine-tuning: report

## 1. Summary and deliverables

This project fine-tunes a 4B instruction model to answer one clinical question from one encounter note plus one structured table. It covers the four required behaviours: fact extraction, simple numeric reasoning, deterministic tool use (`unit_convert`, `calculate_bmi`), and stating what is missing instead of inventing it. Stretch A adds a third tool, `calculate_egfr`.

The decisive finding came from the data. 78 training BMI labels call the tool with measurements that the input does not contain.

- Raw-label training reproduced unsupported arguments on the inspected missing-input cases.
- Filtering reduced unsupported calls, but fabrication remained in prose.
- Reviewed abstention targets reduced detected fabrication to zero on the measured P1 cohorts in both families, while retaining most valid calls. This does not establish zero fabrication outside those cohorts.

| Role | Model (run, checkpoint) | Tools / prompt / training rows | Adapter sha256 |
|---|---|---|---|
| **Core final model** | Qwen3.5-4B relabel (`w4_q35_4b_relabel_lr1e4`, 250) | 2 / v1 / 2,000 | `b28d7cc2…04a3ff` |
| **Stretch A model** | A-sft2-Q35 (`w4_q35_4b_relabel_egfr2_lr1e4`, 276) | 3 / v1e2 / 2,200 | `ec4141ff…19fea9` |

The recorded adapter repositories are `Harrydongyl/clinqa-<run>`. Confirm access before using the download route; anonymous availability was not verified at release review.

**About the backbone choice.** Qwen3.5 was selected on a qualitative reading of validation numeric answers and the observed extension behaviour, rather than a statistically established Core advantage. F′ remains a reported comparator. Section 4 states the evaluation limitation; the full decision history is in D-095, D-105 and D-106.

## 2. Task, data and the supervision policy

**Contract.**

- Inputs: the note, the complete table and the question. Questions sometimes state measurements themselves, so grounding includes the question.
- Supervision only: the gold answer, the answer type and the gold tool calls.
- Tool arguments are metric, so imperial notes need an implicit conversion.

**Data** (`reports/data_stats.json`):

| | Train | Val | Test |
|---|---|---|---|
| Records | 2,000 | 250 | 400 |
| Extractive / numeric / tool / uncertain | 800 / 400 / 500 / 300 | 100 / 50 / 62 / 38 | 160 / 80 / 100 / 60 |
| Mean note words | 274.5 | 273.7 | 274.9 |
| Labs / vitals tables | 1,393 / 607 | 184 / 66 | 288 / 112 |

The splits are stratified by type. The originals are never modified (`_data/` → `data/`, sha256-checked).

**The audit** ran 19 checks ([DATA_QUALITY.md](DATA_QUALITY.md)). Its key result is Q5: 95 of 534 BMI calls (train 78, val 7, test 10) use measurements absent from the input. Three train views were compared:

| View | Rows | What changes |
|---|---|---|
| Raw | 2,000 | Nothing |
| Filtered | 1,922 | The 78 removed |
| **Reviewed relabel** | 2,000 | The 78 kept, with an abstention target and no call (422 tool / 378 uncertain) |

Example (`train_009`): the gold "Using calculate_bmi with weight 88.8 kg and height 169.1 cm … BMI is 31.1" becomes "Weight and height are not documented, so BMI cannot be calculated from the available information." The review was AI-assisted grounding review, not clinician adjudication.

**Other findings:**

- Filtering leaves a single "both measurements missing" training example.
- Imperial heights do not round-trip to the gold centimetres (51 cases).
- 46 extractive golds cite standard ranges that the input does not supply (task design).
- The table eGFR is unrelated to creatinine (Spearman 0.008).

## 3. Fine-tuning setup

### 3.1 Baseline first

The first deliverable was a complete, runnable loop before any tuning:

- native-template formatting;
- assistant-only QLoRA SFT;
- a rollout that really executes tools;
- the five metrics.

The first SFT model raised test v1 macro from 0.44 (base) to 0.82. Subsequent iterations targeted named failure modes and metrics (3.3). Matched contrasts and bundled interventions are distinguished below.

### 3.2 Key choices

**Formatting** (`src/clinqa/formatting.py`):

- Qwen's native chat template and native tool-call format: JSON for Qwen3, XML for Qwen3.5, with the empty think block.
- The same tool schemas in every sample, so the model learns *when* to call.
- The user message is the note, then the table as Markdown, then the question.
- Tool records carry the gold call, its *executed* result in a `tool` message, then the answer.

**Loss.**

- Token-mean cross-entropy over assistant tokens only: the call, the answer and the end-of-turn marker. Tool observations and evidence are masked.
- Masks are built by prefix-difference rendering, because TRL's assistant mask does not support these templates. The audits pass on every conversation.
- Six rendered examples: `reports/w3/relabel_mask_audit/formatted_examples.md`.

**Recipe** (resolved config and manifest of the core model):

```text
Qwen/Qwen3.5-4B @ 851bf6e8, enable_thinking=false; NF4 4-bit base, double quantisation, BF16 compute
LoRA r16 / alpha 32 / dropout 0.05 on q,k,v,o,gate,up,down + in_proj_qkv,in_proj_z,out_proj (30,474,240 params)
LR 1e-4 cosine, 3% warmup (8 steps), 2 epochs = 250 steps, micro-batch 1 x accumulation 16, clip 1.0, wd 0
gradient checkpointing; Transformers Trainer + PEFT; loss normalised by supervised tokens across accumulation
```

| Choice | Alternatives considered | Evidence |
|---|---|---|
| Native template and Markdown table | Custom ChatML; JSON table | The native format preserves the model's tool interface; Markdown makes the evidence readable. The table formats were not compared in a controlled ablation |
| 4B instruction model | Qwen3-8B; Qwen3-4B vs Qwen3.5-4B | 8B zero-shot invents arguments (P1 28/34). On the same relabel view there is no clear Core separation (val +0.25 pp [−2.15, +3.10]); this is not an equivalence test. A validation numeric reading favours Qwen3.5 (about 7 vs 1 items; not significant), and its extension showed a smaller observed Core drop (section 6; one run per configuration) |
| QLoRA | Full fine-tuning; bf16 LoRA | Full fine-tuning and bf16 LoRA were estimated at about 64 GB and 13 GB; QLoRA measured 7.8–12.1 GiB. These are not matched measured memory comparisons. QLoRA fits one 24 GB GPU with headroom |
| LoRA on all projections, including linear attention | The seven standard names only | q/k/v/o target the 8 full-attention layers; gate/up/down already target MLPs throughout the decoder. The added names cover the linear-attention projections |
| LR 1e-4, 2 epochs, epoch 2 fixed in advance | 5e-5; 1.5e-4; 2e-4; epoch 1 | Observed LR differences were a few percentage points. At epoch 2, 2e-4 exceeded 1e-4 by 2.72 pp (unadjusted CI [+0.35, +5.53]); this single-seed, scorer-dependent result did not establish a robust optimum. We retained 1e-4 as a conservative control and fixed epoch 2 for later comparisons |
| Micro-batch 1 × accumulation 16 | 4 × 4 | Closely matched early loss traces, but 32% slower and 17.8 vs 7.8 GB |
| No truncation; fail-loud length guard | Truncating | Longest conversation 1,781 tokens (core) and 2,053 (Stretch A; guard raised to 2,560) |

### 3.3 Evaluation-driven iterations

| Question / intervention | Metric watched | Result | Decision |
|---|---|---|---|
| Raw vs Q5-filtered labels | Q5 abstention, v1 macro | Raw invents BMI inputs (6/7 val Q5), yet the first selection rule picked it | Replace the selection rule; rebuild the scorer |
| Scorer v1 → v2.1 (blinded adjudication) | Agreement with adjudication | 65.7% → 92.6% on an unseen holdout | Report both (section 4) |
| LR ladder; micro-batch 4; an inference-only prompt | v2.1 macro, speed | Modest LR differences with exploratory uncertainty; micro-batch 4 slower; the prompt was not established as a consistent improvement | Lock the recipe |
| Prompting instead of SFT (strict v3; four shots); 8B base | Grounded tool tasks, P1 | 9/55 and 29/55 vs SFT 54/55; 8B fabricates 28/34 | SFT, not prompting or scale |
| **Filter vs reviewed relabel** (paired missing-measurement probes, P1) | P1 fabrication, intact-partner calls | Qwen3 25/34 → 0/34; Qwen3.5 29/34 → 0/34; partners 33/34 | **Adopt relabel** |
| Label-only control (P1-RAW: raw labels, same rows and steps) | P1 | 32/34 | Same row count and steps alone did not reproduce the relabel outcome |
| Validation loss as a selector | Loss vs behaviour | Raw has the *lowest* loss (0.333) and the worst fabrication | Select on behaviour gates |
| Stretch A data, twice (section 6) | eGFR end-to-end, probe fabrication | 9/19 and 16/19 → 19/19 and 0/19 | Diagnose, then revise |

### 3.4 Stability and reproducibility

| Run | Train loss ep1 / ep2 | Val loss ep1 / ep2 | Grad norm median / max | Train time, peak memory |
|---|---|---|---|---|
| Qwen3.5 relabel (core) | 0.350 / 0.190 | 0.273 / 0.269 | 0.80 / 4.05 (step 1) | 56.6 min, 11.3 GiB (A100) |
| A-sft2-Q35 (Stretch A) | 0.316 / 0.171 | 0.278 / 0.271 | 0.77 / 3.19 | 87.0 min, 12.1 GiB (RTX 4090) |
| F′ (Qwen3 comparator) | 0.522 / 0.266 | 0.359 / 0.346 | 0.69 / 3.97 | 42.5 min, 7.8 GiB (RTX 4090) |

- No run had a non-finite loss; a stability check runs after every training run.
- Validation loss does not rise between epochs.
- A same-recipe refit (F vs F′) reproduced every safety result, but changed 114/250 outputs and 1.7 pp of macro. That is one observation, not a variance estimate.
- A parity regeneration after code changes gave 250/250 identical outputs.
- Every run records its code, data, prompt, template, schema and config hashes.

### 3.5 Inference

The rollout is a bounded state machine (`src/clinqa/evaluate.py`):

```text
generate -> strict parse (no repair) -> schema check -> real executor -> result to model -> answer
```

Decoding is greedy, batch 2, with at most one call, two assistant turns and 256 tokens per turn. Raw turns, calls, results and log-probabilities are logged.

## 4. Evaluation protocol

**The five assignment metrics** come from v1 (`src/clinqa/metrics.py`). v1 checks that the answer reproduces the gold's numbers and direction words, and checks tool selection and arguments within ±0.05.

**Diagnostic v2.1** (`src/clinqa/scorer_v2.py`, frozen) computes keys from the input before reading the answer, scores tools on the executed outcome, and expects abstention on Q5. Agreement with blinded adjudication rises from 65.7% (v1) to 92.6% (v2.1). Per answer type:

- **Extractive and numeric:** v1 fails correct answers worded differently from the gold. v2.1 fixes this but does not check extra claims.
- **Tool:** v1 rewards calling with invented Q5 arguments. v2.1 expects abstention there.
- **Uncertain:** largely unchanged.

Details: [docs/SCORER.md](../docs/SCORER.md#v1--v21-at-a-glance-by-answer-type); known defects S-01 to S-18: [SCORER_V2_1_KNOWN_ISSUES.md](../docs/SCORER_V2_1_KNOWN_ISSUES.md).

**Behaviour diagnostics:**

| Diagnostic | Definition |
|---|---|
| Grounded tool tasks | Tool records whose inputs support the call: 55 val, 90 test |
| Natural Q5 | The Q5 records, where abstention is correct: 7 val, 10 test |
| P1 probes | 34 val BMI items with both measurements removed (expected: abstain), each paired with its intact original (expected: call) |
| Over-call | Calls on non-tool records |

Fabrication counts come from a frozen classifier plus reading every case.

**Discipline.**

- Training uses only train-derived records; development comparisons and their criteria are recorded in the decision log.
- **Evaluation limitation:** these are reused-test comparisons, and the final backbone was chosen after test outputs were available; model selection was therefore not test-blind. See [D-105 and D-106](../docs/DECISIONS.md) for the chronology.
- Intervals are item bootstraps for fixed models (2,000 resamples, stratified by type). They exclude seed and scorer uncertainty.

## 5. Core results

### Test, required metrics (v1)

| Configuration | Extractive /160 | Numeric /80 | Tool selection /100 | Tool arguments /100 | Uncertainty /60 |
|---|---|---|---|---|---|
| Qwen3 base, prompt v1 | 56 | 20 | 52 | 50 | 46 |
| C: Qwen3 filtered | 152 | 44 | 91 | 85 | 60 |
| F′: Qwen3 relabel | 151 | 42 | 90 | 84 | 59 |
| **Qwen3.5 relabel (core final)** | **153** | **47** | 90 | **88** | 59 |
| A-sft2 (Qwen3, three tools) | 151 | 44 | 90 | 84 | 59 |
| **A-sft2-Q35 (Stretch A final)** | **155** | **48** | 90 | **88** | 58 |

The 100 tool labels include the 10 Q5 records, where abstention is correct, so the selection metric penalises the desired behaviour. C's 91 is one more unsupported call. Arguments given an attempted call: base 50/73, Qwen3.5 relabel 88/90.

### Test, diagnostic (v2.1) and behaviour

| Configuration | Ext /160 | Num /80 | Tool /100 | Unc /60 | Macro | Natural-Q5 fabrication | Grounded tool /90 (strict v1) |
|---|---|---|---|---|---|---|---|
| Qwen3 base | 157 | 71 | 57 | 44 | 79.3 | 1/10 | 48 (40) |
| C: Qwen3 filtered | 160 | 73 | 92 | 60 | 95.8 | **6/10** | 87 (84) |
| F′ | 160 | 74 | 99 | 59 | 97.5 | 0/10 | 89 (83) |
| **Qwen3.5 relabel** | 159 | 75 | 98 | 59 | **97.4** | **0/10** | 88 (87) |
| A-sft2-Q35 | 160 | 76 | 97 | 58 | 97.2 | 0/10 | 87 (88) |

The macro is an equal-weight four-type diagnostic, not the fraction of fully correct clinical answers.

**Readings:**

- **Missing inputs.**
  - C fabricates on 6/10 by the union of call and text checks; these categories overlap. Some answers claim a call that was never made. In test_288 it invents values and then declines; v2.1 credits the abstention (S-17).
  - No fabrication was detected in these ten Q5 cases for the relabel models; other cohorts still contain unsupported claims.
- **Tools.** The same-family Qwen3 base → F′ comparison gains 42 tool-policy and 15 uncertainty items, against three extractive and three numeric items (+18.16 pp macro). Qwen3.5 relabel − Qwen3 base is +18.1 pp [+14.1, +22.2], but that is a cross-family system comparison, not an isolated SFT effect.
- **Core model choice.** Qwen3.5 relabel has no clear separation from F′ (−0.09 pp [−1.8, +1.8]) or C (+1.55 pp [−0.46, +3.62], mostly Q5 policy). These intervals do not establish equivalence.
- **Residual errors.** Of Qwen3.5 relabel's nine v2.1 failures:

  | Kind | Items |
  |---|---|
  | Real errors | test_008 (an unsupported dosing claim beside a correct abstention); test_210 (a midpoint relation) |
  | Reader/scorer defects | Six: test_006, test_043, test_214, test_336, test_373, test_391 |
  | Ambiguous key | test_347 |

- **Not checked by v2.1.** False extra numeric claims are not scored. For example, F′'s test_330 is "a 48.2 deficit … a 48.2% reduction". Numeric accuracy is therefore not established.

### Validation evidence behind the policy

| Cohort | Filtered | Relabel | Intact-partner calls |
|---|---|---|---|
| P1, Qwen3 | 25/34, all in prose | 0/34 (F, F′) | 33 / 34 of 34 |
| P1, Qwen3.5 | 29/34 | **0/34** | 33 / 33 |
| Natural Q5 abstention, Qwen3.5 | 2/7 | 7/7 | — |
| P1, raw labels (P1-RAW) | 32/34 | — | 33 |

Zero-shot Qwen3.5 is a strong but reckless tool caller: grounded 53/55, but it over-calls on 51/188 and states the BMI category on only 12/55. Zero of 34 has a Wilson upper bound of 10.2%, and zero of 10 has one of 27.8%.

## 6. Stretch A: `calculate_egfr`

**The tool.** CKD-EPI 2021 (race-free) from creatinine (mg/dL), age and sex. It is checked against the NKF equation, and four implementations agree on 21,258 inputs. It returns an integer, a disclosed deviation from the assignment's `-> float`.

**Data.**

- Evaluation: 19 validation positives plus the same notes with the age removed, frozen before output and checked by AI with independent recomputation.
- Training: rows generated from train notes whose table has no eGFR (164 eligible).

Detail: [docs/STRETCH_A.md](../docs/STRETCH_A.md).

| Configuration (validation) | Complete answer /19 | Missing-age fabrication /19 |
|---|---|---|
| Qwen3 two-tool model, zero-shot, prompt v1e | 3 | 3 |
| Qwen3.5 relabel, zero-shot, prompt v1e2 | 7 | 1 |
| A-sft (Qwen3): 52 eGFR rows | 9 | 16 |
| A-sft2 (Qwen3): 200 rows, prompt v1e2 | 17 | 0 |
| **A-sft2-Q35**: the same 200 rows and prompt | **19** | **0** |

**All measured Stretch arms call correctly on the 19 positive cases** (selection, grounded arguments, execution and result reporting). The bottlenecks were the KDIGO category and abstaining without an age.

- **The first A-sft** invented ages on 16/19 probes (65 on 12 of 16). Call-prefix discrimination remained high (AUROC 0.983), consistent with an overly call-prone generation policy. Sparse age-negative supervision and age-presupposing templates are plausible contributors; neither was isolated, and no explicit binary decision threshold was deployed.
- **The revision** (D-100) changed several things at once: 60 age negatives (40 paired), stage-balanced positives, neutral templates, range-before-category answers and a KDIGO table in the prompt. The Q35 zero-shot v1e2 arm reaches 7/19 versus 19/19 for the trained extension under the same offered prompt/schema. This supports the training package comparison, but does not isolate the category table or individual data changes.

**Core cost of adding the tool** (Stretch model minus its own core model, v2.1 macro):

| | Validation | Test |
|---|---|---|
| Qwen3.5 | −0.8 pp | −0.20 pp [−1.76, +1.35] |
| Qwen3 | −2.4 pp | −1.28 pp [−2.66, −0.06] |

On identical Stretch data, A-sft2-Q35 − A-sft2 = +3.56 pp core macro on validation [+0.25, +7.10].

**Pre-registered gate** (D-105, fixed before output): every criterion was met except discordance with the Qwen3.5 core model over the 250 validation items.

- The result is 6 against a maximum of 5: A-sft2-Q35 is better on 2 items, the control on 4.
- The control's four wins are real errors: imperial drift on val_052 and val_105, a wrong entity on val_069, and unsupported measurement fabrication on val_204. In the last case, the input states BMI 39.3 without weight or height; the extension invents 100.5 kg and 1.75 m and derives BMI 32.8.

The gate is reported as failed. Per the pre-registered rule, A-sft2-Q35 is the Stretch A deliverable because its eGFR criteria are met. Its test results measure Core retention only, since test has no eGFR items.

## 7. Reproduction and resources

**Setup.** Python 3.11 with uv, pinned by `uv.lock`.

- `make setup`: GPU; torch 2.14, transformers 5.17, peft 0.21, bitsandbytes 0.50, flash-linear-attention 0.5.2.
- `make setup-cpu`: analysis only.
- `make test`: 277 unit tests (4 need the GPU packages and skip on CPU).

Pod setup: [docs/RUNBOOK.md](../docs/RUNBOOK.md).

```bash
make data analyze views-all
make w4-score OUT=reports/final_validation_review      # CPU: inspect shipped final evidence
uv run --frozen python -m clinqa.formatting --config configs/format_w4_q35_4b.yaml
make train RUN=reproduce_core_q35                     # GPU: fresh output identity
uv run --frozen python scripts/w3_epochs.py generate --run reproduce_core_q35 --config configs/eval_reproduce_core_q35.yaml
make score-v21 LABELS=reproduce_core_q35_step000250 OUT=reports/reproduce_core_v21
# Optional Stretch reproduction and probe scoring: docs/RUNBOOK.md
```

**Notes.**

- Completed run directories are never overwritten. The reproduction configs provide fresh names; for another rerun use a new identity ([README](../README.md#add-a-new-experiment)). The existing-adapter path requires verified access and revision.
- Test runs only through `scripts/w4_test.py` from a committed frozen list.

**Measured resources** (NF4/BF16, greedy, batch 2):

| Job | Time | Peak memory |
|---|---|---|
| Train the core model (2,000 rows, 250 steps) | 56.6 min (A100) | 11.3 GiB |
| Train the Stretch A model (2,200 rows, 276 steps) | 87.0 min (RTX 4090) | 12.1 GiB |
| Generate 400 test items | 31.5 min (core model, RTX 4090); 33.6 min (Stretch A model) | — |

- Qwen3.5 needs flash-linear-attention to import. `causal-conv1d` is not installed, so its short convolution uses the PyTorch path.
- Timings are not compared across GPUs.

**Artefacts.** `outputs/<run>/` holds training manifests and logs; `outputs/<label>/<split>/` holds trajectories, scores and `run.json` (index: `outputs/README.md`). Adapters are on the Hub. Manifests show `git.dirty: true` because outputs are written before the status is captured; provenance rests on the recorded hashes.

## 8. Limitations and next steps

1. **Backbone evidence.** The numeric advantage (H2) was not significant; small observed differences do not establish a superior Core backbone.
2. **Evaluation independence.** The limitation is stated in section 4. The Stretch A development items were also inspected during revision design.
3. **Scorers.** v1 has a surface bias, and v2.1 misses false extra claims (S-01, S-17, S-18). Numeric accuracy is unverified. The adjudicators were LLM agents.
4. **Small cohorts and one seed.** Natural Q5 has 10 test items, P1 34 and Stretch A 19. There is one seed and one refit (1.7 pp).
5. **Confounds.**
   - Relabel versus filter also changes row count and steps. The same-size P1-RAW control shows that matching these quantities alone does not reproduce the relabel result; it is not a multi-seed isolation of every training difference.
   - The Stretch revision bundles several changes.
   - Hardware differs between rounds (A100 and RTX 4090).
6. **Data and residual errors.** The data are synthetic, with remaining gold defects. Residual model errors: imperial-conversion drift, false extra numeric claims, unsupported statements (test_008), fabricated measurements outside Q5 (Stretch val_204), and implicit allergy questions.
7. **Provenance.** The recorded linear-attention kernel field does not establish whether flash-linear-attention ran (a recorder bug, fixed for future runs).

**Next steps:**

- seeds;
- a claim-level scorer validated on a fresh protocol (`scripts/scorer_v22.py` is a prototype);
- a deterministic KDIGO mapping and an imperial-conversion check after the tool call;
- isolating the Stretch A components;
- multi-call support for µmol/L creatinine.

RL is premature: target correction fixed the observed failure, and the reward signals have known defects.

## 9. Assignment requirements and where they are met

| Requirement | Where |
|---|---|
| **Data formatting pipeline**: JSONL to SFT chat format; note and table in the user message; tool-call format; answer-type distribution | `src/clinqa/formatting.py` (`python -m clinqa.formatting --config …` writes `data/sft_*/<view>/train.jsonl` plus sidecars); sections 2 and 3.2. The splits are stratified and every record is kept; relabelling changes the train type mix by design |
| **Fine-tuning script** (QLoRA; base model choice explained) | `src/clinqa/train.py` (Transformers + PEFT; `make train RUN=…`); section 3 |
| **Data quality analysis**: ≥ 5 formatted examples; statistics; issues and their handling | [DATA_QUALITY.md](DATA_QUALITY.md) (six examples), [data_analysis.md](data_analysis.md), [quality_flags.jsonl](quality_flags.jsonl) |
| **Evaluation script**: the five metrics per answer type, with limitations | `src/clinqa/evaluate.py` and `src/clinqa/metrics.py` (`metrics.json` per run); sections 4–5; [docs/SCORER.md](../docs/SCORER.md) |
| **Report**: setup, commands, hardware and runtime, artefacts, what was tried and why, results, limitations, next steps, ending with key findings | Sections 1–8 and 10 |
| **Reproducibility**: pinned versions, seeds, determinism | `uv.lock`; seed 42; greedy decoding; per-run hashes (section 3.4) |
| **Test not used in fine-tuning** | Train-only views; evaluation limitations disclosed in section 4 |
| **Runnable end to end** | Section 7; [docs/RUNBOOK.md](../docs/RUNBOOK.md); `make help` |
| **Stretch A**: a new tool, the pipeline and evaluation expanded, 10–20 annotated examples with results | `calculate_egfr` (`src/clinqa/tools.py`, `schemas.py`); 19 positives + 19 probes (`configs/w4/egfr_*.json`); section 6 |
| **Project structure** (`data/`, `src/`, `configs/`, `reports/`) | Followed; see `README.md` |

## 10. Key findings

1. Within Qwen3, SFT with native tool formatting and real execution improved grounded tool tasks from 48/90 to 89/90 and uncertainty from 44/60 to 59/60 (+18.16 pp diagnostic macro). The final Qwen3.5 Core configuration reaches 88/90 and 59/60; comparing it with Qwen3 base also changes the backbone.
2. Filtering left fabrication in prose. Reviewed abstention targets reduced detected fabrication on both P1 cohorts (25/34 and 29/34 → 0/34) and natural test Q5 (6/10 → 0/10), while retaining most useful calls. The same-size raw control supports a supervision-policy effect; these results do not establish zero fabrication elsewhere.
3. The observed grounding change from the data policy was larger than the measured Core separation between the two backbones. Qwen3.5 was adopted on qualitative numeric evidence and its smaller observed extension cost; neither establishes universal backbone superiority.
4. New-tool execution already worked on the measured positives; reliable interpretation and missing-argument handling improved with the extension package. The revised extension reaches 19/19 with 0/19 fabrication on Qwen3.5, at −0.2 pp core cost on test, with its narrowly failed core gate reported.
5. Evaluator design changes the apparent gains. Extraction and arithmetic improvements are much smaller than v1 suggests, and numeric accuracy remains unverified.
