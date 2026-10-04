# Clinical QA fine-tuning: report

## 1. Summary and deliverables

This project fine-tunes a 4B instruction model to answer one clinical question from one encounter note plus one structured table. It covers the four required behaviours: fact extraction, simple numeric reasoning, deterministic tool use (`unit_convert`, `calculate_bmi`), and stating what is missing instead of inventing it. Stretch A adds a third tool, `calculate_egfr`.

The decisive finding came from the data. 78 training BMI labels call the tool with measurements that the input does not contain.

- Training on them teaches fabrication.
- Deleting them moves the fabrication into prose.
- Relabelling them as reviewed abstentions removes it, on two model families, with tool use intact.

| Role | Model (run, checkpoint) | Tools / prompt / training rows | Adapter sha256 |
|---|---|---|---|
| **Core final model** | Qwen3.5-4B relabel (`w4_q35_4b_relabel_lr1e4`, 250) | 2 / v1 / 2,000 | `b28d7cc2…04a3ff` |
| **Stretch A model** | A-sft2-Q35 (`w4_q35_4b_relabel_egfr2_lr1e4`, 276) | 3 / v1e2 / 2,200 | `ec4141ff…19fea9` |

The adapters are public on the Hugging Face Hub (`Harrydongyl/clinqa-<run>`).

**About the backbone choice.** It was made late, and the decision is disclosed in full (D-095, D-105, D-106).

- Until the first test run, the pre-registered core model was the Qwen3 relabel model **F′**.
- The final models moved to Qwen3.5 afterwards, on a manual reading of validation numeric answers. Because the test results were already known, an influence of test cannot be excluded.
- On test the two core models tie (−0.09 pp). F′ remains reported throughout.

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

The first SFT model raised test v1 macro from 0.44 (base) to 0.82. From there, each iteration changed one thing and was judged on a named metric (3.3).

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
| Native template and Markdown table | Custom ChatML; JSON table | The native format is the model's own tool interface. Table reading is not a bottleneck: base extractive is 157/160 on test under v2.1 |
| 4B instruction model | Qwen3-8B; Qwen3-4B vs Qwen3.5-4B | 8B zero-shot invents arguments (P1 28/34). On identical data Qwen3 and Qwen3.5 tie (val +0.25 pp [−2.15, +3.10]). A validation numeric reading favours Qwen3.5 (about 7 vs 1 items; not significant), and it absorbs the third tool with less core cost (section 6) |
| QLoRA | Full fine-tuning; bf16 LoRA | About 64 GB, about 13 GB, and 7.8–12.1 GiB measured. QLoRA fits one 24 GB GPU with headroom |
| LoRA on all projections, including linear attention | The seven standard names only | Those names reach only 8 of Qwen3.5's 32 layers |
| LR 1e-4, 2 epochs, epoch 2 fixed in advance | 5e-5; 1.5e-4; 2e-4; epoch 1 | All within noise (≤ 0.03 v2.1 macro). Epoch 1 vs 2 differ by 0.1–2.3 pp, in both directions. The endpoint was fixed before results and never moves |
| Micro-batch 1 × accumulation 16 | 4 × 4 | Identical loss curve, but 32% slower and 17.8 vs 7.8 GB |
| No truncation; fail-loud length guard | Truncating | Longest conversation 1,781 tokens (core) and 2,053 (Stretch A; guard raised to 2,560) |

### 3.3 Evaluation-driven iterations

| Change (one at a time) | Metric watched | Result | Decision |
|---|---|---|---|
| Raw vs Q5-filtered labels | Q5 abstention, v1 macro | Raw invents BMI inputs (6/7 val Q5), yet the first selection rule picked it | Replace the selection rule; rebuild the scorer |
| Scorer v1 → v2.1 (blinded adjudication) | Agreement with adjudication | 65.7% → 92.6% on an unseen holdout | Report both (section 4) |
| LR ladder; micro-batch 4; an inference-only prompt | v2.1 macro, speed | All within noise; micro-batch 4 slower; the prompt changed 0–2 items | Lock the recipe |
| Prompting instead of SFT (strict v3; four shots); 8B base | Grounded tool tasks, P1 | 9/55 and 29/55 vs SFT 54/55; 8B fabricates 28/34 | SFT, not prompting or scale |
| **Filter vs reviewed relabel** (paired missing-measurement probes, P1) | P1 fabrication, intact-partner calls | Qwen3 25/34 → 0/34; Qwen3.5 29/34 → 0/34; partners 33/34 | **Adopt relabel** |
| Label-only control (P1-RAW: raw labels, same rows and steps) | P1 | 32/34 | Rules out row count and training length |
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

- Every choice was made on validation, with written criteria (D-xxx).
- Test ran once from a list frozen before output (`configs/w4/final_test.json`). A second, separately frozen run covered A-sft2-Q35 only (D-105).
- Test was also used in wave 1 and for scorer validation, so all test numbers are reused-test comparisons.
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
  - C fabricates on 6/10, four of them in prose, and three of those claim a `calculate_bmi` call that was never made. One more (test_288) invents values and then declines; v2.1 credits it as an abstention (S-17).
  - The relabel models fabricate on none.
- **Tools.** The gain over base is behavioural: +41 tool-policy and +15 uncertainty items, against +2 extractive and +4 numeric. Qwen3.5 relabel − Qwen3 base = +18.1 pp [+14.1, +22.2].
- **Core model choice.** Qwen3.5 relabel ties F′ (−0.09 pp [−1.8, +1.8]) and edges C (+1.55 pp [−0.46, +3.62], mostly Q5 policy).
- **Residual errors.** Of Qwen3.5 relabel's nine v2.1 failures:

  | Kind | Items |
  |---|---|
  | Real errors | test_008 (an unsupported dosing claim beside a correct abstention); test_210 (a midpoint relation) |
  | Scorer defects (S-03, S-04, S-06, S-10) | Five |
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

**Every arm already calls correctly** (selection, grounded arguments, execution, result reported on 19/19). The bottlenecks were the KDIGO category and abstaining without an age.

- **The first A-sft** invented ages (65 on 12 of 16) because it saw 40 positives against 6 age negatives under templates that presuppose age. Discrimination stayed high (AUROC 0.983); the threshold moved.
- **The revision** (D-100) changed several things at once: 60 age negatives (40 paired), stage-balanced positives, neutral templates, range-before-category answers and a KDIGO table in the prompt. The zero-shot v1e2 arm (7/19) shows that the table alone explains only part of the gain.

**Core cost of adding the tool** (Stretch model minus its own core model, v2.1 macro):

| | Validation | Test |
|---|---|---|
| Qwen3.5 | −0.8 pp | −0.20 pp [−1.76, +1.35] |
| Qwen3 | −2.4 pp | −1.28 pp [−2.66, −0.06] |

On identical Stretch data, A-sft2-Q35 − A-sft2 = +3.56 pp core macro on validation [+0.25, +7.10].

**Pre-registered gate** (D-105, fixed before output): every criterion was met except discordance with the Qwen3.5 core model over the 250 validation items.

- The result is 6 against a maximum of 5: A-sft2-Q35 is better on 2 items, the control on 4.
- The control's four wins are real errors, three of them imperial-conversion drift.

The gate is reported as failed. Per the pre-registered rule, A-sft2-Q35 is the Stretch A deliverable because its eGFR criteria are met. Its test run is the second use of test and measures core retention only, since test has no eGFR items.

## 7. Reproduction and resources

**Setup.** Python 3.11 with uv, pinned by `uv.lock`.

- `make setup`: GPU; torch 2.14, transformers 5.17, peft 0.21, bitsandbytes 0.50, flash-linear-attention 0.5.2.
- `make setup-cpu`: analysis only.
- `make test`: 277 unit tests (4 need the GPU packages and skip on CPU).

Pod setup: [docs/RUNBOOK.md](../docs/RUNBOOK.md).

```bash
make data analyze views-all                           # CPU: data copy + checksums, quality report, train views
uv run --frozen python -m clinqa.formatting --config configs/format_w4_q35_4b.yaml   # SFT chat JSONL
make train RUN=w4_q35_4b_relabel_lr1e4                # GPU: core model (or STAGES=r1 bash scripts/run_w4_round.sh)
uv run --frozen python scripts/w3_epochs.py generate --run w4_q35_4b_relabel_lr1e4 --config configs/eval_w4_q35_4b.yaml
STAGES=r2c UPLOAD=1 bash scripts/run_w4_round.sh      # GPU: Stretch A model (zero-shot arm, training, evaluation)
uv run python scripts/score_v21_val.py --out reports/<new dir> --labels w4_q35_4b_relabel_lr1e4_step000250   # CPU
```

**Notes.**

- Completed run directories are never overwritten. Use a new run name ([README](../README.md#add-a-new-experiment)) or evaluate the published adapters.
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

1. **The backbone switch.** The final models moved to Qwen3.5 after the first test run (D-105). The rationale is validation-based, but the decision is not test-blind. The numeric advantage (H2) was not significant.
2. **Test reuse.** Test was reused: wave 1, scorer validation, and two frozen runs. The Stretch A evaluation items were inspected during the A-sft diagnosis.
3. **Scorers.** v1 has a surface bias, and v2.1 misses false extra claims (S-01, S-17, S-18). Numeric accuracy is unverified. The adjudicators were LLM agents.
4. **Small cohorts and one seed.** Natural Q5 has 10 test items, P1 34 and Stretch A 19. There is one seed and one refit (1.7 pp).
5. **Confounds.**
   - Relabel versus filter also changes the row count and steps; P1-RAW addresses this.
   - The Stretch revision bundles several changes.
   - Hardware differs between rounds (A100 and RTX 4090).
6. **Data and residual errors.** The data are synthetic, with remaining gold defects. Residual model errors: imperial-conversion drift, false extra numeric claims, occasional unsupported statements (test_008), and implicit allergy questions.
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
| **Test not used in fine-tuning** | Train-only views; test only through frozen lists (section 4) |
| **Runnable end to end** | Section 7; [docs/RUNBOOK.md](../docs/RUNBOOK.md); `make help` |
| **Stretch A**: a new tool, the pipeline and evaluation expanded, 10–20 annotated examples with results | `calculate_egfr` (`src/clinqa/tools.py`, `schemas.py`); 19 positives + 19 probes (`configs/w4/egfr_*.json`); section 6 |
| **Project structure** (`data/`, `src/`, `configs/`, `reports/`) | Followed; see `README.md` |

## 10. Key findings

1. SFT on a 4B model, with native tool formatting and real execution, substantially improved tool use and missing-input behaviour. On test: grounded tool tasks 48 → 88 of 90, uncertainty 44 → 59 of 60, and +18 pp diagnostic macro over base.
2. Removing contradictory tool labels did not teach abstention; it moved fabrication into prose. Reviewed abstention targets removed it on two model families (val P1 25/34 and 29/34 → 0/34; test natural Q5 6/10 → 0/10) while keeping useful calls. A label-only control rules out data size.
3. The backbone mattered less than the data policy: on identical data, Qwen3 and Qwen3.5 tie on the core task. Qwen3.5 was adopted late, on validation numeric reading and on its smaller cost when adding a tool; this choice is disclosed as post-test.
4. New-tool execution transferred immediately, but category interpretation and missing-argument abstention needed targeted supervision. The revised extension reaches 19/19 with 0/19 fabrication on Qwen3.5, at −0.2 pp core cost on test, with its narrowly failed core gate reported.
5. Evaluator design changes the apparent gains. Extraction and arithmetic improvements are much smaller than v1 suggests, and numeric accuracy remains unverified.
