# Confirmatory test runs (D-098, D-101, D-102, D-105, D-106): pre-specified report

## Integrity

- **The run.** It ran once from the frozen list `configs/w4/final_test.json`, committed at `7ae0034` before any output, on an RTX 4090 (NF4/BF16, greedy, batch 2).
- **Coverage and hashes.** All five arms cover exactly the 400 canonical test IDs. Every adapter, protocol, eval-config, frozen-config and split hash matches the freeze. There is no `RERUN.txt`.
- **`git.dirty: true` in the run files.** This is expected: generated files are written before the git status is captured, and the source and config hashes match the frozen protocol.
- **Earlier test outputs.** `outputs/base_test` and `outputs/raw_lr1e4_test` are the separate wave-1 test outputs.
- **Scoring.** CPU only, without regeneration: v1 from each `metrics.json`, and v2.1 with `scripts/score_v2.py --split test` using the scorer hash recorded in the protocols.

**Disclosure.**

- The canonical test was evaluated in wave 1 and used for scorer validation (v2.1 was tuned on holdout 1). These are frozen comparisons on a reused test set, not a fresh blind holdout.
- F′ and A-sft2 differ in prompt (v1 vs v1e2) and in tool list (two vs three tools). Their comparison is between configurations, not an ablation.
- The test has no eGFR items, so A-sft2 is measured here on core retention only.

**Final models (D-106).** Core: **Qwen3.5 relabel** epoch 2. Stretch A: **A-sft2-Q35** epoch 2. The first run's list (D-098) named F′ as the core model; the switch to Qwen3.5 came after this run (D-105) and is disclosed there. The second run (last section) is the A-sft2-Q35 core-retention check.

## Required metrics (v1)

| Configuration | Extractive /160 | Numeric /80 | Tool selection /100 | Tool arguments /100 | Uncertainty /60 |
|---|---|---|---|---|---|
| Qwen3 base, prompt v1 | 56 | 20 | 52 | 50 | 46 |
| C-filtered | 152 | 44 | 91 | 85 | 60 |
| **F′ (core final)** | 151 | 42 | 90 | 84 | 59 |
| Qwen3.5 relabel | 153 | 47 | 90 | 88 | 59 |
| A-sft2 (Stretch A) | 151 | 44 | 90 | 84 | 59 |

- **Arguments given an attempted call** (attempts include malformed calls): base 50/73, C 85/91, F′ 84/90, Qwen3.5 88/90, A-sft2 84/90.
- **The original 100 tool labels** are 90 supported calls plus 10 Q5 records where abstention is correct. The selection metric penalises those abstentions. C's 91 is one more unsupported call, not better routing.

## Diagnostic scores (v2.1)

| Configuration | Extractive /160 | Numeric /80 | Tool policy /100 | Uncertain /60 | Macro |
|---|---|---|---|---|---|
| Qwen3 base | 157 | 71 | 57 | 44 | 79.30% |
| C-filtered | 160 | 73 | 92 | 60 | 95.81% |
| **F′** | 160 | 74 | 99 | 59 | **97.46%** |
| Qwen3.5 relabel | 159 | 75 | 98 | 59 | 97.36% |
| A-sft2 | 159 | 72 | 97 | 59 | 96.18% |

The macro is an equal-weight diagnostic over the four types, not the fraction of fully correct clinical answers.

## Behaviour diagnostics

| Diagnostic | Base | C | **F′** | Qwen3.5 | A-sft2 |
|---|---|---|---|---|---|
| **Natural-Q5 fabrication /10** (frozen p1-2 classifier, every case read) | 1 | **6** | **0** | 0 | 0 |
| Grounded tool tasks /90, v2.1 | 48 | 87 | **89** | 88 | 87 |
| Grounded tool tasks /90, strict v1 end-to-end | 40 | 84 | 83 | 87 | 83 |
| Grounded requests with the right tool executed /90 | — | 90 | 90 | 90 | 90 |
| Clinical context on executed grounded calls (post-hoc) | 41/51 | 87/90 | 88/90 | 89/90 | 86/90 |
| Over-call /300; parse or schema errors | 6; 2 | 0; 0 | 0; 0 | 0; 0 | 0; 0 |

**Uncertainty.** Zero of 10 has a Wilson 95% upper bound of 27.8%, and C's 6/10 has an interval of [0.31, 0.83]. For C vs F′ on the ten paired items (6 vs 0), the exact McNemar test gives p = 0.031. This is unadjusted, and it does not repair the reused-test exposure.

**Strict vs tolerant tool scoring.** F′'s six strict-v1 grounded misses (test_045, 139, 168, 176, 311, 351) are small imperial-conversion drifts outside ±0.05 but inside v2.1's outcome tolerance. A seventh, test_261, reports 504 for a result of 503.99. Both definitions are reported.

## Paired differences

v2.1 macro, 2,000 bootstrap resamples stratified by type, seed 42. These are descriptive for fixed models; they include no seed or scorer uncertainty.

| Comparison | Δ macro (pp) [95% CI] | Notes |
|---|---|---|
| F′ − base | +18.16 [+14.21, +22.31] | Per type: +3 extractive, +3 numeric, +42 tool policy, +15 uncertain |
| F′ − C-filtered | +1.65 [−0.08, +3.40] | Almost entirely Q5 policy |
| A-sft2 − F′ | −1.28 [−2.66, −0.06] | Configuration trade-off; 2 of A-sft2's 3 tool failures are reader false fails |
| Qwen3.5 relabel − F′ | −0.09 [−1.82, +1.76] | No separation; no equivalence test |

## Item review

Every case below was read. Scores are frozen and are not repaired.

### Natural Q5

| Model | Fabrications | Cases |
|---|---|---|
| C-filtered | 6 | test_030, test_050, test_182 and test_386 invent values in prose with no call (three write "Using calculate_bmi with …"). test_388 calls with the memorised height 178.5 cm. test_288 invents 50 kg and 150 cm, then says BMI cannot be calculated; v2.1 credits it as an abstention (S-17), which is why v2.1 shows C at 5/10 |
| Base | 1 | test_291, through the call |

F′, Qwen3.5 relabel and A-sft2 abstain on all ten.

### F′'s eight v2.1 failures: real errors versus scorer defects

**Real model errors:**

- **test_020** (uncertain): neither weight nor height is documented. F′ invents a weight of 145.5 lb (A-sft2 invents 68.0 kg), while correctly refusing BMI. The gold is also defective: it asserts a height of 162.8 cm that the input lacks.
- **test_113**: aPTT chosen as most elevated, although INR's relative elevation (about 45%) exceeds aPTT's (19%).
- **test_210**: the midpoint of 12–300 computed as 150 (it is 156).
- **test_385**: correct creatinine excess, then a false claim that uric acid of 6.5 is 1.3 above a 7.2 limit.

**Scorer defects or ambiguous obligations:**

- **test_026**: conversion and direction are correct; "reduced renal perfusion" is read as creatinine low (S-04).
- **test_006** and **test_043**: correct all-normal answers missed by the reader; test_043's key is also partial (S-03).
- **test_347**: an ambiguous "closest to the upper limit". Total protein (95% of its limit) is closer than the gold's alkaline phosphatase (87%).

### False passes

F′ answers that v2.1 passes but that contain false extra claims (S-01, S-06):

- test_330: a 48.2 deficit then called "a 48.2% reduction";
- test_191: platelets of 119.5 called normal against 150–400;
- test_223: total cholesterol of 238.7 called normal against <200;
- test_290: HDL of 53.5 called "mildly low by 13.5" against >40;
- test_293: haemoglobin of 15.7 said to be "0.2 above" 12.0;
- test_391: SpO2 of 93 called normal and below 95;
- test_273: an incoherent percentage comparison.

This list is not exhaustive and is not converted into a corrected accuracy. Numeric accuracy is not established.

### Other noted cases

- A-sft2 test_285: 73.5 in converted to 176.5 cm (true 186.7), which changes the BMI category; both scorers catch it.
- Qwen3.5 test_008 abstains correctly but adds unsupported dosing claims.
- A-sft2 and Qwen3.5 test_336 name the right drug and fail the text fallback on recall (S-10).
- The post-hoc context check fixes the standard 18.5 BMI threshold. test_374 (BMI 18.9) has a gold that applies an elderly 18.5–20 interpretation, which is a reference-policy conflict. test_196 (BMI 29.6 called obese) is a clearer mapping error.

## Runtime

| | Base | C | F′ | A-sft2 | Qwen3.5 |
|---|---|---|---|---|---|
| Generation wall time, 400 items (min) | 23.2 | 33.6 | 32.2 | 32.8 | 31.5 |

- These are whole-job wall times, including different output lengths and call counts, not request latency.
- The generator does not reset its CUDA peak counter between sequential arms, so `peak_vram_gb` is not a per-model inference memory measure.
- Qwen3.5's `linear_attention_kernel` field records the Transformers implementation; use of the flash-linear-attention path is not verified.

## Readings

1. **The data-policy result replicates.** Relabel-trained models abstain on all ten natural-Q5 items in both families, while filter-only fabricates on six, four of them in prose.
2. **F′, the pre-registered core model at the time, is the strongest core configuration on most columns** (v2.1 macro 97.5, grounded tool 89/90), tied with Qwen3.5 relabel. It does not dominate every column (v1 numeric, strict v1 tools). It was chosen before test, from the validation evidence. The later switch of the final models to Qwen3.5 (D-105, D-106), which ties F′ here, is disclosed as a post-test decision.
3. **The gain over base is behavioural**: tool policy +42 items and uncertainty +15, against +3 each for extraction and numeric under v2.1.
4. **Fabrication is reduced, not eliminated** (test_020), and numeric answers contain false extra claims that both scorers miss.

## Second test use: A-sft2-Q35 (D-105, D-106)

**The run.** One model, `asft2_q35_test`, frozen in a separate list (`configs/w4/final_test_q35.json`, commit `724e707`) and run once on an RTX 4090 (33.6 min). It covers the 400 test IDs; the adapter and frozen-config hashes match, and there was no rerun. Scores: `reports/w4/test_q35/`.

**Disclosed.**

- This is the second use of test, and it happened after the first run's results were known. It measures core retention for the Stretch A model and was not used for selection.
- Test has no eGFR items.

| Metric | A-sft2-Q35 | Qwen3.5 relabel | A-sft2 (Qwen3) | F′ |
|---|---|---|---|---|
| Natural-Q5 fabrication /10 | **0** (ten clean abstentions) | 0 | 0 | 0 |
| v1 extractive / numeric | 155 / 48 | 153 / 47 | 151 / 44 | 151 / 42 |
| v1 tool selection / arguments / uncertainty | 90 / 88 / 58 | 90 / 88 / 59 | 90 / 84 / 59 | 90 / 84 / 59 |
| Grounded tool /90, v2.1 (strict v1) | 87 (88) | 88 (87) | 87 (83) | 89 (83) |
| Clinical context (post-hoc) | 88/90 | 89/90 | 86/90 | 88/90 |
| v2.1 macro | 97.2 | 97.4 | 96.2 | 97.5 |

**Paired v2.1 macro**, A-sft2-Q35 minus the other model (2,000 stratified bootstrap resamples, seed 42):

| Comparison | Δ (pp) [95% CI] |
|---|---|
| vs Qwen3.5 relabel | −0.20 [−1.76, +1.35] |
| vs A-sft2 (Qwen3) | +0.99 [−1.04, +3.04] |
| vs F′ | −0.29 [−2.21, +1.65] |

**A-sft2-Q35's nine v2.1 failures,** each read:

| Kind | Items |
|---|---|
| Reader false fails (S-04) | test_173, test_214 ("within the desirable range (<5.18)" read as low cholesterol); test_373 ("postprandial hyperglycemia" bound to fasting glucose) |
| Fabricated-value false fails (S-18) | test_163 and test_307 state the documented weight correctly (63.7 kg; 133.6 lb) and quote the standard BMI range 18.5–24.9, which is flagged as an invented value |
| Shared with F′ | test_006, test_043 (correct all-normal answers missed; partial keys) |
| Ambiguous | test_347 ("closest to the upper limit") |
| Real error | test_210 (computes the 156 midpoint, then calls 71.4 above it) |

**Reading.** Adding the third tool cost Qwen3.5 almost nothing on the core test (−0.2 pp against its own core model), while the Qwen3 extension cost −1.3 pp. A-sft2-Q35 has the highest v1 extractive and numeric counts of all arms.
