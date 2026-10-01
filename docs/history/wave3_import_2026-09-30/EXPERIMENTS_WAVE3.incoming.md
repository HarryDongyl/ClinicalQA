# Wave 3 experiment plan

Implements the decisions D-037 to D-052 ([DECISIONS.md](DECISIONS.md)); the reasoning behind them is in [ROUND4_FINDINGS.md](ROUND4_FINDINGS.md). **Status: planned. None of the code prerequisites in section 4 exist yet.**

## 1. Fixed for every training run (D-037, D-038, D-039, D-041)

| Item | Value |
|---|---|
| Base model | Qwen/Qwen3-4B-Instruct-2507 @ `cdbee75f…`, NF4 QLoRA |
| LoRA | r16, α32, dropout 0.05, all 7 linear projections |
| Optimisation | lr 1e-4, cosine schedule, 3% warmup, 2 epochs, micro-batch 1 × accumulation 16, weight decay 0, max grad norm 1.0 |
| Loss | token-mean cross-entropy over supervised assistant tokens (HF `num_items_in_batch`); no weighting |
| Prompt and format | `system_v1`, native Qwen template, identical `TOOL_SCHEMAS`, empty call turn (except R-VIS) |
| Train view | **`q5_relabeled`** (new, D-043): the 2,000 canonical records, with the 78 Q5 records relabelled as uncertain using the two-part answer. Resulting mix: 800 / 400 / 422 / 378 |
| Checkpoint | epoch 2 (step 250), fixed in advance; epoch 1 is saved as a diagnostic only |
| Seeds | 42 for the control; 43 and 44 for the extra final-configuration seeds |

## 2. Runs

| ID | Type | Seed | Differs from F-s42 by | Question it answers | Val | Test |
|---|---|---|---|---|---|---|
| **F-s42** | final configuration (also the control) | 42 | — | Does relabelling Q5 remove fabrication at the locked lr? | ✓ | ✓ |
| **F-s43** | final configuration | 43 | seed | seed variance | ✓ | ✓ |
| **F-s44** | final configuration | 44 | seed | seed variance | ✓ | ✓ |
| **A-VIS** | ablation (D-042) | 42 | `Working:` + `Answer:` numeric targets built from input-derived keys (only where every check agrees with gold); one "≈" conversion line on imperial call turns | Does showing the comparison fix numeric errors and imperial drift? | ✓ | ✗ |
| **A-IMPL** | ablation (D-045) | 42 | 25 of the 51 explicit-allergy uncertain questions rewritten into implicit form; gold unchanged; count unchanged | Does question-form coverage remove the allergy shortcut? | ✓ | ✗ |
| **R0-v1** | baseline, inference only | — | no adapter, `system_v1` | parity baseline | ✓ | ✓ |
| **R0-v3** | baseline, inference only | — | no adapter, strong prompt `system_v3_r0` (D-040) | prompt-engineering ceiling | ✓ | ✓ |

The existing wave-1 and wave-2 runs remain reference points; none is retrained. The closest earlier configuration is `w2_filtered_lr1e4_mb1` (filtered view, same hyperparameters). Comparing F-s42 with it isolates the effect of relabelling versus filtering; use the same seed where possible.

**Cost estimate** (RTX 4090 or A10G/L4-class):

| Work | Estimate |
|---|---|
| Training | 5 runs × about 44 min ≈ 3.7 h (A-VIS about +10% for longer targets) |
| Val evaluation | 7 models × about 10 min at eval batch 2 (roughly 3 min at batch 8) |
| Test evaluation | 5 models × about 16 min |
| Probes | 6 models × 40 pairs, a few minutes each |

## 3. Evaluation for every run

1. **Scorers:**
   - v2.1, frozen (sha256 in `reports/scorer_v2/scorer_v2.1.sha256`); for A-VIS it reads only the `Answer:` part;
   - v1, reported in parallel (D-048).
2. **Main table (D-051):** the five assignment metrics under their spec names and denominators:
   - tool selection: over all grounded tool records;
   - tool arguments: over calls, reported both strict (±0.05) and by outcome.
3. **Diagnostics:**
   - tool end-to-end;
   - Q5 abstention;
   - **Q5 fabrication, split into in-call and in-text** (new metric: no call, but a weight/height/BMI value not present in the input);
   - over-call and over-refusal;
   - self-correction rate.

   Slices:
   - allergy questions, explicit vs implicit;
   - "most abnormal" questions with matching vs mixed units;
   - imperial vs metric tool records.
4. **Probes (D-050):** P1 remove a measurement, P2 add one, P3 metric → imperial, P4 change a value, P5 drop "allergy" from the question. About 8 pairs each, derived from val. Run on F-s42/43/44, A-IMPL, R0-v1 and R0-v3. Diagnostic only.
5. **Statistics:**
   - Wilson 95% CI per run;
   - mean ± sd across the 3 F seeds;
   - same-seed paired bootstrap (n = 2000, seed 42) for each ablation against F-s42.

## 4. Code prerequisites (not implemented; code is frozen until approved)

| # | Work | Touches |
|---|---|---|
| C1 | `q5_relabeled` train view: generate the two-part uncertain answers from note facts, with a review file | `data_views.py`, `configs/` |
| C2 | Integrate v2.1 into `evaluate.py`; `make eval` writes v1 and v2.1; add the spec-denominator metric block and the fabrication-in-text metric | `evaluate.py`, `scripts/score_v2.py` |
| C3 | D-047 selection: v2.1 macro plus gates, configuration-level mean ± sd, fixed epoch 2 | `evaluate.py`, `configs/eval_core.yaml` |
| C4 | R-VIS: `rationale.py`; `formatting.py` reasoning switch; call-turn content; `call_logprob` at the `<tool_call>` position; `Answer:` extraction in scoring; tests; mask audit | `formatting.py`, `infer.py`, new module |
| C5 | R-IMPL: rewrite explicit allergy questions (deterministic template, 25 fixed IDs, recorded in a manifest) | `data_views.py` |
| C6 | `configs/prompts/system_v3_r0.txt` and R0-v3 eval entries | `configs/` |
| C7 | Probe generator and probe scoring | new `scripts/probes.py` |
| C8 | Slice metrics | `scripts/score_v2.py` |
| C9 | Seed override for training configs (`seed: 43/44`) | `configs/train/*.yaml` |

Before any GPU run, the checks from D-041 must pass: `make audit-masks` on each new view, and token lengths ≤2048.

## 5. Order of execution

1. C1, C2, C3, C6, C9. Run the CPU tests and mask audit for `q5_relabeled`.
2. Train **F-s42** and evaluate it on val. Gate check (below). If it fails, stop and review before spending more GPU time.
3. Train F-s43 and F-s44, then evaluate on val.
4. C4, C5, C7, C8. Train A-VIS and A-IMPL, evaluate on val, then run probes on all models.
5. Evaluate R0-v1 and R0-v3 on val.
6. Freeze `configs/final_eval.yaml` (F-s42/43/44, R0-v1, R0-v3) and commit it. Run **test once** (D-049).
7. Write `reports/DATA_QUALITY.md` and update `reports/REPORT.md`.

## 6. Acceptance criteria and decision rules

| Check | Rule | Applies to |
|---|---|---|
| Q5 fabrication (in call + in text) | ≤1/7 on val, per seed | F-s42/43/44 (gate) |
| Grounded tool no-call rate | not above `w2_filtered_lr1e4_mb1` ep2; paired CI must not show an increase | F runs (gate: abstain vs call trade-off) |
| Parse or format errors | 0 | all trained runs |
| Final-configuration performance | report mean ± sd; no epoch or seed picking | F |
| Ablation effect | paired bootstrap vs F-s42; a CI including 0 means "no difference" | A-VIS, A-IMPL |
| A-VIS success signal | numeric (v2.1) up, and fewer same-unit "most" errors; also report working/answer inconsistency | A-VIS |
| A-IMPL success signal | implicit-allergy slice improves (baseline 8/18) and P5 probe; explicit slice not worse | A-IMPL |

**Ablations are reported, not adopted into the final configuration this round**, even if they win on val. Adopting one would require new seeds and a new frozen test protocol.

## 7. Artifacts to write

- `outputs/<run>/val/` and `outputs/<run>/test/`: trajectories, v1 scored, v2.1 scored, metrics, error analysis.
- `outputs/<run>/manifest.json`: seed, view, config and prompt hashes.
- `reports/wave3/`: the main table, the diagnostics table, slices, probes, paired comparisons, seed summary.
- `reports/DATA_QUALITY.md`, `reports/REPORT.md`.
