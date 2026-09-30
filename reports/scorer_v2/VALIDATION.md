# Scorer v2 validation (2026-09-29)

Design: `docs/SCORER_V2.md`. Reproduce the numbers in section 3:
`uv run python scripts/scorer_v2_agreement.py`. Reproduce the gold self-check:
`uv run python scripts/scorer_v2_gold_check.py train|val`.

## 1. Artifacts in this directory

| Path | Contents |
|---|---|
| `scorer_v2.1.sha256` | Hash of the current `src/clinqa/scorer_v2.py` |
| `scorer_v2.0_frozen_for_holdout1.sha256` | Hash of v2.0, frozen before holdout 1 was scored |
| `val/`, `test/` | `keys.jsonl` (answer keys, built without predictions), `<label>.scored.jsonl`, `summary.json`, `summary.md` |
| `adjudication/dev_val/` | 120-item blinded packet, rater files (A, B, C1–C4) and the identity key |
| `adjudication/holdout1_test/` | 120-item blinded test packet, rater files (HA, HC1–HC4) and the identity key |
| `adjudication/holdout2_test/` | 110-item blinded test packet (question IDs disjoint from holdout 1), rater files (HA, HC1–HC4) and the identity key |

Wave-1 inputs: `~/Downloads/outputs_9599.zip` (RunPod, RTX 4090). Raw outputs are not copied into the repo.

## 2. Adjudication protocol

- **Blinding.** Adjudicators saw the question, note, table, the gold answer (labelled "may be wrong"), the model's tool calls with their results, and the final answer. They saw no scorer output and no model identity.
- **Rubric.** Written per answer type:
  - The input is the source of truth.
  - Unrequested extra numbers are not required.
  - Reasonable rounding is accepted.
  - Tool calls may deviate by ~0.1 in BMI.
  - Q5 records expect abstention.
  - An empty answer is incorrect.
- **Development set (val).** 120 items stratified by answer type; half v1/v2 disagreements, half agreements.
  - Rater A: one agent covering all items.
  - Rater C: four strict agents, 30 items each, required to cite values.
  - Rater B was discarded: one templated reason on 71 items, 114 items marked "high" confidence.
- **Holdout 1 (test).** 120 items stratified by answer type, about 40% v2.0 failures, from `base_test` and `raw_lr1e4_test`.
  - Scorer v2.0 was frozen by sha256 first.
  - HA and HC1–HC4 used the same rubric. HA's instructions did not forbid reading the HC files, and HA finished after them. Its reasons are independently worded (mean similarity 0.33, none identical).
- **Holdout 2 (test).** 110 items from the test question IDs not used in holdout 1 (only 35 numeric IDs remained). Scorer v2.1 was frozen first; its sha256 was checked before scoring. Each rater wrote to its own directory and was explicitly forbidden to read the others. The reference is the 108 items where HA and HC agree; the other 2 differ only on the policy "BMI computed by hand without calling the tool".
- **Adjudicators are LLM agents, not clinicians.**

## 3. Results

### Rater agreement

| Set | Raters | Agreement | κ |
|---|---|---|---|
| dev_val | A vs C | 119/120 | 0.92 |
| dev_val | A vs B (discarded) | 117/120 | 0.71 |
| holdout1_test | HA vs HC | 120/120 | 1.00 |
| holdout2_test | HA vs HC | 108/110 | 0.93 |

### Scorer agreement with the reference

| Scorer | dev_val | holdout1_test | holdout2_test |
|---|---|---|---|
| v1 (`metrics.py`) | 65.8%, κ 0.17 (41 false fails) | 67.5%, κ 0.34 | 65.7%, κ 0.26 (35 false fails) |
| User bounded-contract v2 | review→fail: 71.7%, κ 0.21; resolved only: 96.4% on 84/120; 36 items unresolved | not run (refuses test) | not run |
| v2.0 (frozen) | 99.2%, κ 0.92 (tuned) | **76.7%, κ 0.48**: clean | — |
| **v2.1 (frozen)** | 100% (tuned) | 95.0%, κ 0.86 (tuned) | **92.6%, κ 0.72**: clean (2 false passes, 6 false fails) |

**Reading the table**

- The only clean generalisation estimate is **v2.0 on holdout 1 (76.7%)**. It shows that the development set was overfit.
- The clean estimate for **v2.1 is holdout 2: 92.6%, κ 0.72**. The fixes derived from holdout 1 generalised: accuracy rose from 76.7% to 92.6% on unseen items. v1 scores 65.7% on the same items.
- Remaining v2.1 errors on holdout 2:
  - 3 false fails from the last-assertion rule picking a non-final statement;
  - 1 tool status-consistency false fail;
  - 2 uncertain "missing field" taxonomy misses (timestamp vs "prior value");
  - 2 false passes: a hedged "cannot be safely initiated" counted as hedged, and a final "Answer:" line contradicting the body on a "most" question.
- These all come from regex parsing of answers. Replacing that layer is the proposed next step: an LLM parser with a deterministic verifier (see `docs/SCORER_V2.md` section 5).

### Gold self-check (gold as the prediction)

| Version | train | val |
|---|---|---|
| first v2 | 90.2% | — |
| v2.1 | 98.6% (1183/1200) | 100% (150/150) |

v2.1 structured coverage on train:

- **Extractive:** 596 structured, 25 partial, 179 text fallback. The 179 are genuine free-text questions.
- **Numeric:** 302 structured, 90 partial, 8 text fallback.

## 4. Wave-1 rescoring with v2.1 (no generation, CPU)

| Run | Split | extr | num | tool | unc | macro (v1 → v2.1) |
|---|---|---|---|---|---|---|
| base | val | 0.39→0.93 | 0.40→0.82 | 0.47→0.60 | 0.66→0.61 | 0.479→0.738 |
| raw_lr5e5 ep1 | val | 0.98→1.00 | 0.52→0.86 | 0.87→0.87 | 0.90→0.90 | 0.816→0.906 |
| raw_lr5e5 ep2 | val | 0.98→0.99 | 0.52→0.86 | 0.89→0.89 | 0.87→0.84 | 0.814→0.895 |
| raw_lr1e4 ep1 (wave-1 selection) | val | 0.99→0.99 | 0.60→0.88 | 0.89→0.90 | 0.92→0.92 | 0.850→0.924 |
| raw_lr1e4 ep2 | val | 0.99→1.00 | 0.54→0.86 | 0.87→0.87 | 0.95→0.95 | 0.837→0.920 |
| q5filtered_lr5e5 ep1 | val | 0.99→1.00 | 0.46→0.82 | 0.89→0.98 | 0.92→0.90 | 0.815→0.925 |
| q5filtered_lr5e5 ep2 | val | 0.98→0.99 | 0.48→0.82 | 0.87→0.95 | 0.95→0.95 | 0.820→**0.927** |
| base_test | test | 0.35→0.98 | 0.25→0.89 | 0.40→0.57 | 0.77→0.73 | 0.442→0.793 |
| raw_lr1e4_test | test | 0.94→1.00 | 0.56→0.90 | 0.83→0.87 | 0.97→0.97 | 0.824→0.934 |

Observations:

- **Most of v1's extractive gap between base and SFT was a scoring artefact.** Under v2.1, base already reaches 0.93–0.98.
- **The durable SFT gains are behavioural:** tool use (base 0.57–0.60 vs 0.87–0.98), uncertainty (base 0.61–0.73 vs 0.84–0.97) and format.
- **On tool calls, the Q5-filtered runs lead** (0.95–0.98 vs 0.87–0.90). They abstain on the Q5 records where the raw runs invent BMI arguments.
- **On macro, q5filtered ep2 is highest (0.927) and the wave-1 selection is close behind (0.924).** The difference is within noise (val n = 250). No new checkpoint selection has been made with v2.1.
