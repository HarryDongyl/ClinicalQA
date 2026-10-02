# Wave-four runbook: local (CPU) and RunPod A100 commands

Design: [EXPERIMENTS_WAVE4](INTERVIEW_PREP.md) decisions Wave4-Q1 to Q5 and [STRETCH_A_PLAN.md](STRETCH_A_PLAN.md).

The commands below call scripts directly. They do not depend on the items that are still missing:

- a Makefile `w4-score` target;
- unit tests for the new wave-four code;
- `docs/EXPERIMENTS_WAVE4.md`;
- the `STRETCH_A_PLAN.md` update to Qwen3.5 as the primary backbone (F as fallback).

`scripts/run_w4_round.sh` refuses to start unless all tracked changes are committed.

**Status of the wave-four tooling.** `run_w4_round.sh` and `w4_test.py` were syntax-checked and partly unit-checked, but never executed end to end. Watch the output of every step on the first pod run.

---

## Step 0. Move the code to the machine that runs experiments (once)

On the development laptop, push a branch. If pushing is not allowed there, copy the folder instead and commit on the other machine.

```bash
cd ~/Desktop/Personal/ClinicalQA
git checkout -b w4
git add src/clinqa scripts tests configs data/stretch_a data/processed reports/stretch_a reports/w3 \
        docs/STRETCH_A_PLAN.md docs/INTERVIEW_PREP.md docs/EXPERIMENTS_WAVE3.md docs/WAVE3_RESULTS_REVIEW_8349.md \
        docs/RUNBOOK_W4.md docs/history/local_2026-09-29
git status --short          # check nothing is missing and no large file was added
git commit -m "w4: Qwen3.5 relabel, P1-RAW, Stretch A (calculate_egfr) code, configs and frozen eval sets"
git push -u origin w4
```

## Step 1. Local preparation on the other laptop (CPU)

```bash
git clone https://github.com/HarryDongyl/ClinicalQA.git && cd ClinicalQA   # or git fetch in an existing clone
git checkout w4
make setup                       # torch/transformers + qwen35 extra (no bitsandbytes on macOS)
uv run pytest -q                 # all must pass
```

Optional: catch template problems before renting a GPU. This needs Hugging Face access; if it is unavailable, skip it, because the pod repeats both audits.

```bash
uv run python scripts/w3_prep.py audit --format configs/format_w4_q35_4b.yaml --view q5_relabeled \
  --out reports/w4/mask_audit_q35_4b_relabel.json
uv run python scripts/w3_prep.py audit --format configs/format_w4_q35_4b_egfr.yaml --view q5_relabeled_egfr \
  --out reports/w4/mask_audit_q35_4b_egfr.json
```

Both must print `"passed": true`. Commit and push any new report files.

---

## Step 2. RunPod A100 80GB, round 1 (about 2 h)

**Environment** (same as the wave-three Qwen3.5 round):

```bash
cd /workspace
git clone https://github.com/HarryDongyl/ClinicalQA.git && cd ClinicalQA
git checkout w4
bash scripts/setup_runpod.sh          # uv, HF_HOME, etc.; or follow the README RunPod steps
make setup                            # includes flash-linear-attention (qwen35 extra)
uv run --frozen hf auth login         # private adapter repositories
uv run --frozen pytest -q
```

**Adapters needed.** The script downloads them on demand. Check that they exist first:

```bash
uv run --frozen hf download Harrydongyl/clinqa-w3_q35_4b_filtered_lr1e4 --include "checkpoint-121/*" --local-dir checkpoints/w3_q35_4b_filtered_lr1e4
uv run --frozen hf download Harrydongyl/clinqa-w3_q35_4b_filtered_lr1e4 --include "checkpoint-242/*" --local-dir checkpoints/w3_q35_4b_filtered_lr1e4
uv run --frozen hf download Harrydongyl/clinqa-raw_lr1e4 --include "checkpoint-250/*" --local-dir checkpoints/raw_lr1e4
```

If a repository name differs or is missing, place the checkpoint at the same path by hand. `scripts/w3_epochs.py ckpt` checks each adapter against `outputs/<run>/manifest.json` and refuses on a mismatch.

**Run round 1:**

```bash
tmux new -s w4
STAGES=r1 UPLOAD=1 bash scripts/run_w4_round.sh 2>&1 | tee outputs/w4_r1.console.log
```

In order, it runs:

1. the mask audit;
2. A-Q35-relabel training, alone on the GPU;
3. both epochs on val and P1, plus train-fit on epoch 2;
4. P1 on A-Q35-filter epoch 1;
5. the parity regeneration of A-Q35-filter epoch 2 on val;
6. P1-RAW.

Results are committed and pushed after each step. When it finishes, the pod can be stopped.

---

## Step 3. Local scoring and the gate (CPU)

```bash
git pull
OUT=reports/w4/r1
REF="w3_c_filtered_s42 w3_relabel_lr1e4_s42_step000250 w3_q35_4b_filtered_lr1e4_step000242 w3_r0_q35_4b"
VAL="w4_q35_4b_relabel_lr1e4_step000121 w4_q35_4b_relabel_lr1e4_step000250 w4_q35_4b_filtered_lr1e4_step000242"

# v2.1, bounded v2, P1, train-fit, training metrics, C10 and clinical context
uv run python scripts/score_v21_val.py         --out $OUT/v21        --labels $VAL $REF
uv run python scripts/rescore_validation_v2.py --out $OUT/bounded_v2 --labels $VAL $REF
uv run python scripts/w3_analyze.py p1 --out $OUT/p1 --labels \
  w4_q35_4b_relabel_lr1e4_step000121 w4_q35_4b_relabel_lr1e4_step000250 \
  w4_q35_4b_filtered_lr1e4_step000121 w4_raw_lr1e4_step000250 $REF
uv run python scripts/w3_analyze.py trainfit --out $OUT/trainfit --labels \
  w4_q35_4b_relabel_lr1e4_step000250 w3_q35_4b_filtered_lr1e4_step000242 w3_relabel_lr1e4_s42_step000250
uv run python scripts/w3_analyze.py train --out $OUT/training --runs w4_q35_4b_relabel_lr1e4 w3_q35_4b_filtered_lr1e4 w3_relabel_lr1e4_s42
uv run python scripts/w3_analyze.py c10   --out $OUT/c10_v21 --v21 $OUT/v21 --labels $VAL $REF
uv run python scripts/clinical_context.py score --out $OUT/clinical_context --labels $VAL $REF

# Parity: A-Q35-filter epoch two under the wave-four code must match the wave-three output item by item
uv run python scripts/compare_rollouts.py w3_q35_4b_filtered_lr1e4_step000242 w4_q35_4b_filtered_lr1e4_step000242 \
  --out $OUT/parity_q35_filter_ep2.json

# Gate for A-Q35-relabel (control: A-Q35-filter epoch two; thresholds identical to F-s42's)
uv run python scripts/w3_analyze.py gate --candidate w4_q35_4b_relabel_lr1e4_step000250 \
  --gates configs/w4/gates_q35.yaml --out $OUT/gates
```

Then:

1. **Review by hand.** Read the gate report in `$OUT/gates/`, all 34 P1 answers and the 7 natural-Q5 answers of A-Q35-relabel.
2. **Parity must show no differences.** If it does, stop and find the cause before continuing.
3. **Record the reviewed decision.** Round 2 reads this file; `decision` is `pass` or `fail`:

```bash
cat > configs/w4/gate_q35.json <<'EOF'
{"candidate": "w4_q35_4b_relabel_lr1e4_step000250", "approved": true, "decision": "pass",
 "reviewer": "<name>", "date": "2026-10-xx", "report": "reports/w4/r1/gates/gate_w4_q35_4b_relabel_lr1e4_step000250.json"}
EOF
git add reports/w4 configs/w4/gate_q35.json && git commit -m "w4: r1 scoring and Q35-relabel gate" && git push
```

**Stopping rule (Wave4-Q2).** A failed gate is reported. It is never re-thresholded and never switched to epoch 1. Stretch A then falls back to F (Qwen3).

**Numeric audit and H2.** This can run in parallel with round 2:

```bash
uv run python scripts/w4_numeric_audit.py packet --labels w3_relabel_lr1e4_s42_step000250 w4_q35_4b_relabel_lr1e4_step000250
# Two independent LLM raters read reports/w4/numeric_audit/packet.jsonl and rubric.md (never key.json)
# and write rater_A.jsonl and rater_B.jsonl
uv run python scripts/w4_numeric_audit.py merge --raters reports/w4/numeric_audit/rater_A.jsonl reports/w4/numeric_audit/rater_B.jsonl
# On disagreements: the user writes adjudication.jsonl ({"item","label"} per line) and reruns merge with --adjudication
uv run python scripts/w4_numeric_audit.py h2 --a w3_relabel_lr1e4_s42_step000250 --b w4_q35_4b_relabel_lr1e4_step000250
```

**H2 rule (Wave4-Q4).** "Qwen3.5 stronger on numeric" requires:

- Qwen3.5-only correct items outnumber F-only correct items, with an exact two-sided sign test p < 0.05;
- all non-inferiority checks hold: P1 fabrication ≤ 5%; grounded tool tasks ≥ 52/55; intact-partner valid calls ≥ 31/34; the paired CIs of the other assignment metrics are not all below 0.

---

## Step 4. RunPod A100, round 2: Stretch A (about 3 h)

```bash
cd /workspace/ClinicalQA && git pull           # gets gate_q35.json
tmux new -s w4r2
STAGES=r2 UPLOAD=1 bash scripts/run_w4_round.sh 2>&1 | tee outputs/w4_r2.console.log
```

The gate decides the backbone: `pass` uses Qwen3.5; `fail` falls back to Qwen3 (F) automatically. It runs:

1. the three zero-shot arms (A-zs-v1, A-zs-v1e, A-zs-base), each on core val and both eGFR sets;
2. A-sft training, then evaluation on val, P1 and both eGFR sets.

## Step 5. Local Stretch A scoring (CPU)

```bash
git pull
F=q35_4b                              # q3 if the gate failed
SFT=w4_q35_4b_relabel_egfr_lr1e4      # w4_q3_relabel_egfr_lr1e4 if the gate failed
uv run python scripts/stretch_a_score.py score --out reports/w4/stretch_a --labels \
  w4_sa_zs_v1_$F w4_sa_zs_v1e_$F w4_sa_zs_base_$F ${SFT}_step000257
uv run python scripts/score_v21_val.py --out reports/w4/r2/v21 --labels w4_sa_zs_v1_$F w4_sa_zs_v1e_$F w4_sa_zs_base_$F \
  ${SFT}_step000128 ${SFT}_step000257 w4_q35_4b_relabel_lr1e4_step000250
uv run python scripts/w3_analyze.py p1 --out reports/w4/r2/p1 --labels ${SFT}_step000257
git add reports/w4 && git commit -m "w4: Stretch A scoring" && git push
```

Take the A-sft step numbers from `outputs/$SFT/manifest.json`. With 2,052 rows and effective batch 16, epoch 1 is about step 128 and epoch 2 about step 257; adjust the labels above if the manifest differs.

**Pre-registered criteria** (STRETCH_A_PLAN §8):

- A-sft: end-to-end ≥ 15/19; E-AGE fabrication ≤ 2/19;
- no harm to core: grounded tool tasks ≥ 53/55; P1 fabrication ≤ 1/34; eGFR over-call ≤ 1/21;
- zero-shot arms: numbers only, no threshold.

---

## Step 6. Freeze the test list (local), then run test on RunPod (about 1 h)

Freeze and commit the list **before** any test output exists:

```bash
Q=w4_q35_4b_relabel_lr1e4; F3=w3_relabel_lr1e4_s42; QF=w3_q35_4b_filtered_lr1e4
uv run python scripts/w4_test.py freeze \
  --entry q35_relabel_test configs/eval_w4_q35_4b.yaml $Q  checkpoints/$Q/checkpoint-250 \
  --entry f_relabel_test   configs/eval_w4_v1.yaml     $F3 checkpoints/$F3/checkpoint-250 \
  --entry q35_filter_test  configs/eval_w4_q35_4b.yaml $QF checkpoints/$QF/checkpoint-242 \
  --entry r0_q35_4b_test   configs/eval_w4_q35_4b.yaml base none
git add configs/w4/final_test.json && git commit -m "w4: freeze test list" && git push
```

`freeze` hashes the adapters, so the checkpoints must be present locally (`hf download`). Alternatively, run `freeze` on the pod and commit there. If the gate failed, replace the first entry with F as the final model.

On the pod:

```bash
cd /workspace/ClinicalQA && git pull
STAGES=test bash scripts/run_w4_round.sh 2>&1 | tee outputs/w4_test.console.log
```

Back on the local machine:

```bash
git pull
uv run python scripts/score_v2.py --runs-dir outputs --split test --out reports/w4/test \
  --labels q35_relabel_test f_relabel_test q35_filter_test r0_q35_4b_test
```

**Report only the pre-specified metrics (option B):**

- natural Q5 fabrication on test (n = 10, underpowered);
- grounded tool tasks;
- the five assignment metrics under v1 and v2.1.

Disclose that test was used once in wave 1 and for scorer validation. Test runs once: a rerun needs `scripts/w4_test.py run --rerun-reason "..."`, and the rerun is written to `RERUN.txt` and disclosed.

---

## GPU budget

| Round | Content | Estimate |
|---|---|---|
| r1 | A-Q35-relabel train + eval, P1 on A-Q35-filter epoch 1, parity regeneration, P1-RAW | about 2 h |
| r2 | Stretch A: 3 zero-shot arms + A-sft train + eval | about 3 h |
| test | 4 models on test | about 1 h |
| **Total** | all on A100 80GB | **about 6 h** |
