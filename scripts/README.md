# Scripts

Library code lives in `src/clinqa/` (formatting, training, evaluation, tools, scorers). These scripts orchestrate it. Run everything from the repository root with `uv run python scripts/<name>.py`. GPU scripts run on a RunPod pod; everything else is CPU-only.

## Pod setup and experiment rounds (GPU)

| Script | Purpose |
|---|---|
| `setup_runpod.sh`, `runpod_env.sh` | Pod setup (uv, Python, CUDA-13 driver check) and per-terminal environment (caches on `/workspace`) |
| `run_w3_round.sh` | Wave-3 round: baselines, C-filtered, F-s42, P1 probes, Qwen3.5 filtered (`make w3-round`) |
| `run_w4_round.sh` | Wave-4 stages: `r1` (Qwen3.5 relabel), `refit` (F′), `r2` (Stretch A zero-shot and A-sft), `r2b` (A-sft2), `test` (frozen test run) |
| `w2_epochs.py` | Per-epoch helpers used by the runners (`stable`: training stability check) |
| `w3_epochs.py` | `ckpt` resolves an epoch checkpoint against the run manifest; `generate` evaluates both epochs on val, P1, train-fit and extra record files |
| `w3_prep.py` | Mask audits of a format config plus train view (`audit`), and wave-3 review artefacts |
| `w4_test.py` | `freeze` the test list (`configs/w4/final_test.json`), then `run` it once from a clean, committed tree |

## Scoring and analysis (CPU)

| Script | Purpose |
|---|---|
| `score_v2.py` | Scorer v2.1 on saved trajectories (`--split val|test`); `--scorer 2.2` uses the unadopted prototype |
| `score_v21_val.py` | Guarded validation wrapper: exact 250 IDs, refuses test labels and existing output directories |
| `w3_analyze.py` | `p1` probes, `gate`, `trainfit`, `train` (training metrics), `c10` (calibration diagnostics) |
| `compare_rollouts.py` | Item-by-item comparison of two saved validation rollouts (parity checks) |
| `clinical_context.py` | Post-hoc check of the brief clinical context in tool answers |
| `numeric_audit_prep.py`, `w4_numeric_audit.py` | Numeric audit strata, blinded packets, rater merge and the H2 sign test (audit not run, D-101) |
| `scorer_v2_gold_check.py` | Gold-as-prediction self-check of the scorer |
| `scorer_v22.py` | Scorer v2.2 prototype (numeric contradiction guards); not adopted (D-096) |

## Stretch A: `calculate_egfr`

| Script | Purpose |
|---|---|
| `stretch_a_data.py` | Generated the first eGFR rows (`data/stretch_a/`) and the frozen 19 + 19 evaluation sets |
| `stretch_a_data_v2.py` | Generated the revised, stage-balanced rows used by A-sft2 (`data/stretch_a_v2/`, D-100) |
| `stretch_a_review.py` | Review sheet and decisions for the evaluation items |
| `stretch_a_score.py` | `freeze` the evaluation sets; `score` eGFR end-to-end, KDIGO category, probe fabrication and core over-call |

## Legacy

Superseded and one-off scripts are in `scripts/legacy/`; see its README.
