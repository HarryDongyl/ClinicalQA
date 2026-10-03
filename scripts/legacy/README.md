# Legacy scripts

Superseded or one-off scripts, kept so that historical results stay reproducible. They are not part of the current pipeline. Each moved from `scripts/` to `scripts/legacy/` on 2026-10-03, and file history is preserved by git rename tracking.

Historical documents (`docs/history/`, `docs/DECISIONS.md`, `docs/EXPERIMENT_JOURNAL.md`) and some config comments (`configs/wave2_decisions.yaml`, `configs/train/w2_*_mb2.yaml`) still cite the old `scripts/<name>` paths. They are left unchanged because configs are hash-recorded in run manifests.

| Script | Original use | Superseded by |
|---|---|---|
| `run_experiments.sh` | Wave-1 runner on RunPod | `scripts/run_w3_round.sh`, `scripts/run_w4_round.sh` |
| `export_results.py`, `package_runpod.py` | Wave-1 result export and packaging (each imports the other) | Git commits from the pod plus Hugging Face uploads |
| `review_wave1.py` | One-off wave-1 review (checkpoint provenance, HF verification) | `scripts/score_v21_val.py`, `scripts/w3_analyze.py` |
| `propose_q5_uncertain.py` | Draft uncertain labels for the 78 Q5 records | Reviewed relabels in `configs/w3/q5_relabel_review.jsonl` (`scripts/w3_prep.py`) |
| `run_w2_round.sh` | Wave-2 runner (prompt ablation, LR ladder); still `make w2-round` | `scripts/run_w3_round.sh`, `scripts/run_w4_round.sh` |
| `rescore_validation_v2.py` | Bounded-contract v2 scorer (`src/clinqa/scoring_v2.py`); still used by `make score-v2`, `make w2-score`, `make w3-score` | Scorer v2.1 (`scripts/score_v2.py`) |
| `scorer_v2_agreement.py` | Agreement of v2.0/v2.1 with the blinded adjudication sets (results in `reports/scorer_v2/VALIDATION.md`) | Recorded results; see `docs/SCORER_V2_1_KNOWN_ISSUES.md` S-16 for its limits |

Python scripts here resolve the repository root as `Path(__file__).resolve().parents[2]`. Run them from the repository root, for example `uv run python scripts/legacy/rescore_validation_v2.py --help`.
