# v1 review: key decisions and open issues

Review date 2026-09-29. Four independent code reviews (data and formatting, training and environment,
evaluation, docs versus code) were run against [ASSIGNMENT.md](ASSIGNMENT.md). All 162 unit tests pass on CPU.
No GPU run exists yet, so nothing here is a model result.

## 1. Coverage of the assignment

| Requirement | Status | Where |
|---|---|---|
| Data formatting to chat format (note + table in the user message, tool-call pattern, answer-type distribution kept) | done | `formatting.py`, `reports/formatted_examples.md`, `reports/mask_audit.md` |
| Runnable SFT script (LoRA/QLoRA) | implemented, not run on GPU | `train.py`, `configs/train_raw.yaml` |
| Data quality analysis (at least 5 examples, split stats, note length, tool frequency, table types, issues) | done | `reports/data_analysis.md`, `reports/data_stats.json`, `docs/DATA.md` |
| Evaluation: extractive, numeric, tool selection, tool arguments, uncertainty | implemented, CPU-tested | `metrics.py`, `evaluate.py`, `infer.py` |
| Report: setup, commands, hardware, artifacts, what was tried, results, limitations | skeleton; sections 3, 7, 8 and 9 are PENDING | `reports/REPORT.md` |
| Pinned dependencies | `uv.lock` yes; **`requirements.txt` is stale** (see I-2) | |
| Seeds and determinism | seed 42 everywhere, greedy decoding, `use_deterministic_algorithms(warn_only=True)` | `seed.py` |
| Version control | **no commits yet** (see I-3) | |
| Stretch | not attempted (`data/reference.jsonl` is only analysed; no `reference_lookup` tool) | |

## 2. Decision register (verified against the code)

Legend: OK means the code matches the documented decision. DRIFT means the docs and code disagree.
UNDOC means the decision appears only in code.

### Data and quality

| ID | Decision | Implemented at | Check | Assessment |
|---|---|---|---|---|
| D-010 | `data/` files are byte-identical copies of `_data/`; SHA-256 manifest, re-verified before use | `data_io.py:107-151`, `data/MANIFEST.json` | OK | Strong provenance |
| D-012 / P-012 | Raw view keeps all 2,000 train rows (R1). Train-only Q5-filtered view keeps 1,922 (drops 78 tool calls whose gold arguments cannot be recovered from the input) (R2) | `data_views.py:21-34`, `analysis/checks.py:339-386` | OK | val and test are never filtered, and no labels are read while building views |
| D-009 / D-016 | Quality heuristics (Q1-Q19) produce review candidates, not relabels. Q10 flags were parser false positives and those rows are kept | `analysis/checks.py` | OK | The manual review of the 78 Q5 packets is still outstanding |
| D-017 | Test labels were inspected for the audit only. Never used for training or selection | `evaluate.py:283` (`generate` accepts only train/val), `evaluate.py:340-374` | OK | Enforced in code, not just by policy |
| P-006 | No upsampling; splits used as-is | `formatting.py` | OK | Split distributions are 40/20/25/15 (val 40/20/24.8/15.2) |

### Prompt and chat formatting

| ID | Decision | Implemented at | Check | Assessment |
|---|---|---|---|---|
| P-002 | Table serialised as Markdown; all rows kept; empty unit shown as `(no unit)`; pipes escaped | `formatting.py:53-71` | OK | User message sections: `## Encounter note` / `## Table (type)` / `## Question` |
| P-003 / D-025 | Frozen system prompt `configs/prompts/system_v1.txt`, hashed into every manifest; grounding covers note, table and values stated in the question | `formatting.py`, `run_info.py` | OK | |
| D-021 | Custom prefix-diff loss masks instead of TRL `assistant_only_loss` (the stock Qwen3 template has no generation markers). Assistant content plus `<\|im_end\|>` is supervised; the header is masked | `formatting.py:137-173` | OK | Audit: 2,000/2,000 clean, 5.5% of tokens supervised. val masks are not audited (low risk, same code) |
| — | Base model is the non-thinking Instruct-2507 variant, so no `<think>` block handling is needed | configs | UNDOC (implicit) | Correct choice for masking simplicity |
| — | Sidecar files hold answer_type, gold and flags; the conversation never contains labels. Test is formatted prompt-only | `formatting.py:111-114,245-246` | OK (tested) | Good leakage guard |

### Tool-call protocol

| ID | Decision | Implemented at | Check | Assessment |
|---|---|---|---|---|
| D-024 | Native Qwen `<tool_call>{"name","arguments"}</tool_call>` through `apply_chat_template(tools=...)`. The call turn has no text. The tool result goes in a `tool` role message (rendered as `<tool_response>`), followed by the final answer | `formatting.py:85-100`, `schemas.py` | OK | Matches the model's pretraining format, which is the best prior |
| D-024 / P-005 / D-003 | Targets copy the gold metric arguments directly (single call). The model must learn imperial to metric conversion implicitly. An explicit `unit_convert`-then-BMI chain is only the optional R3 ablation | `formatting.py:96-100` | OK | Follows the assignment's intent. Risk: the arithmetic happens internally with no visible trace (R3 is the fix) |
| D-024 | `substance` is always emitted (`null` for body measurements) | `formatting.py:99` | OK | Fixed argument shape |
| D-008 / P-001 | Tool formulas exactly as specified; `unit_convert` rounds to 2 dp, BMI to 1 dp; unsupported conversions return an error string | `tools.py:63-125` | OK | Python `round()` (banker's rounding) differs from the `ROUND_HALF_UP` diagnostic in `checks.py:266`. No live mismatch (Q4 = 0) |
| D-013 | Harmless unit aliases normalised (lbs, kgs, °f, degf...); no semantic argument repair | `tools.py:18-48` | OK; alias list UNDOC | |
| D-026 / D-033 | Malformed JSON is not repaired. A schema-invalid call stops the rollout (`schema_error`); executor errors return `{"error":...}` | `schemas.py:100-179`, `infer.py:78-79` | OK | Strict. It measures raw reliability but penalises cosmetic slips |
| — | A call object must have exactly the keys `{name, arguments}`; the `unterminated` status is used for an unclosed `<tool_call>` | `schemas.py:22-27`, `infer.py` | UNDOC | Add to D-024/D-033 |

### Base model and quantisation

| ID | Decision | Implemented at | Check | Assessment |
|---|---|---|---|---|
| D-002 / D-020 | Qwen/Qwen3-4B-Instruct-2507, revision `cdbee75f…`, single model | all configs | OK | Good fit: native tool-calling template, fits 24 GB with QLoRA, strong at 4B |
| D-022 / D-023 | QLoRA NF4 on CUDA (bf16 compute) for both training and inference, for R0 and all adapters; unquantised on CPU for smoke only | `modeling.py:36-56` | OK | Same quantisation for base and adapters keeps R0 versus R1 fair |
| D-018 | Qwen3-0.6B is a smoke model only, using the 4B tokenizer/template (shared vocabulary) | `configs/smoke.yaml` | OK | |
| D-001 | "Transformers + TRL + PEFT" | — | **DRIFT**: TRL is not a dependency; D-021/D-022 use a plain `Trainer` | Mark D-001 as superseded |

### LoRA and hyperparameters (P-008, D-022)

| Setting | Value | Assessment |
|---|---|---|
| LoRA | r=16, alpha=32, dropout 0.05, all 7 linear projections | Standard for about 2k examples on a 4B model |
| LR / schedule | 1e-4, cosine, warmup 3% (8 steps) | Conservative for QLoRA; reasonable given only 250 steps |
| Epochs / batch | 2 epochs; micro-batch 1 x grad_accum 16 = effective 16; 125 steps per epoch, 250 total | micro-batch 1 is safe but slow. Lengths cluster between 1.3k and 1.6k tokens, so micro-batch 2-4 with accum 8-4 would give the same effective batch with little padding |
| max_length | 2048 (observed p95 1,477, max 1,614 tokens) | No truncation |
| Checkpoints | per epoch (`checkpoint-125`, `checkpoint-250`) plus `final/`; `save_only_model` | No resume after a crash (acceptable for a run of about 3 h) |
| Diagnostics | per-answer-type teacher-forced val loss at each epoch; `train_log.jsonl` plus TensorBoard; W&B off | Good. It gives early warning if one answer type collapses |
| No sweep | one R1 run, epoch chosen on val | Matches the 4-hour scope; list HP sweeps as next steps |

### Evaluation and selection

| ID | Decision | Implemented at | Check | Assessment |
|---|---|---|---|---|
| D-023 | Greedy HF `generate`, left padding, batch 16; budget of 1 call, 2 assistant turns, 256 tokens per turn, 512 total | `infer.py:41-45,155,166-169`, `eval_core.yaml` | OK | A model that writes text before calling, or calls on turn 2, cannot finish and fails tool E2E |
| D-028 / P-009 | Deterministic scoring, no LLM judge. Extractive: grounded gold numbers at source precision plus direction words, falling back to token-F1 >= 0.5. Numeric: derived numbers exact plus at least half the grounded numbers plus a swapped-analyte guard. Uncertainty: abstention phrase plus the missing field named plus no fabricated value | `metrics.py:198-299` | OK | Regex-bound to English phrasing; will under-credit paraphrases |
| D-027 | Tool args: metric exact at source precision; imperial-derived within ±0.05; `unit_convert` value exact | `metrics.py:304-343` | OK | |
| P-010 / D-034 | Extra metrics: tool E2E, over-call, over-refusal, unsupported-argument rate, result incorporated in answer; Wilson 95% CIs | `metrics.py`, `evaluate.py:57-64` | OK | Goes beyond the required metrics |
| PLAN §6 / D-034 | Selection = highest macro mean of 4 task success rates (tool component = E2E) on the Q5-grounded val subset (243/250); ties within 0.005 go to fewer unsupported calls, then fewer over-calls, then the earlier checkpoint | `evaluate.py:260-271` | OK | Predeclared. `select` cannot target test |
| D-030 / D-036 | Test runs once from `configs/final_eval.yaml`; refuses a dirty or uncommitted tree or an adapter that is not `checkpoints[selected]`; a rerun needs `--rerun-reason` | `evaluate.py:340-374` | OK | Strongest safeguard in the repo |
| — | R0 = the same prompt, tools and budget with no adapter (not a few-shot or prompt-tuned baseline) | `eval_core.yaml:20` | UNDOC caveat | State in REPORT that R0 is a zero-shot baseline |
| — | Paired bootstrap (n=2000, seed 42) per answer type, with fixed/broken ids | `evaluate.py:220-257` | OK | |

### Reproducibility and infrastructure

| ID | Decision | Check | Assessment |
|---|---|---|---|
| D-007 / D-032 | Python 3.11, uv lock, GPU stack pinned in the `train` extra, re-validated by `make smoke` on the GPU | OK, but see I-1 | |
| D-029 | Every run records git commit and dirty flag, package versions, hardware, data/prompt/template hashes and runtime | OK | With zero commits, every manifest records `commit: null, dirty: true` |
| D-018 | RunPod plus network volume | **superseded for this run** by [SAGEMAKER.md](SAGEMAKER.md) | |
| D-031 | Baseline commit, then one commit per module, pushed to a private GitHub repo | **not done** | |

## 3. Open issues found in the review

| ID | Severity | Issue | Fix |
|---|---|---|---|
| I-1 | **Blocker (possible)** | `torch==2.14.0` in `uv.lock` bundles CUDA 13 libraries (`uv.lock:876-891`) and needs driver >= 580. The README says "CUDA 12 template". | Check with `nvidia-smi` on SageMaker. If the driver is older, use the cu126 override in SAGEMAKER.md step 3 and record it |
| I-2 | High | `requirements.txt` was exported **without** `--extra train` (header line 2), so it has no torch/transformers/peft/bitsandbytes. `pip install -r requirements.txt` cannot train | Run `make lock` (it exports with `--extra train`) and commit |
| I-3 | High | The repo has no commits. `final-eval` refuses to run, the manifests have no commit hash, and the "version-controlled" criterion is unmet | Commit before pushing to SageMaker (SAGEMAKER.md step 1) |
| I-4 | Medium | Stale docs: PLAN.md:33-38, FINDINGS.md:24-27 and DATA.md:159 still describe formatting, training and eval as "not yet implemented". README says "D-018 to D-032" but the decisions go up to D-036. D-001 mentions TRL | Doc refresh |
| I-5 | Low | `evaluate.py:268-270`: `select()` raises `TypeError` if a candidate's macro is `None` (an empty answer-type bucket). Unreachable on the full val set, but unguarded | Skip or raise a clear error |
| I-6 | Low | `infer.py:112`: the per-batch `max_new_tokens` is the minimum remaining budget across the batch, so co-batched examples can cap each other. It is masked by 256 << 512, but the comment at :107 claims there are no batch effects | Cap per sequence, or correct the comment |
| I-7 | Low | `evaluate.py:331`: the epoch label uses `round(epoch)`, so two checkpoints could collide on the same label. Impossible with per-epoch saves, unguarded otherwise | Assert the labels are unique |
| I-8 | Low | `Makefile:82` uses bare `python3` instead of `$(PY)` | Use `$(PY) python` |
| I-9 | Low | `audit_masks.py:151` hardcodes the length of the assistant header string (display only) | — |
| I-10 | Info | Q5 manual review (78 train packets; D-019 asks for a 15-packet spot check if R2 is trained) is not done | Do it while R1 trains |

## 4. What to fill in after the GPU run

REPORT section 3 (GPU type, driver, CUDA path taken for I-1, peak VRAM, train and eval wall time from
`outputs/*/run.json`), section 7 (the five required metrics plus E2E, over-call and over-refusal, with Wilson
CIs, R0 versus the selected R1 on val and on test), section 8 (failure modes from `outputs/<label>/val/error_analysis.md`,
especially tool-call schema errors and uncertainty misses), and the R2 comparison if it was run.
