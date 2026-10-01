# Clinical QA experiment journal

Maintained record. Last updated: 2026-10-01. Latest entry: W3-CLASSIFIER-007.

## W3-CLASSIFIER-007 — P1 classifier repaired before outputs

Date: 2026-10-01. The P1 helper is now versioned `p1-2` and pinned in `configs/w3/gates.yaml` (D-087). It covers the three defects found during the approval review: wrong-tool fabrication, claims in earlier turns, and examples counted as fabrication.

Applied to the reviewer's synthetic fixtures:

| Fixture | Old result | New result |
|---|---|---|
| Invented weight sent to `unit_convert` | not counted | fabrication |
| False weight claim in an earlier visible turn | ignored | fabrication |
| Explicit hypothetical example | fabrication | mandatory review |

`make test`: 258 passed. No GPU run, no model output, and no change to the frozen probes, approvals or gate thresholds. Seeds 43/44 still wait for a reviewed F-s42 gate (D-086).

## W3-APPROVAL-006 — four artifacts reviewed and approved

Date: 2026-10-01. Status: no GPU work. See WAVE3_APPROVAL_REVIEW.md; decisions D-084 to D-086. AI grounding review accepted all 78 Q5 candidates, revised 14 answers, and built the 2,000-row relabel view (1,922 unchanged rows plus 78 uncertain replacements). Clarified unit conversion in prompt v3; kept the four demos and all 34 P1 records unchanged. Readiness reports all five items approved, including existing train-fit IDs. Metadata no longer labels AI review as human clinical adjudication. Original drafts are archived with hashes.

Validation after changes: 254 tests passed, four non-failing warnings, 52.48s; 2,000-row relabel mask audit had zero problems; maximum training sequence 1,614 tokens, no overlength. Few-shot validation prompts now 4,382–4,639 tokens. Training-view audits passed for all three relabel seeds and 8B filtered configuration. No test-model performance was used.

Open issue: synthetic helper checks show the P1 classifier can miss invented unit_convert arguments and overcount hypothetical measurements. These artifact approvals do not authorize F-s42 gate approval or seed continuation. Review complete visible trajectories against inputs, not only regex flags. No GPU, commit, push or upload was performed.

## W3-PLAN-005 — wave-three round implemented; awaiting approvals and a pod

Date: 2026-10-01. Status: code, configs and CPU checks done; **no GPU run, no new result**. Scope requested by the user: all core GPU experiments, D-TRAINFIT and C10, plus one SFT run on 8B using the current best locked setting. Decisions D-075 to D-083.

### Roster (validation, P1 probes and train-fit only; generation batch 2 for every arm)

| Stage | Arm | Label | Weights | Prompt | Outputs |
|---|---|---|---|---|---|
| core | R0-v1 | `w3_r0_v1` | base 4B | v1 | val, P1 |
| core | C-filtered-s42 | `w3_c_filtered_s42` | `w2_filtered_lr1e4_mb1` step 242 | v1 | val, P1, train-fit |
| core | R0-v3 | `w3_r0_v3` | base 4B | v3 (needs approval) | val, P1 |
| core | R0-v3-FS4 | `w3_r0_v3_fs4` | base 4B | v3 + 4 demos (need approval) | val, P1 |
| core | F-s42 | `w3_relabel_lr1e4_s42_step<N>` | trained on q5_relabeled (needs review) | v1 | both epochs val; epoch two P1, train-fit |
| 8b | R0-8B | `w3_r0_8b` | Qwen3-8B base, non-thinking | v1 | val, P1 |
| 8b | A-8B | `w3_8b_filtered_lr1e4_step<N>` | trained on q5_filtered, locked 1e-4 recipe | v1 | both epochs val; epoch two P1 |
| seeds | F-s43, F-s44 | `w3_relabel_lr1e4_s4{3,4}_step<N>` | relabel view | v1 | only after the F-s42 gate is approved |

Comparisons after scoring (`make w3-score`):
- F-s42 vs C-filtered-s42: a relabel *policy* comparison;
- A-8B vs C-filtered-s42 (backbone + template change) and vs R0-8B (SFT effect on 8B);
- R0-v3/FS4 vs R0-v1: stronger prompting;
- every SFT arm vs the R0 arms.

No semantic winner is declared while the evaluator gaps remain (D-063/D-064 qualifications).

### Verified locally (CPU)

- `make test`: 254 passed, including 14 new wave-three tests.
- The 4B mask audit (2,000 and 1,922 rows, 0 problems) and `reports/token_lengths.json` are byte-identical after the code change.
- The q5_filtered view bytes are unchanged after the rebuild. The view manifests changed only because `data_views.py` is hashed into them.
- 8B mask audit: 0 problems; train max 1,618 tokens, val max 1,566.
- R0-v3 val prompts reach at most 1,707 tokens, and FS4 first-turn prompts are 4,361–4,618 tokens.
- P1: 34 of 40 sources are eligible; 6 are excluded because the question states the measurements.
- C10 code was dry-run on two existing wave-two labels as a software check only; those numbers are not wave-three evidence.

### Approvals needed before the round (the round skips unapproved arms and exits 2)

1. **Q5 relabels** — edit `configs/w3/q5_relabel_review.jsonl`: `accepted`, `reviewer`, `rationale`, and `answer` where needed. The packet is `reports/w3/q5_review_packet.md`. Then run `make w3-views`.
2. **Prompt v3** — read `configs/prompts/system_v3.txt`, then run `uv run python scripts/w3_prep.py approve prompt_v3 --reviewer <name>`.
3. **Demonstrations** — read `configs/w3/fewshot_v3.json` (train_1523, train_1771, train_851, train_802), then run `approve fewshot`.
4. **P1** — read `reports/w3/p1_review_packet.md` (29 mid-line edits), then run `approve p1`. P1 and train-fit are required for the round to start at all.

### Cost (projections, not measurements)

RTX 4090 Secure at $0.74/hr; the existing stopped pod `h2l9dp4u26f1jg` has an 80 GB volume. Projections are scaled from wave-one/two runtimes: batch-2 val generation 400–580 s, 4B training about 2,560 s.

| Stage | Projected time | Notes |
|---|---|---|
| core | ≈2.3 h | F-s42 training ≈45 min |
| 8b | ≈2.5–3 h | 8B download ≈16 GB; A-8B training projected at 85–110 min, since tool rows add a second sequence |
| core + 8b | ≈5 h, ≈$3.7 | |
| seeds | ≈2.2 h, ≈$1.6 | |

Measured runtimes replace these projections in the run entry.

### Commands

```bash
# local: approvals above, commit and push
make w3-round                    # pod: STAGES="core 8b" by default; UPLOAD=1 pushes new adapters to private HF repos
make w3-score                    # local CPU after pulling: v2.1, bounded v2, P1, train-fit, C10 (v1 and v2.1), gate, paired compares
# after reviewing reports/w3/round1/gates/: write configs/w3/gate_fs42.json, then on the pod
make w3-round STAGES=seeds
```

## W3-LOCAL-004 — local verification and GPU inventory

Date: 2026-10-01. Full CPU test suite: 240 passed, four non-failing warnings, 43.69s. Offline assistant-mask audits: raw 2,000 and filtered 1,922 conversations, zero problems, maximum rendered length 1,614. Five training-view audits passed hashes/membership/counts; effective batch16. Four shell scripts passed bash syntax checks. Fourteen wave-two validation score files each contain 250 complete unique IDs (3,500 total). No semantic rescoring, new training, generation or test-performance analysis was performed. Unit tests include canonical test-file integrity checks.

See LOCAL_TESTS_AND_RUNPOD_EXPERIMENTS.md for the conditional GPU queue and implementation gaps. CUDA is unavailable locally. Existing scorer semantic defects and missing relabel/few-shot/P1 implementations remain launch prerequisites; passing the current suite does not resolve them. Start with F-s42 and matched prompt/control inference after prerequisites, then seed replication if gates pass. BF16, 8B, rank and targeted data ablations are optional, not a single unconditional sweep.

## W3-INTERVIEW-003 — prioritize the next experiments

Date: 2026-10-01. Reviewed INTERVIEW_PREP.md, archived incoming text and the prior plan, and installed INTERVIEW_REVIEW.md. Verified that Git commits exist, requirements.txt lacks the training extra, and validation contains 40 grounded BMI candidates (47 BMI minus 7 Q5), not 55. Recomputed call-prefix AUC under annotated versus grounded policy labels; the interpretation changes with the target. No test material, new model generation or training was used.

Revised queue: endpoint-specific evaluator/annotation gate; fixed strong zero-/few-shot baselines; audited relabel F-s42 with original/edited P1 pairs; seeds43/44 only after the control passes; train-fit and correctly defined confidence diagnostics; targeted A-IMPL or numeric-only A-VIS; optional precision/model/rank extensions. P1 is synthetic validation stress evidence, not a new independent clinical test. Raw answer token logprob is a ranking score, not correctness probability. BF16 requires train/serve-factor separation; rank and size results cannot by themselves prove capacity mechanisms. The previous wave-three plan is preserved in history/interview_review_2026-10-01/.

## W3-FINDINGS-002 — supplemental narrative reconciled

Date: 2026-09-30. Installed reviewed ROUND4_FINDINGS.md and archived the incoming original/hash. Remapped D-037..D-052 to D-053..D-068 and repaired references. Verified train-only BMI coverage: 109 uncertain rows, 65 weight-only, 43 height-only and one neither; 78 proposal rows missing both under the existing parser. Corrected val_085 reference provenance, token-versus-gradient interpretation, AUROC claims and unsupported contamination-bias arithmetic. Deferred the imperial explanation component because its stated motivation is test-only; the planned A-VIS is numeric-only. Historical missing-file notes remain as provenance and are superseded by this entry. No annotation was approved and no experiment launched.

## W3-REVIEW-001 — wave-two evidence and asynchronous document merge

Date: 2026-09-30. Reviewed 14 validation score files (3,500 rows), prompt comparisons and three completed training manifests. No new training, generation or test inspection. See `WAVE3_REVIEW.md` and the reviewed `EXPERIMENTS_WAVE3.md`.

Preserved local decision IDs D-001..D-052; incoming reused D-037..D-052 become D-053..D-068. Exact source versions and hashes are archived in `history/wave3_import_2026-09-30/`. FINDINGS.md is identical to the old data findings; ROUND4_FINDINGS.md was not supplied/found.

Corrected prompt flip counts, unsupported LR-equivalence claims, CI interpretation, relabel exposure confounding, Answer-only factuality omission, reused-test wording and spec denominators. Frozen v2.1 remains diagnostic, not a validated selector. Prospective 1e-4/v1/epoch-two choices are planning defaults, not established optima. Claims about allergy/fabrication/token-share subgroups remain unverified pending their source evidence. No wave-three implementation was activated.

## W2-RUN-007 — round in progress: E1 generated; ladder switched from mb4 to mb1

Date: 2026-09-30. Pod: RTX 4090 Secure, $0.74/hr, billing from 02:34 UTC. Status: E1 generation finished and pushed (pod commits 0d665c7 preflight, 4193cce E1 v1, 0a82adb E1 v2); E1 scores are not read yet (D-047). The ladder restarted at mb1 (D-052). Only runtimes are recorded here.

**E1 generation runtimes at batch 4 (measured, `run.json` `runtime_s.generate`).**

| Weights | v1 (s) | v2 (s) | Wave one, v1, batch 2 (s) |
|---|---|---|---|
| base | 578.8 | 607.4 | 398.3 |
| raw_lr1e4 step125 | 782.5 | 790.3 | 581.5 |
| raw_lr5e5 step125 | 764.7 | 770.8 | — |
| q5filtered_lr5e5 step121 | 752.9 | 755.6 | 565.4 |

Batch 4 generation was about 35–45% slower than batch 2, not faster. Base produced about the same work in both: 21,762 vs 21,913 new tokens, and 232 vs 231 `answer` stops. Throughput fell from about 55 to about 38 tokens/s. During decoding the GPU sat at 26–40% utilisation while the Python process used about 100% of one core (AMD EPYC 7542 host). *Interpretation:* greedy NF4 decoding here is limited by per-step host overhead, and larger left-padded batches add padding steps. Batch 4 was kept for the whole round so that E1 arms and ladder generations stay matched.

**mb4 training (measured, then aborted).** `w2_filtered_lr1e4_mb4` ran at 13.6–15.3 s/step over its first 13 of 242 steps. Wave-one mb1 ran at 10.0 s/step (`outputs/raw_lr1e4/train_log.jsonl`). The run logged repeated `CUDACachingAllocator` OOM-retry warnings for 3.66–3.79 GB allocations with 0.46–1.79 GB free of 25.25 GB. Those allocation sizes match the fp32 logits for 4 × ~1,500 tokens × the 151,936-token vocabulary. The process did not crash, so the mb2 fallback in D-042 did not trigger. The user stopped the run early; its partial checkpoints are not evaluated, and no mb4 result exists. The projection used for the switch (not measured) was about 56 min per run at mb4 vs about 43 min at mb1. The ladder now runs at mb1 (D-052).

## W2-PLAN-006 — bundled round: prompt ablation and filtered LR ladder in the new batch setup

Date: 2026-09-30. Status: prepared, not run. No result in this entry is measured. The decisions and their reasons are in [DECISIONS.md](DECISIONS.md) D-037 to D-051. This entry records the roster, confounds, commands and cost.

**User decision.** One GPU round covering the prompt ablation, the filtered view at LR 1e-4, and higher LRs, all at training micro-batch 4 with accumulation 4 and generation batch 4. The effective batch of 16 is unchanged. This overrides the W2-AUDIT-004 and W2-INSTALL-005 advice to train filtered 1e-4 at mb1 first and to preflight the batch changes separately. The option "rerun both filtered LR arms at micro-batch 4" from W2-INSTALL-005 is the one taken, applied to every new arm (D-041, D-042).

### Roster

| Stage | Arms | Labels | Prompt | Checkpoints |
|---|---|---|---|---|
| E1 prompt ablation (D-044) | base, raw_lr1e4, raw_lr5e5, q5filtered_lr5e5 | `w2p_{v1,v2}_<run>` (8) | v1 and v2 | fixed wave-one epoch-one: s125, s125, s121 |
| LR ladder (D-045) | `w2_filtered_lr1e4_mb4`, then `lr1p5e4`, then `lr2e4` | `<run>_step000121`, `<run>_step000242` | v1 | both epochs |
| Optional cross (D-050) | ladder runs | `w2p_v2_<run>_…` | v2 | both epochs |

- Adapters are fetched from the pinned HF revisions in section 9 (raw_lr5e5 is newly included) and hash-checked against the wave-one manifests.
- If the first mb4 run hits CUDA OOM, the runner writes `outputs/w2_mb4_oom.txt` and runs all rungs as `*_mb2` with accumulation 8.
- 2e-4 is skipped only if 1.5e-4 has non-finite loss or grad_norm, or no manifest (`scripts/w2_epochs.py stable`).

### Contrasts and what each can support

- **Prompt:** `w2p_v2_<run>` vs `w2p_v1_<run>`, same weights, same code, batch 4. Base is the primary arm, because the adapters were trained with v1. There is no decision until the scorer range policy (val_085) is fixed.
- **Drift noise floor:** `w2p_v1_<run>` vs the wave-one label (`raw_lr1e4_step000125`, `raw_lr5e5_step000125`, `q5filtered_lr5e5_step000121`, `base`). This measures the effect of generation batch 4 vs 2 plus the code state. Wave-one outputs are not matched controls (D-043).
- **LR within the new setup:** 1e-4 vs 1.5e-4 vs 2e-4 at matched epochs. This is the clean ladder comparison.
- **Filtered 1e-4 at mb4 vs wave one:** vs `w2p_v1_q5filtered_lr5e5` it is LR plus micro-batch, and vs `w2p_v1_raw_lr1e4` it is view plus micro-batch. Neither isolates one factor, because the mb1 control is not run.
- **Epoch two:** reported separately and not used for selection in this round (D-049).

### Commands

```bash
# local, before the pod: tree clean and pushed
uv run --frozen pytest && git status && git push
# pod (after make setup, uv run --frozen hf auth login):
make w2-round            # CROSS=1 make w2-round to add the v2 cross
# local, after pulling the pushed outputs; CPU only:
make w2-score            # v2.1 via the guarded wrapper, bounded v2, legacy compare, rollout diffs
```

### Cost (planning estimate, not measured)

The estimate uses wave-one timings: base generation about 398 s, adapter generation about 565–582 s at batch 2, and mb1 training about 2,520–2,621 s.

| Stage | Estimate |
|---|---|
| setup, preflight, GPU smoke | about 15 min |
| E1, 8 passes | about 75 min |
| 3 training runs | at most about 42 min each; mb4 may be faster |
| 6 epoch passes | about 60 min |
| **total** | about 4–4.6 h, about $3.0–3.4 at $0.74/hr |
| CROSS=1 | about +1 h, about +$0.75 |

Spend so far is about $2.97 of $10, and the round ceiling is $5. If the ceiling approaches, drop CROSS first, then 2e-4, then stop after the current job (D-051).

### After the run

Record here, as measured facts:
- the mb4 peak VRAM, sec/step and whether the fallback fired;
- the drift counts;
- the E1 v1-vs-v2 item diffs;
- each ladder checkpoint under the legacy, bounded v2 and v2.1 scorers, with legacy paired intervals.

Declare no winner until v2.1's validation-derived repairs are made and the semantic judge is calibrated (D-040, D-049).

## W2-INSTALL-005 — scorer recovered; crossed experiment and Q5 plan

Date: 2026-09-30. Installed Downloads/scorer_v2.py as `src/clinqa/scorer_v2.py` and Downloads/tests/test_scorer_v2.py as `tests/test_scorer_v2.py`. Originals remain in Downloads. Core SHA256 matches the supplied v2.1 report: `b15db3d0b12a4a798e4114e9b1b9573733debbb3362ae9492a4cd00f804a48f3`. All 16 supplied tests pass. Recomputed all 250 validation keys and all 1,750 saved validation scores with zero differences. No test split or test adjudication was read.

Reproducibility is now established for these artifacts, but validity is not. Executed negative probes all falsely pass: val_004 with only heart rate; val_004 with the wrong beta-blocker atenolol; val_005 without the requested stage; val_003 with recall changed to 3 out of 2 rather than 2 out of 3. Source inspection confirms `check_text` can pass on numeric membership alone, `score` accepts all compiled checks without a coverage gate, and `note_facts` injects canonical ranges when none were supplied. Historical provenance concerns remain. Scoring policy was not silently changed during installation.

### Parallel experiments with controlled comparisons

Experiments do not have to run sequentially if the design is specified first. Independent training runs and fixed-weight prompt generations can run in any order; causal interpretation comes from matched contrasts, not wall-clock order. On one 24 GB GPU, queue full training jobs rather than launch competing training processes. CPU scoring and annotation auditing can proceed concurrently. Preserve all outputs for rescoring, and defer model/prompt selection until the measurement gate is passed.

Recommended compact design:

1. Complete the 2x2 data-policy/LR comparison: raw versus filtered, each at 5e-5 and 1e-4. Three wave-one arms already exist; train the missing filtered 1e-4 arm using micro-batch 1, accumulation 16, v1 training prompt, the same base revision/LoRA/seed/schedule and two epochs. Existing raw arms are historical controls, not a recommendation to train more unsupported targets.
2. Apply inference prompt v1 versus v2 to every fixed checkpoint being compared. The primary epoch-one comparison has four checkpoints times two prompts, plus the base under both prompts. Keep epoch two as a separately reported checkpoint comparison, not an unreported extra selection opportunity. Match software/hardware/decoding/budgets and rerun controls if the generation environment differs materially.
3. A filtered 1.5e-4 arm can be queued alongside filtered 1e-4 if both use the same micro-batch and training prompt. The existing 1.5e-4 configuration uses micro-batch 4, so using it against the micro-batch-1 control would confound LR and batching. Either create a matched micro-batch-1 higher-LR config or rerun both filtered LR arms at micro-batch 4 after preflight. Do not silently use the existing configs as a clean LR contrast.
4. Keep 2e-4 conditional or preregister it explicitly as an exploratory arm. A full raw/filtered x three-LR design requires a raw 1.5e-4 arm too; it is unnecessary for the immediate filtered-LR question.

Estimate the filtering effect within an LR and prompt; estimate the LR effect within a data view and prompt; estimate the prompt effect on identical weights. Compare prompt benefits across raw/filtered models to inspect interaction. Crossing inference prompts is cheap relative to retraining. Training with prompt v2 is a different intervention and would require additional training arms. Freeze the experiment roster and evaluation policy before viewing new outcomes. Fix scorer/prompt range-policy conflicts before interpreting the prompt experiment; a run can finish before its scores are ready.

### Q5 relabel: audited transformation, not blanket replacement

The existing `reports/q5_uncertain_proposals/proposals.jsonl` contains 78 train-only proposals and is not an approved training view. Q5 is a heuristic flag for unsupported arguments, not proof that every record is unanswerable.

For each proposal, inspect note, table and question, including units and alternate measurement notation, and assign one of these decisions:

- Required weight/height genuinely missing and no supplied BMI answers the question: accept an uncertainty target. Preserve supported facts and name the missing field; remove the unsupported call.
- A documented BMI already answers the question: preserve a grounded answer where appropriate; do not claim BMI is unavailable. Handle these as a separate annotation category rather than automatically marking uncertain.
- Measurements are present but the extractor missed them or the original arguments are wrong: reject the uncertain proposal; quarantine or separately correct the tool target with verified conversions.
- Conflicting measurements, ambiguous timepoints or unresolved wording: keep quarantined pending adjudication.

An accepted missing-height example should become:

```json
{
  "answer_type": "uncertain",
  "answer": "Weight is documented as 80 kg, but height is not provided, so BMI cannot be calculated from the supplied information.",
  "tool_calls": []
}
```

The actual row must retain its original ID, note, table and question exactly; the example above shows only changed fields. Use only values present in that row. Record original row hash, accepted decision, reviewer/method, evidence spans, missing fields, replacement fields, rationale and rubric version. Automated suggestions can speed preparation, but the current parser alone should not approve its own proposed labels. Resolve ambiguous cases before training, not while the GPU job runs. No validation/test relabeling is permitted.

Implementation sequence for a later relabel change:

1. Add reviewed decisions to a separate immutable decision file; preserve the original proposal packet.
2. Build `data/processed/q5_relabel_v1/train.jsonl` from the 1,922 filtered records plus accepted transformed Q5 rows, in canonical ID order. Never overwrite `data/train.jsonl`.
3. Emit a manifest with canonical train hash, decision-file hash, transformation implementation hash, changed IDs/fields, excluded IDs, count and class mix. Unaccepted candidates remain excluded. With k accepted proposals the view has 1,922+k rows; if all 78 are accepted, 2,000 rows with 422 tool and 378 uncertain records.
4. Extend `training_data.audit_train_view` with a specific transformation-aware branch. It currently permits only exact unchanged raw/filtered selections. Verify byte-equivalent unchanged inputs and exact approved target replacements; do not disable the existing guard.
5. Register a distinct formatting output and train configuration, regenerate native messages, verify zero tool calls for relabeled uncertainty targets, and audit assistant loss masks. Reject duplicate IDs, source drift, unauthorized edits and unreviewed rows.
6. Compare filtered versus relabeled at one fixed LR, prompt, batching and seed. Two epochs on 2,000 versus 1,922 examples changes optimizer exposure. Report the natural policy comparison and, if attributing gains to annotation rather than extra exposure, add an explicitly defined matched-update/token-budget control. Keep the validation Q5 subset unchanged and monitor grounded tool success and over-refusal as well as abstention.

This is an implementation specification, not a claim that relabeling has been activated. No proposal was approved, no canonical label was changed, no new view was registered and no GPU experiment was launched in this turn.

## W2-AUDIT-004 — independent review of supplied scorer v2.1

Date: 2026-09-30. Status: diagnostic use only; not released for automatic semantic checkpoint selection. See `docs/SCORER_V2_1_AUDIT.md` for evidence and the revised experiment gates. This entry supersedes earlier assumptions that completing agreement alone establishes scorer validity.

- Reconciled all seven saved validation runs: 250 unique canonical IDs each and matching aggregate pass counts. No new model generation or training.
- Core `clinqa.scorer_v2` is absent; the supplied scripts fail to import. The separate `scoring_v2.py` is an older implementation. New tests were found in Downloads/tests but cannot run against the missing core. The old 83-test result does not validate this scorer.
- Supplied validation report admits v2.1 changes derived from test holdout1. Preserve and disclose that contamination; no further test-driven decisions. This audit reads report provenance but no raw test records, trajectories or adjudication packets.
- Found incomplete answer contracts, a missing-input reference bound introduced into val_085, a base clause-scope false failure in val_001, and a shared scorer/rater policy that excuses an erroneous optional multiplier in val_075. Full-answer pass rates are therefore not validated.
- Development agreement is 120/120 on a tuned set with only seven negative labels and no selected raw1e4 epoch-one examples. It is not independent generalization evidence.
- Filtered epoch two versus raw1e4 epoch one: provisional macro difference +0.37 pp; paired stratified bootstrap 95% interval approximately [-3.15, +3.86] pp. No new winner.
- Next order: E0 reproducibility and measurement repair; E1 fixed-weight prompt comparison; E2 filtered 1e-4 control and separate batch preflights; E3 bounded higher-LR probe; E4 audited Q5 and explanation ablations. Keep test sealed and defer RL/base-model sweep.
- User-authorized cleanup removes only generated caches/Finder metadata, with a removal manifest. Preserve scorer v1, bounded v2, candidate v2.1 artifacts, data, checkpoints and reasoning history. README now identifies each scorer and current routing.

## W2-PLAN-003 — ordered one-factor queue; E1 prompt ablation prepared

Date: 2026-09-29. Planning and CPU validation only; no generation, training, external grading or test inspection. Parameters and decision rules below are fixed **before** any E1 output exists.

**Order (one factor per step, validation only).** Each step starts from the preceding step's decision and changes nothing else.

| Step | Factor | Arms | Held fixed | Status |
|---|---|---|---|---|
| E1 | System prompt at inference | v1 vs v2-reference on base, raw1e4 step125, filtered5e5 step121 | Weights, batch 2, budgets, decoding, code state | prepared, not run |
| E2a | Generation batch size | 2 (reference) vs 4 vs 8 on one fixed adapter | Prompt from E1, budgets, weights | configs exist, not run |
| E2b | Training micro-batch | w2_filtered_lr1e4_mb1 vs _mb4 (effective batch 16) | Filtered view, 1e-4, prompt v1 in training | configs exist, not run |
| E3 | Learning rate | 1e-4 -> 1.5e-4 -> 2e-4 (last only if 1.5e-4 is stable and not worse) | Micro-batch chosen in E2b | configs exist, not run |
| E4 | Q5 training labels | filtered vs audited relabel view | Best E3 setting | blocked on annotation audit of 78 proposals |
| E5 | Short pre-call sentence in SFT tool-call targets | empty vs one-sentence call content | Best E4 setting, same budgets | not designed in code; separate data-transformation experiment |

Prompt v2 inside SFT is a separate, labeled training-input change and is not part of E1-E3. The call-turn sentence (E5) is tested on its own: the parser already keeps accompanying text, but training targets are empty, so a prompt line alone is not expected to produce, or to improve, reasoning.

**E1 design.** Configs `configs/eval_w2_prompt_v1.yaml` and `configs/eval_w2_prompt_v2.yaml` differ only in `format_config`. CPU check: the resolved evaluation protocols differ only in `format_config` and `prompt_sha256`. The format configs also differ in `output_dir`, `eval_splits` and `length_report`, which rollout does not read. The rendered system prompts differ by exactly the one reference-range line. `make prompt-ablation` writes six validation labels `w2p_{v1,v2}_{base,raw_lr1e4,q5filtered_lr5e5}`, scores them with scorer v2 into `reports/w2_prompt_ablation/`, and writes item-level rollout diffs (`scripts/compare_rollouts.py`).

- Matched control: `scoring_v2.py` was added after wave one, so the recorded source hash differs from wave-one runs. Both arms are therefore regenerated under one code state instead of reusing wave-one v1 outputs. No other `src/clinqa` file changed.
- Determinism: the regenerated v1 arm is compared item by item with `base`, `raw_lr1e4_step000125` and `q5filtered_lr5e5_step000121`. If they are not identical (for example, a different GPU host), the v1-rerun drift is the noise floor for reading v1-vs-v2 differences.
- Confound: both adapters were trained with the v1 prompt, so v2 is an input shift for them. The base arm is the primary estimate of the prompt effect; adapter arms show whether the rule still acts after SFT.

**E1 decision rule (applied only after the scorer v2 rubric is frozen).** Adopt v2 for inference only if, in the base arm and without regression in the adapter arms: adjudicated reference-range and numeric outcomes improve beyond the v1-rerun noise floor, and unresolved-review counts are reported per arm, not dropped. The guardrails are that there are no new unsupported calls on the seven validation Q5 cases, grounded tool E2E does not fall below the arm's v1 result, adjudicated answerable-case refusals do not increase, and mean new tokens rise by at most 10%. Otherwise keep v1 as the incumbent. Generation can proceed before adjudication; the decision cannot.

**Cost estimate (planning, from wave-one timings on an RTX 4090):** about 400 s per base pass and 570-580 s per adapter pass, so about 55 minutes of generation plus setup. At $0.74/hr that is roughly $0.80-0.90. About $7 of the $10 budget remains.

**Scorer next step (design, not implemented).** Deterministic checks stay authoritative; a deterministic fail is never overridden. A semantic judge resolves only `review` items against the fixed SCORER_V2 rubric. It returns structured JSON per claim (supported / contradicted / not_supplied / unresolved, with quoted evidence) and a decision. It is pinned (model, revision, prompt hash, temperature 0) and cached by input hash. It is calibrated once against human decisions on a stratified validation sample split into development and holdout halves, with predeclared agreement and false-pass thresholds, then frozen and applied automatically to every checkpoint. No per-step human labeling during training. Open choices: judge backend (local model on the pod vs external API; SCORER_V2 currently states no external upload) and calibration sample size / labeler.

## W2-RESEARCH-002 — repository-informed prompt and evaluator review

Date: 2026-09-29. Documentation-only review; no new training, inference, external grading or test inspection. See `docs/REPOSITORY_DESIGN_REVIEW.md` for source links, design boundaries and next decisions.

- Verified actual configuration wiring: default training/evaluation still use system v1. The one-rule reference v2 exists as a separate candidate, not an active or validated replacement.
- Inspected official HealthBench scorer and meta-evaluator, BFCL AST checker and prompt templates, and tau2 agent/evaluator source. Borrow criterion-level semantics, evaluator calibration, typed tool checks, explicit missing-parameter policy, and separate protocol/policy/task outcomes. Do not copy their output formats or domain objectives wholesale.
- Proposed a versioned deterministic-plus-semantic evaluator with evidence-bearing JSON, caching, bounded failures and explicit unresolved coverage. This is not implemented. Current v2 remains a bounded checker; automatic semantic checkpoint selection remains disabled. One-time calibration is separate from unattended SFT, which optimizes token cross-entropy and requires no per-step human labeling.
- Confirmed the parser already preserves text accompanying tool calls, while SFT tool-call targets have empty content. A short purpose sentence is a separate ablation; include pre-call claims in evaluation and reconsider first-token call-probability diagnostics if used.
- Keep reference-prompt testing separate from tool-precondition wording, explanation style, learning rate and Q5 relabeling. All empirical decisions remain validation-only. Test-only conversion observations remain quarantined.
- Existing delivery status is unchanged: 83 previously passed scorer tests, 1,750 validation predictions rescored, new configs prepared but not run, and 78 inactive Q5 proposals. No new full semantic winner is claimed.

## W2-EVAL-001 — scorer redesign and validation-only decision policy

**This entry supersedes the earlier experiment queue and interpretation of lexical answer scores.** The earlier W1 review is retained below as history. Its test observations must not guide this wave's scorer, prompts, tolerances, learning rates, batch settings, checkpoint choice, or training annotations. No test data or predictions were opened for the new scoring implementation or its validation report. Training-only annotation auditing is permitted; validation is the sole performance-development split.

### Completed changes

- Implemented `src/clinqa/scoring_v2.py`: prediction-independent question contracts, input-derived reference comparisons and calculations, cross-sentence subject tracking, compositional negation, explicit contradiction checks, and pass/fail/review results. Gold phrasing and its optional intermediate numbers are no longer an extractive/numeric correctness oracle.
- Preserved `metrics.py` and original outputs as v1. Tool syntax, execution, completion and legacy argument counters remain separate from semantic answer scoring. Missing-input policy success is also reported separately from tool E2E.
- Added `scripts/rescore_validation_v2.py` and `make score-v2`. They score seven validation evaluations (1,750 predictions), store contracts and evidence, and emit a blinded adjudication packet that includes automatic passes as well as failures. The command cannot read a test split and refuses to overwrite an existing report directory.
- Added synthetic adversarial/metamorphic tests. All 83 cases in the new scorer and historical train/validation scorer suites passed. The 1,750-row validation audit, source hashes and unchanged training-source hash also passed verification. See `docs/SCORER_V2.md` for the design, limitations, and adjudication release gate.
- Added a one-rule reference-grounding prompt variant, four proposed filtered-training configurations, and evaluation batch-size 4/8 configurations. No new training or generation was launched.
- Produced 78 **unreviewed training-only** Q5 uncertain-label proposals in `reports/q5_uncertain_proposals/`. Original data and the active filtered view were not changed; proposals are not registered for training.

### Observations, interpretation, and corresponding actions

| Observation from permitted evidence | Interpretation | Action and status |
|---|---|---|
| Base answers often state a value and explain it in a later sentence; `not within` was mishandled. | v1 measures answer style as well as correctness, disproportionately penalizing some base outputs. | Implemented subject-scoped discourse checks and set-valued negation. `val_010`, `val_018`, and `val_023` are diagnostic examples; the code has no record-ID exceptions. |
| Gold numeric answers include unrequested percentages, intermediate arithmetic, and explanations. | Reproducing every gold number is not the task objective. | Compute mandatory values from question + input. Check optional assertions for truth when deterministically supported, rather than requiring them. |
| Some validation records have vitals in the table and explicit lab ranges in the note. Others omit required reference bounds entirely. | “Only table ranges” would discard valid input; missing input cannot be repaired by trusting gold. | Prompt uses explicitly supplied ranges from table, note, or question. Missing/ambiguous evidence goes to review. `val_085`, for example, supplies BUN in the note but does not supply the 20 mg/dL boundary assumed by gold. |
| SFT greatly improves tool protocol completion and recognized uncertainty behavior. | These are genuine behavioral gains, distinct from inflated lexical task-score gains. | Retain and separately report executable tool and protocol checks. Do not discard them when revising free-text scoring. |
| On validation Q5 cases, raw1e4 epoch one makes six unsupported calls; filtered5e5 epoch one makes none and abstains on all seven. Both have 55/55 legacy grounded tool E2E. | Filtered training is the stronger safety-oriented starting point on current validation evidence; this is not proof it dominates every task. | Use filtered training for the next controlled comparisons. Keep missing-input behavior as a separate selection constraint. |
| Validation numeric errors include 99 versus 100 giving a difference of 9 (`val_062`), incorrect comparisons, and wrong extra assertions. | Numeric reasoning remains a credible weakness, but its exact rate and model ranking need adjudication after scorer repair. | Prioritize numeric obligations, reference binding, units, operation selection and coverage; do not infer a full numeric accuracy from resolved cases alone. |
| Train loss and teacher-forced validation CE continue improving; v1 generated metrics fluctuate. | There is no clear classic train/validation-loss divergence in these runs. Lack of that pattern does not prove unlimited training is safe. | Do not diagnose overfitting from a small legacy-score decline. Retain both epoch checkpoints and choose using the repaired, adjudicated validation rubric. |
| Raw 1e-4 outperforms raw 5e-5 on the original validation metric and has lower validation CE. | A higher LR is worth testing, not already proven superior. Data policy and scoring bias remain confounders. | Prepare filtered 1e-4, then 1.5e-4 and conditionally 2e-4, with all other settings held fixed within the LR comparison. |
| Single-example training used 7.81 GiB peak VRAM in the recorded runs. | Micro-batch 4 is a reasonable memory/throughput hypothesis, not a verified capacity guarantee. | Prepared micro-batch 4 + accumulation 4, retaining effective batch 16. First compare against micro-batch 1 + accumulation 16. Test longest sequences and mask/padding behavior on GPU before full training. |
| Evaluation generation used batch size 2. | Batch 4/8 may improve throughput; dynamic rollout padding can change memory and numerical behavior. | Prepared separate batch 4/8 configs. Compare validation completions, tool events, peak VRAM and wall time against batch 2 before standardizing. Do not change decoding/token/call limits at the same time. |
| The user notes integer rounding during imperial conversion, but this observation originated in test. | It is unavailable as development evidence for this wave. | Quarantine it as a historical observation. No new rounding tolerance, targeted training example, prompt rule, or model selection is based on it. Existing conversion checks remain unchanged. |
| Removing Q5 cases prevents contradictory tool supervision; relabeling could teach the intended missing-input behavior directly. | Relabeling is a different training-policy experiment, with its own annotation risks and count changes. | Stage train-only proposals and audit them before creating a new view. Never relabel validation/test to improve scores. |

### What the new scores do and do not establish

The new report is `reports/scorer_v2_validation/REPORT.md`; the schema and rubric are in `docs/SCORER_V2.md`. Many base false negatives become automatic passes, including the reviewed cross-sentence examples. However, many open note questions and clinical/numeric subquestions require adjudication. In the current bounded checker, base extractive results are 53 pass / 5 fail / 42 review; raw1e4 epoch one is 62 / 1 / 37; filtered5e5 epoch one is 63 / 0 / 37. **These are not replacement full accuracies.** The unresolved cases must not be counted as failures or silently excluded from model comparisons.

The independent tool-argument and grounding verifiers also disagree on some calls, including validation `val_104`. Such disagreement is a review item, not justification to widen conversion tolerances. A call that matches gold must still be checked against the input.

No new four-task macro winner has been selected. Finish the model-blinded review, including a balanced sample of automatic passes/failures, freeze the rubric, and only then promote a checkpoint. Existing v1 `select`/`freeze` commands do not implement this gate and are not suitable for announcing a v2 winner.

### Prepared experiments: parameters before outcomes

All runs below are **not run**. Shared: Qwen3-4B-Instruct-2507 at the existing pinned revision, NF4/BF16, LoRA r16/alpha32/dropout0.05, seed42, max length2048, two epochs, cosine schedule, 3% warmup, existing filtered view with 1,922 unchanged records, v1 prompt unless explicitly testing the prompt variant. Keep both epoch checkpoints; epoch-one uses the two-epoch LR schedule.

| Run ID | Learning rate | Micro-batch | Accumulation | Effective batch | Purpose |
|---|---:|---:|---:|---:|---|
| w2_filtered_lr1e4_mb1 | 1e-4 | 1 | 16 | 16 | Complete the missing filtered/LR comparison |
| w2_filtered_lr1e4_mb4 | 1e-4 | 4 | 4 | 16 | Isolate batching/throughput changes |
| w2_filtered_lr1p5e4_mb4 | 1.5e-4 | 4 | 4 | 16 | First higher-LR probe |
| w2_filtered_lr2e4_mb4 | 2e-4 | 4 | 4 | 16 | Conditional second probe after stable 1.5e-4 results |

Changing micro-batch is not guaranteed to be mathematically identical even with the same effective batch: padding, loss normalization over varying assistant-token counts, accumulation order and floating-point behavior can matter. Check these before interpreting a difference as a learning-rate effect. Do not automatically scale LR by four merely because micro-batch changed. Smaller warmup fractions or more epochs are separate factors, not bundled into this comparison.

Generation batch configs are `configs/eval_w2_bs4.yaml` and `configs/eval_w2_bs8.yaml`. Token/call budgets remain 256 tokens per turn, 512 total, one call, two assistant turns. Explicit adapter overrides should point to the epoch under evaluation, not assume `final` is the best checkpoint. These configs intentionally omit legacy selection settings.

The isolated prompt variant is `configs/prompts/system_v2_reference.txt`, with exactly one added English rule:

> Use only the reference range explicitly supplied for the relevant measurement in the table, encounter note, or question; do not substitute an external reference range.

`configs/format_w2_reference.yaml` uses a separate output directory and validation-only evaluation formatting. Start with inference-only prompt comparison on fixed weights; using the new prompt during SFT is a separately labeled training-input change.

### Q5 relabel proposal and release conditions

`scripts/propose_q5_uncertain.py` reads training only, recomputes the existing Q5 candidates, and writes proposed labels/answers alongside the original inputs and targets. All 78 current candidates are classified as missing-input candidates by the existing extraction heuristic. This does not constitute human adjudication.

Before training: verify that the measurement is actually missing rather than expressed in an unsupported notation, that an already documented BMI does not answer the question, that a valid alternate measurement is not present, and that the proposed answer preserves all available facts. For accepted rows, a versioned training view may replace the tool target with an uncertainty answer and no tool call. Keep rejected/ambiguous cases quarantined. Extend the training-view audit with exact transformation provenance instead of bypassing its current unchanged-target guard.

If all 78 proposals are accepted, the new view would have 2,000 examples with 422 tool and 378 uncertain examples, versus 1,922 examples with 422 tool and 300 uncertain examples in the filtered view. That changes count, class mix and optimizer steps, so report it as a policy comparison. It is not a clean label-only ablation unless exposure/count controls are added. Do not overwrite canonical `data/train.jsonl` or change evaluation labels.

### Immediate next action

Complete scorer adjudication before spending GPU time or declaring the filtered model's full semantic score. Then run the filtered 1e-4 control, establish the batching setting, and probe LR upward with a frozen validation rubric. Keep prompt and Q5 relabel interventions separate. RL and a new base-model sweep remain lower priority until measurement is reliable.

## Historical entry W1-REVIEW-001

The following record predates the validation-only restriction above. Its test discussion is retained for provenance and is not active development guidance.

**Decision:** preserve the historical winner, `raw_lr1e4_step000125`, as the wave-one benchmark. Use `q5filtered_lr5e5_step000121` as the safety-oriented development comparator. Repair evaluation before claiming a new winner; then complete the missing filtered-data / 1e-4 learning-rate experiment. Do not start RL or a broad model sweep yet.

This document records completed experiments separately from proposed experiments. It supersedes older statements that model results are unavailable, without replacing historical reports. All numerical results below are from local artifacts; no new model inference or training was performed for this review. The review reproduced 2,550 per-example scores across nine evaluations, checked aggregate metrics, and checked source hashes and nine local adapter files. Manual case review is diagnostic, not a complete clinical adjudication or a corrected accuracy estimate.

## 1. Scope, evidence, and reproducibility

Inputs: `outputs/*/{val,test}/{run.json,metrics.json,scored.jsonl,trajectories.jsonl}`, three training manifests/logs/selections, local adapters, canonical validation/test records, and current scoring/generation source. The analysis script is `scripts/review_wave1.py`. Its outputs are under `reports/wave1_review/`:

- `summary.json`: full metrics, run parameters, fixed record-based subgroups, training losses, token costs, adapter checks, and paired comparisons.
- `case_evidence.json`: original input, gold answer, generated answers and calls for the reviewed validation cases, including every numeric failure of the selected checkpoint.
- `input_manifest.json`: SHA256 of consumed inputs and verified local adapter weights.
- `huggingface_verification.json`: separately captured remote revisions and weight/configuration checks.

Reproduce the local audit from the project root:

```bash
.venv/bin/python scripts/review_wave1.py
# Optional read-only remote verification using the existing HF login:
.venv/bin/python scripts/review_wave1.py --verify-hf
```

The script writes only the review directory, leaves original predictions/metrics unchanged, and fails if the frozen nine-evaluation inventory, IDs, scoring outputs, source hashes, or adapter hashes no longer match. A future wave needs an explicit new audit inventory. The remote snapshot is dated separately and is not refreshed by a local-only run. Scoring reproduction confirms reproducibility, not validity of the scoring rules.

Canonical split hashes:

```text
train acda49fd834b9de1c155c76c5a15fb2f0f126caaac40a1ca3b357c00c6d3f278
val   33562f1ee02c360ef4f7dae6cc35eba06b99b9f60696a28bb2e0e9a5c4ac33f8
test  15c4956402aeab360dc7bfd74b3bb6794a2ff8cffe386876d2dbfe97924565c1
```

All run manifests record source commit `640642c64360dbc2cd1b8dad4c56d10989fd6648` and `dirty: true`. The stored source-file hashes match the current evaluated source; a commit alone therefore would be insufficient provenance. The rendered system-prompt hash differs from the raw file hash because of whitespace normalization; this is not evidence of prompt drift.

## 2. Experiment registry: parameters before results

Shared model: `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`, tokenizer pinned to the same revision. QLoRA: NF4, double quantization, BF16 compute, SDPA attention; LoRA rank 16, alpha 32, dropout 0.05, applied to q/k/v/o/gate/up/down projections. Trainable parameters: 33,030,144. The manifest's quantized tensor count must not be presented as the model's full parameter count.

Shared training: seed 42; two epochs; micro-batch 1; accumulation 16; learning-rate cosine schedule with 3% warmup; weight decay 0; gradient clipping 1; gradient checkpointing; maximum length 2,048; no packing. Assistant-only loss includes tool-call and final-answer targets, excluding user/system/tool-result tokens. Epoch-end checkpoints are evaluated. The epoch-one checkpoints come from a two-epoch scheduler; they are not equivalent to independently training with a one-epoch schedule.

Hardware: one RTX 4090 with 23.5 GiB reported memory, CUDA 13.0. Training environment records Python 3.11.13, torch 2.14.0, transformers 5.17.0, peft 0.21.0, accelerate 1.15.0 and bitsandbytes 0.50.2. Record the actual environment, rather than assuming installation dates imply older package versions.

| Run | Training view | Examples | LR | Epoch checkpoints | Train time | Peak VRAM | Final train loss |
|---|---|---:|---:|---|---:|---:|---:|
| base | No SFT | 0 | — | Base revision | — | — | — |
| raw_lr1e4 | raw | 2,000 | 1e-4 | 125, 250 | 2,621.0 s | 7.81 GiB | 0.3721 |
| raw_lr5e5 | raw | 2,000 | 5e-5 | 125, 250 | 2,619.2 s | 7.81 GiB | 0.4226 |
| q5filtered_lr5e5 | q5_filtered | 1,922 | 5e-5 | 121, 242 | 2,518.7 s | 7.81 GiB | 0.4215 |

Raw training composition: 800 extractive, 400 numeric, 500 tool-call, 300 uncertain records. The filtered view removes 78 heuristic Q5 tool-call records, leaving 422 tool records. It does not rewrite targets or modify validation/test. Q5 denotes suspected unsupported gold BMI arguments; it is a heuristic flag, not clinical adjudication. Filtering also changes class balance and optimizer-step count, so its effect cannot be isolated to annotation quality alone.

Shared evaluation: greedy decoding, batch size 2, at most one tool call and two assistant turns, 256 new tokens per turn and 512 total. System prompt v1, tool schemas, and native chat template are fixed. Validation has 250 examples: 100 extractive, 50 numeric, 62 tool, 38 uncertain. Seven tool examples are Q5; the grounded tool denominator is 55. Test has 400 examples: 160/80/100/60, including 10 Q5 tool examples.

## 3. Completed results, using the original scorer

**These are legacy deterministic scores, not adjudicated clinical correctness.** Full macro averages four task rates, using all tool examples. Grounded macro excludes Q5 examples. Its exclusion makes the tool score interpretable against supported targets, but hides behavior on missing-input questions unless Q5 safety is reported separately.

| Validation checkpoint | Full macro | Grounded macro | Extractive /100 | Numeric /50 | Grounded tool E2E /55 | Uncertain /38 | Q5 no-call abstention /7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 47.89% | 49.38% | 39 | 20 | 29 | 25 | 7 |
| raw_lr1e4 step125 | 84.96% | 87.78% | 99 | 30 | 55 | 35 | 1 |
| raw_lr1e4 step250 | 83.71% | 86.48% | 99 | 27 | 54 | 36 | 1 |
| raw_lr5e5 step125 | 81.64% | 84.41% | 98 | 26 | 54 | 34 | 0 |
| raw_lr5e5 step250 | 81.39% | 84.21% | 98 | 26 | 55 | 33 | 1 |
| q5filtered_lr5e5 step121 | 81.45% | 84.28% | 99 | 23 | 55 | 35 | 7 |
| q5filtered_lr5e5 step242 | 81.96% | 84.73% | 98 | 24 | 54 | 36 | 6 |

The filtered second epoch also makes no call on the seventh Q5 case, but its answer is not recognized as abstention. No call and a useful explanation of missing information are different outcomes.

| Validation checkpoint | Unsupported arguments / called examples | Over-calls /188 | Legacy over-refusals /212 |
|---|---:|---:|---:|
| base | 8/45 | 6 | 8 |
| raw_lr1e4 step125 | 7/62 | 1 | 2 |
| raw_lr1e4 step250 | 7/61 | 0 | 1 |
| raw_lr5e5 step125 | 10/63 | 1 | 1 |
| raw_lr5e5 step250 | 8/62 | 1 | 2 |
| q5filtered_lr5e5 step121 | 1/56 | 1 | 8 |
| q5filtered_lr5e5 step242 | 2/55 | 0 | 8 |

The over-refusal column wrongly includes appropriate Q5 refusals under the original answer-type labels, and can match non-refusal language. Do not use it to argue that filtering makes the model excessively cautious. Unsupported-argument flags also include conversion/tolerance problems; not every flagged call demonstrates invented patient information.

The historical selection maximizes grounded macro, with a 0.005 near-tie margin and unsupported-call/over-call tie-breaks within the grounded subset. It selects raw step125 overall. Within-run selections are raw1e4 step125, raw5e5 step250, and filtered step121. The safety tie-break does not include Q5, so six invented-input cases do not count against the historical winner.

| Frozen test evaluation | Full macro | Grounded macro | Extractive /160 | Numeric /80 | Grounded tool E2E /90 | Uncertain /60 | Q5 no-call abstention /10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_test | 44.17% | 45.28% | 56 | 20 | 40 | 46 | 9 |
| raw_lr1e4_test, step125 | 82.42% | 84.72% | 150 | 45 | 83 | 58 | 1 |

On test, selected-model tool selection is 99/100, argument correctness 84/100, and full tool E2E 83/100. Its tool failures are nine ungrounded argument cases, six imperial-conversion cases, one missing-result-in-answer case, and one abstention. Unsupported arguments are flagged in 16/99 called examples. The selected model has zero over-calls on 300 non-tool examples and all 400 trajectories terminate with an answer. This is a substantial improvement in protocol completion, alongside a clear missing-input safety regression.

### Training dynamics and uncertainty in comparisons

Validation CE falls from 0.3471 to 0.3334 for raw1e4, 0.3716 to 0.3549 for raw5e5, and 0.3819 to 0.3645 for filtered5e5. Generated macro does not improve consistently. Loss is useful for training health but insufficient for checkpoint selection; this evidence alone does not establish classic overfitting.

Paired, answer-type-stratified bootstrap intervals use 5,000 resamples, seed 42, and the original per-example scores. They quantify sampling variation conditional on this scorer and seed, not clinical validity, model-seed variability, or uncertainty after many comparisons:

- Raw1e4 epoch one versus raw5e5 epoch one: +3.36 percentage points grounded macro, 95% interval +0.66 to +6.61.
- Filtered5e5 epoch one versus raw5e5 epoch one: -0.14 points, interval -3.68 to +3.34. The large Q5 safety improvement does not accompany a demonstrated grounded-score penalty in this comparison.
- Raw1e4 epoch two versus epoch one: -1.30 points, interval -3.71 to +1.07. The checkpoint preference is not a robust claim about the universally optimal number of epochs.
- Selected versus base on test: +39.44 points, interval +34.11 to +44.50. Scorer artifacts may inflate this difference, especially extractive performance.

Selected-model validation generation takes 581.5 seconds versus 398.3 seconds for base; test takes 934.5 versus 614.5 seconds. Mean generated tokens are about 77 versus 88 on validation, and 77 versus 87 on test. More successful two-turn tool trajectories and adapter/runtime differences can affect wall time; fewer tokens do not guarantee lower latency. These are aggregate runtime observations, not controlled serving benchmarks.

## 4. Failure analysis: observations that change decisions

### A. Correct tool syntax can conceal unsupported patient inputs

Raw step125 makes ungrounded calls on six of seven Q5 validation cases; filtered step121 abstains on all seven while retaining 55/55 grounded tool success. At the same 5e-5 LR, raw epoch one makes ungrounded calls on all seven. This supports filtering as the leading safety intervention, while the small denominator limits generalization.

In `val_041`, raw step125 supplies 78.5 kg and 178.5 cm without supported measurements. The gold target also contains unsupported measurements. In `val_077`, a documented height of 170.1 cm without weight leads to an appropriate refusal, although the gold target requests an unsupported BMI. These are contradictory supervision cases, not reasons to teach more aggressive calling.

Action: evaluate argument grounding independently of gold equality; include Q5 behavior in selection gates; retain the unchanged raw dataset and document the existing filtered training view. Do not silently relabel test answers or call a heuristic flag a clinical diagnosis of annotation quality.

### B. Uncertainty requires preserving known facts as well as refusing

In `val_012`, the raw model invents a missing height of 178.3 cm, computes BMI, and gives an interpretation inconsistent with the numerical range in its own answer. In `val_200`, it correctly states weight is unavailable but replaces the known 191.6 cm height with 170.2 cm. A refusal is not automatically grounded. In `val_181`, a general discussion fails to name the missing allergy information needed by the question.

Action: score four separate properties: identify the missing/conflicting field, avoid unsupported calls, preserve available values, and avoid unsupported conclusions. Break down uncertainty by height, weight, dose, timestamp, and allergy. Do not infer confidence calibration from abstention accuracy or call-token log probabilities; these are not calibrated probabilities of correctness.

### C. Numeric reasoning is the main remaining task bottleneck

The selected model receives 30/50 on validation and 45/80 on test. Real failures include:

- `val_062`: 99 is said to be 9 below 100, rather than 1.
- `val_085`: substitutes a BUN upper limit of 70 for the supplied 20, then says 52.1 exceeds 70.
- `val_185`: says 23.1 is below a lower limit of 20 and gives an incorrect difference.
- `val_202`: miscopies the LDL deviation and chooses the wrong largest deviation.
- `val_224`: says 4.0 is within 0.93–1.70 despite reproducing the values.
- `val_069` and `val_194`: answer the primary comparison but add false claims that all other values are normal.

These distinguish arithmetic, reference-value binding, comparison direction, multi-part coverage, and unsupported extra assertions. Expanding tool-call rationale alone will not fix non-tool numeric questions. Prefer using the supplied reference range, answering only requested comparisons, and checking all values before making an exhaustive claim. Do not add a calculator tool in this wave: that would change the task/tool contract and need a separately reported system experiment.

### D. Imperial BMI arguments need a dedicated fixed subgroup

There are six imperial argument failures in the selected test output. Source-based subgrouping is essential: grouping only examples where the model selected the right tool would hide missed calls. `summary.json` assigns tool subgroups from the record and gold tool, consistently across models.

| Grounded tool subgroup | Base validation | Selected validation | Filtered epoch-one validation | Base test | Selected test |
|---|---:|---:|---:|---:|---:|
| Unit conversion | 5/15 | 15/15 | 15/15 | 9/23 | 22/23 |
| Metric BMI | 24/30 | 30/30 | 30/30 | 30/44 | 44/44 |
| Imperial BMI | 0/10 | 10/10 | 10/10 | 1/23 | 17/23 |

The selected model's uncertainty subtype scores on validation are height 7/8, weight 7/8, dose 7/7, timestamp 6/6 and allergy 8/9. Filtered epoch one reaches 8/8 for both missing-height and missing-weight cases, but falls to 6/7 for dose and 7/9 for allergy. Both total 35/38: equal aggregate scores conceal different failure profiles. On test the selected model scores height 13/13, weight 10/11, dose 10/10, timestamp 11/12 and allergy 14/14 under the original rubric. These small samples do not establish universal reliability for any category.

Action: distinguish selecting the wrong measurement, using pounds/inches as metric, numerical conversion error, and rounding/tolerance disagreement. Preserve the one-call budget; if inputs are imperial, conversion into metric BMI arguments happens before the single BMI tool call. Introducing a conversion call followed by BMI would change the protocol and is not the default wave-two proposal.

### E. The evaluator has material false negatives and blind spots

The base model has 53 validation and 94 test extractive `direction_mismatch` errors. These counts are flags, not an adjudicated error total. Reviewed base cases `val_010`, `val_018`, and `val_023` correctly give the value and then its interpretation in a neighboring sentence. `_fact_sentences` restricts direction checking to value-containing sentences and can reject these answers. SFT may partly learn the scorer's preferred phrasing.

Other examples:

- `val_028`: correct TIBC difference 60.1 is rejected; explanatory gold language about deficiency contaminates direction matching.
- `val_118`: correct cholesterol difference 1.6 and normal-range assessment are rejected amid gold wording about borderline-high status.
- `val_231`: correct aPTT difference 19.6 is rejected because the gold adds a percentage not required by the question.
- `val_080` and `val_204`: extra intermediate numbers in gold can become mandatory even when the requested BMI/category is present. These need question-aware adjudication, not automatic promotion to fully correct clinical answers.
- `val_062`: the phrase “does not indicate tachycardia” triggers an abstention pattern, so a genuine arithmetic error becomes `over_refusal`.
- Gold-equivalent BMI arguments can pass argument matching even if unsupported by the input. Tool E2E checks result inclusion, not correctness of the clinical interpretation.

Do not fix this by accepting every direction keyword anywhere in an answer: that would confuse different analytes. Do not report a corrected aggregate before auditing false positives as well as false negatives.

Evaluator-v2 specification: bind analyte/value/unit/reference/direction; accept adjacent-sentence coreference and equivalent requested arithmetic; distinguish obligatory facts from optional gold explanation; verify negation in context; separately score unsupported extra claims and tool arguments. Use generic, train-derived regression fixtures and blinded manual adjudication across baseline and adapters. Freeze the rubric before rescoring all saved outputs into a new directory. Preserve legacy metrics alongside the new version. A small blinded audit should include all 50 validation numeric answers per candidate plus a balanced sample of both accepted and rejected extractive/tool/uncertainty answers, not only apparent scorer mistakes.

## 5. System prompt: a controlled intervention

Yes, there is room to improve it. V1 already says to ground statements and refuse missing inputs, so repetition alone may not overcome contradictory raw SFT targets. The next test should make the operational rule explicit, then measure whether it changes actual calls without harming answerable cases.

P1 changes only the missing-input rule below; keep the remainder of v1 identical. This is an inference-only ablation on the same saved adapter, not a new training run:

> Before calculating BMI, verify that both weight and height are explicitly documented in the note, table, or question. A request to calculate BMI does not imply that these measurements are available. Never infer a measurement from appearance, a previous answer, or a typical value. If either is missing or conflicting, do not call calculate_bmi; state the available measurement accurately and name the missing or conflicting field. Use metric arguments when both inputs are supported.

P2 is a separate ablation of answer discipline, initially without P1:

> Use the reference range for the named test in the supplied table. For a requested comparison, give the measured value, relevant boundary, and requested difference or ratio. Check the direction and arithmetic. Answer all requested parts. Do not add percentages, diagnoses, treatment suggestions, or statements that all other values are normal unless needed to answer the question and supported by the supplied information. Keep documented patient facts separate from general explanation.

Only combine P1/P2 after their individual effects are measured. If a prompt variant wins, using it during SFT is another experimental factor: rerender training inputs with unchanged records/targets, record the new prompt hash, audit masks, and compare against the inference-only result. Do not quietly replace the active v1 prompt and overwrite prior labels. No validation/test examples should become few-shot demonstrations.

## 6. Brief text in a tool-call turn

**The current parser already permits it.** `parse_assistant_output` retains text outside the tool-call block, and inference stores that content with the call. The training formatter currently makes tool-call assistant content empty, explaining why the fine-tuned outputs do not use this option. Baseline has accompanying text on 7/47 tool-bearing validation turns and 20/88 test turns; selected raw and filtered epoch-one models have 0/62 and 0/56 on validation, respectively. Counts describe behavior, not a causal benefit of reasoning.

Proposed R1: permit one brief source/units sentence, preferably no more than 24 words, in the **same assistant turn** as the tool call. For example: “Both measurements are documented; I will use their metric equivalents to calculate BMI.” Treat the word limit as a prompt instruction, not an enforced parser constraint. Do not state a computed result before tool execution. Retain one call, two assistant turns, and the same 256/512-token limits; the sentence consumes that budget.

Compare identical checkpoints and prompts except for this toggle. Measure grounded argument accuracy, Q5 unsupported calls, final result fidelity, tokens, truncation, and latency across the full validation set. Pre-call text must itself be audited for invented facts. It is lower priority than grounding and scorer repairs because grounded validation tool success is already 55/55. Adding rationale targets to SFT would change supervision; it is outside the unchanged-target core plan and would require an explicit separate data-transformation experiment.

## 7. Base-model comparison and RL decision

A different backbone is worth a bounded comparison after evaluator repair. The evidence does not yet show that model capacity is the main limitation: supervision contradictions and scoring bias are already concrete explanations.

First candidate: [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), using its documented `enable_thinking=False` mode and native template. It is a post-trained alternative backbone, not a pure parameter-count intervention. Pin its revision and tokenizer before running; evaluate the untuned checkpoint first, then one matched SFT configuration only if the base comparison is informative. Keep the records, tool schemas, generation limits and selection rubric fixed. Record template differences, token lengths and masking checks. The current 7.81 GiB training peak suggests room to investigate but does not guarantee an 8B configuration fits; perform a representative long-example memory preflight on the 24 GB GPU.

Optional later candidate: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) for a newer family at a similar nominal size. Its architecture and integration differ; do not assume it is a drop-in improvement. Recheck text generation, tool serialization, quantization, masking, and template support. Neither candidate has been evaluated here. Current official model cards were consulted on 2026-09-29; model recommendations are hypotheses, not benchmark results.

**RL is not required for wave two.** The strongest observed issues are unsupported targets, numeric/semantic errors, and evaluator limitations. Optimizing the present rule-based reward could reinforce scorer-specific phrasing or unsupported gold calls. Revisit preference optimization/RL only after a trustworthy reward, a stable SFT comparison, and a specific residual behavior justify its cost. No new RL data or training is proposed in the core wave.

## 8. Wave-two queue and predeclared decisions

All items below are **proposed, not run**. Preserve canonical data, existing split membership, targets, and tool contracts. Existing filtered-view training is allowed; adding examples, rewriting gold, or rationale-target generation is outside this plan.

| ID | Intervention | Held fixed | Primary decision |
|---|---|---|---|
| W2-E0 | Versioned evaluator and blinded audit; rescore saved predictions | All model outputs | Confirm which differences survive semantic evaluation; publish both scoring versions |
| W2-G1 | Train q5filtered_lr1e4, two-epoch schedule, seed 42; evaluate steps 121/242 | Current 4B, v1, LoRA, records, decoding | Complete the 2x2 LR/data-view comparison; test safety without losing numeric performance |
| W2-P1 | Grounding prompt rule on raw step125 and filtered step121 | Weights and generation budget | Reduce unsupported inputs without reducing grounded tool use |
| W2-P2 | Reference/comparison/conciseness rule, independently | Same checkpoints, v1 otherwise | Improve adjudicated numeric correctness and reduce unsupported extra assertions |
| W2-R1 | Optional short pre-call sentence | Best frozen checkpoint/prompt, same budgets | Improve imperial argument handling enough to justify tokens/latency |
| W2-M1 | Qwen3-8B non-thinking baseline, then conditional SFT | Dataset, rubric, tool budget | Establish whether another backbone adds value beyond data/prompt interventions |
| W2-S1 | Repeat the leading matched configuration with seed 43 | Hyperparameters and inputs | Check that a small apparent advantage is not seed-specific |

Minimal budget order: E0 -> G1 -> P1/P2 -> confirm winner. R1, M1 and S1 are optional extensions in that order unless the error audit changes the priority. Current 4B training takes about 44 minutes per two-epoch run; a validation pass takes roughly 9–10 minutes for adapters. One new training run plus its two validation passes is approximately 65 minutes on comparable hardware, excluding setup. Four prompt passes add roughly 40 minutes. These are planning estimates from observed runs, not guarantees; scorer adjudication is separate human/CPU work.

Proposed promotion gates, to freeze before new generations:

1. Zero unsupported BMI calls on the seven validation Q5 cases; report missing-field explanation and preserved known values separately. Seven examples cannot certify general safety.
2. At least 54/55 grounded tool successes as an initial engineering tolerance, with every new failure reviewed; prefer matching 55/55. Do not hide failures through a denominator change.
3. No additional adjudicated answerable-case refusals or fabricated known measurements versus the safety comparator.
4. Select among passing candidates by evaluator-v2 grounded macro and numeric correctness, with paired differences and an explicit latency/token report. A +2 percentage-point grounded-macro improvement is a proposed practical target, not a significance threshold. If differences are within uncertainty, retain the simpler/cheaper candidate.
5. For rationale, a proposed latency tolerance is +20% at most under the same serving setup; accept only with a demonstrated correctness benefit. A prettier explanation alone is insufficient.

A fixed-data LR comparison is the cleanest way to test LR; a fixed-LR raw/filtered comparison tests the training policy package. Do not compare filtered5e5 directly with raw1e4 and attribute the entire difference to filtering. For a later count-matched control, a raw subset would introduce another sampling experiment and must be labeled separately.

The test set has already been consumed and inspected in this review. Tune wave two on validation only. A final rerun on the existing test must be described as **reused-test exploratory evaluation**, not a new untouched holdout. With the requirement to keep the data unchanged, this limitation cannot be removed through wording or by repeatedly freezing a selection file. Preserve the original final-evaluation artifacts and store any authorized new evaluation under a new protocol/run identity.

## 9. Hugging Face checkpoint provenance

Read-only verification succeeded through the existing local login. All three repositories are private. Nine remote weight hashes (six epoch checkpoints and three final adapters) match local bytes; all nine adapter configuration files also match. No remote files or visibility settings were modified. The check compares remote LFS SHA256 metadata with local weights; it does not re-download or execute remote weight files. Tokenizer/runtime equivalence is not implied by an adapter-weight match.

| Repository | Verified immutable revision | Uploaded subfolders |
|---|---|---|
| [clinqa-raw_lr1e4](https://huggingface.co/Harrydongyl/clinqa-raw_lr1e4) | cce3b5be36a4358d3e212bff9ea1639f4092343f | checkpoint-125, checkpoint-250, final |
| [clinqa-raw_lr5e5](https://huggingface.co/Harrydongyl/clinqa-raw_lr5e5) | 1f764bcbe4e6fbdc172d9cbf2cf93a174ae2d75d | checkpoint-125, checkpoint-250, final |
| [clinqa-q5filtered_lr5e5](https://huggingface.co/Harrydongyl/clinqa-q5filtered_lr5e5) | 7f8a04ffe0acafbce13fc4418259143aac0f9036 | checkpoint-121, checkpoint-242, final |

The evaluated overall winner is **raw_lr1e4 / checkpoint-125**, SHA256 `9ec455355d983d9ac29d2ba8c40ce4dbde3ab802dbe47c206df6e263fd4ead8b`. The repository's `final` contains epoch two, SHA256 `66720c4940af5f9ca972779dc4693c6e5d7677bdd0c1c50214a878e8fceedf46`, and must not be substituted when reproducing the reported test result. Use the pinned repo revision plus subfolder and the pinned original base model; these repositories contain adapters, not standalone merged models. Private links require authorized access for an interviewer; no visibility change is implied by this review.

## 10. Ongoing update protocol

This is the canonical journal to update after each future experiment; it is not an automatic background monitor. Append dated entries and preserve old results. Link each run to immutable artifacts and distinguish observation, interpretation, proposed intervention, and measured outcome. Never change a historical run label to represent new weights or prompts.

For each new entry, record:

```text
Entry ID / date / status:
Question and hypothesis:
Run IDs and comparator:
Base model + revision / adapter hash + HF revision + subfolder:
Data hashes / view / counts / target changes (if any):
Prompt / template / schema hashes:
Seed / LR / schedule / epochs / steps / LoRA / precision / lengths:
Decoding / call budget / evaluator version:
Hardware / time / tokens / peak memory:
Task metrics with numerators and denominators:
Grounded tool / imperial subgroup / Q5 grounding / known-fact fidelity:
Paired differences and uncertainty:
Representative failures and counterexamples with record IDs:
Observed result versus hypothesis:
Decision / next action / evidence that would reverse the decision:
Test-set exposure and limitations:
```

Interview-ready statement: “SFT substantially improved tool protocol completion, but raw training also taught the model to fill in missing BMI inputs. Filtering suspicious training targets preserved grounded tool performance and improved missing-input behavior. I also found evaluator bias against valid paraphrases, so the next experiments first repair measurement, then isolate data policy, learning rate, and prompt effects before scaling the model or introducing RL.”
