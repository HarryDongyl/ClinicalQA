# Wave 3 approval review

Date: 2026-10-01. Reviewer: Codex AI grounding reviewer, acting on the user's request to review the four approval items. Scope: input-grounding and experimental artifact review, not human clinical adjudication. No GPU training, inference, upload or push was performed. No F-s42 release approval was issued.

## Decisions

### Q5 relabels: approved after 14 answer revisions

Accepted 78, rejected 0, undecided 0. The derived view contains 2,000 records: 800 extractive, 400 numeric reasoning, 422 tool-call and 378 uncertain. The 1,922 retained filtered records are unchanged. Canonical notes, tables, questions, IDs and split files are unchanged. The original unsupported tool targets are replaced only in the derived relabel view.

Review combined the canonical train inputs, every candidate question, body-measurement/BMI concordances using an independent broad lexical search, table contents/labels, the grounding parser and source-context review of ambiguous measurements and compound questions. Parser absence alone was not treated as sufficient evidence. This was a targeted grounding review, not a clinical review of every statement in each synthetic encounter. Source-input hashes, quoted evidence and per-record decisions are in reports/w3/q5_approval_evidence.jsonl and configs/w3/q5_relabel_review.jsonl.

No candidate provides an absolute body weight and height, or a numerical BMI that answers the question. The following distractors require particular care:

- train_516, train_746 and train_1683 report weight changes (5 kg, 8 lb and 8 kg), not current absolute weights. Their new answers preserve the documented changes and explain why BMI remains unavailable.
- train_1296 reports a 37 cm neck circumference; this is not body height.
- Wound dimensions, calf-circumference differences and walking distances are not BMI inputs.
- Descriptions such as obese, thin or elevated BMI do not establish a numerical BMI. They must not be erased as documented observations or converted into invented measurements.

Revised the 11 compound-question targets train_269, train_290, train_827, train_961, train_1203, train_1350, train_1514, train_1594, train_1716, train_1946 and train_1973. They now distinguish the unavailable calculation from documented context. For example, train_269 preserves both the history of obesity and the conflicting underweight concern without resolving the contradiction by invention. Together with the three weight-change cases, 14 targets changed from the draft; 64 retain the original abstention target.

All review rows identify the reviewer as AI. Corrected derived-view metadata so completion of an AI review does not automatically become manual_review_complete=true or clinical_adjudication=true. Added regression assertions for that distinction. This metadata change does not change the 4B serialization or loss contract.

### Prompt v3: approved with a narrow clarification

The draft is a reasonable fixed stronger baseline: it asks for input-grounded facts, complete answers, explicit missing information, valid tool inputs and actual tool results. Existing general-reference attribution policy was retained; this approval does not resolve scorer disagreements over unavailable reference ranges.

Clarified two related phrases before freezing:

- Convert documented imperial measurements while preparing calculate_bmi arguments, without a separate conversion-tool call. This matches the one-call budget and the assignment's implicit-conversion contract.
- Prohibited ungrounded parameters while explicitly permitting deterministic unit conversions of documented values. The prior literal wording could be read as prohibiting converted numbers because they do not appear verbatim in the input.

Conversion constants already appear in the assignment; no rounding rule, imperial rationale target or other intervention was derived from test outputs. Prompt v1 and all 4B SFT training prompts remain unchanged. Prompt v3 is one combined intervention; this experiment cannot attribute gains to individual prompt sentences.

### Four-shot demonstrations: approved unchanged

IDs and order remain train_1523 (extractive), train_1771 (numeric), train_851 (tool) and train_802 (uncertain). Verified the deterministic selection rule, canonical source/message equality and the recomputed tool result. Question-relevant checks:

- Ferritin 6.0 ng/mL is below the supplied lower limit 12.
- LDL 95.2 mg/dL is 4.8 below 100; all four listed lipid values satisfy their respective supplied thresholds.
- BMI uses the documented 93.4 kg and 191.3 cm and the executor returns 25.5.
- Thyroid values are documented, but collection date/time is absent; temporal uncertainty is appropriate.

This approval concerns demonstration targets and grounding, not endorsement of every unrelated clinical claim in the input notes. Demonstration messages were not rewritten. After the prompt clarification, validation first-turn lengths are min 4,382, median 4,479 and max 4,639 tokens. These are inference contexts, not 2,048-token training examples. Longer-context GPU memory and throughput remain untested. The 200 train-fit IDs are unique, contain 50 examples per type and exclude all four demos.

### P1 probes: approved and frozen unchanged

Retain 34 probes from 40 source candidates, with the same six exclusions: val_017, val_037, val_139, val_140, val_189 and val_190. Their questions contain the measurements; exclusion is a conservative pre-generation choice, not outcome-driven selection. The resulting set does not cover the explicit-measurement-in-question stratum, which must be disclosed.

Reviewed all deletion diffs, including the 29 mid-line edits, and reran every probe's automatic grounding/residual checks. No clinically relevant non-anthropometric clause was removed in the displayed diffs. Three edits (val_052, val_086, val_166) also remove the initial 'Exam:' label; the subsequent vitals and findings remain intact and readable. This minor presentation change does not justify rebuilding the frozen set. Qualitative body habitus may remain: it does not license inventing a numerical BMI.

The 34 original partners remain available in matching full-validation outputs. Compound questions still require context-sensitive uncertainty; the automatic missing-word check alone is not a complete semantic endpoint. With 34 edited probes, the 5% point gate permits at most one fabrication. Zero of 34 still has an upper 95% Wilson bound of approximately 10.2%, so passing is not evidence that the population error rate is below 5%.

## Verification actually run after changes

- Full local suite: 254 passed, 4 non-failing warnings, 52.48 seconds. No skips or failures. Warnings relate to the tiny CPU resume-test model and pinned memory without an accelerator.
- Wave3 formatting: filtered 1,922, relabeled 2,000 and validation 250 records; no overlength records. Maximum training length remains 1,614 tokens.
- Relabel native-template mask audit: 2,000 records, zero problems.
- Transformation-aware training audits pass for relabel seeds42/43/44 and the filtered 8B configuration. All return clinical_adjudication=false.
- Readiness with all required items: prompt_v3, fewshot, p1, trainfit and relabel are approved; accepted=78, rejected=0.

Used the existing .venv interpreter with HF_HUB_OFFLINE=1 for tokenizer-dependent checks. Unit tests include structural/hash checks of canonical splits; no test-model performance was inspected for approval decisions. Canonical data remains intact. The 8B tokenizer/template was exercised by the local suite; full 8B GPU behavior was not certified.

## Separate limitations that these approvals do not clear

The new data/prompt/probe approvals are not a model-quality approval. Keep gate_fs42.json absent/unapproved until outputs exist and are reviewed. Historical scorers remain diagnostic where semantic coverage is incomplete.

Additional adversarial helper checks (reports/w3/gate_review_caveats.json), using synthetic fixtures rather than model outputs, expose limitations in scripts/w3_analyze.py:

- A fabricated weight passed to unit_convert is marked called=true but fabrication=false, because call fabrication currently checks calculate_bmi only. Probe intended behavior says no tool call; a wrong-tool call must not evade that endpoint.
- The helper inspects final_answer rather than every visible assistant claim. A synthetic earlier-turn claim is therefore ignored. This is helper-level coverage evidence, not proof that this exact fixture is reachable under the current rollout stopping rules.
- An explicitly hypothetical example containing measurements is marked as patient fabrication. Numeric mentions need attribution/context review.

Do not automatically approve F-s42 from that summary. Before seed continuation, inspect all 34 probe responses (not only flagged ones) and all seven natural-Q5 responses against their inputs, including every call/argument and every visible assistant turn. Correct false positives/negatives in a separately versioned adjudicated report, preserve raw diagnostic scores and keep the predeclared threshold unchanged. Alternatively repair/version the classifier and add regression cases before outputs exist. Completing both historical scorers does not resolve these errors by itself.

The 8B recipe is not a pure parameter-count intervention: backbone, post-training, template and segmentation differ. Tool records produce two sequences, so a record-level micro-batch of one can contain two token sequences; the smoke test must exercise that memory case. No GPU memory guarantee follows from a mask audit.

## Files and next step

Original drafts and pre-change provenance code are preserved under docs/history/w3_approval_review_2026-10-01 with hashes. Approval hashes are stored in configs/w3/approvals.json; per-record Q5 approvals are in configs/w3/q5_relabel_review.jsonl. Current review packets and the experiment journal point to this review.

The four requested artifact approvals are complete. Before a Pod run, commit/review the intended local changes (the runner rejects dirty tracked files), transfer the approved artifacts and use the existing environment/preflight/GPU-smoke sequence. Do not rerun the draft builders over approved artifacts or change them without a versioned review. No Pod, commit, push or paid execution was initiated in this review.
