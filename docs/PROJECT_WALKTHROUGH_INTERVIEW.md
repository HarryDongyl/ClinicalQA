# Clinical QA: interview walkthrough and technical defense

Prepared: 2026-10-02. Evidence checked against the current assignment, walkthrough, decision log through D-097, Wave 4 round-one family comparison, and recorded gate approval. This guide reorganizes the project for an interview; the chronological walkthrough remains the historical reference.

## How to use this guide

Practise the opening first, then the ten-minute narrative. Use the technical sections when the interviewer asks for depth. Do not recite every experiment. Each transition should answer why the next experiment became necessary.

The central chain is:

**Define useful grounded behavior → inspect the supervision → build a faithful training and execution pipeline → discover that the score mismeasures behavior → repair measurement → distinguish filtering from teaching abstention → test both missing and intact inputs → replicate the intervention → choose the simpler candidate while preserving uncertainty.**

The interview is about whether your decisions follow from evidence, including evidence that contradicts your initial approach.

## 1. The ninety-second opening

> I built an end-to-end fine-tuning pipeline for clinical QA over an encounter note and a structured table. The model needed to extract facts, perform simple calculations, call two deterministic tools, and state what was missing when it could not answer.
>
> I used a 4B instruction model with QLoRA to fit a single 24 GB GPU. The central challenge turned out to be the supervision and evaluation. I found 78 training records whose BMI tool arguments were unsupported by their inputs. Training on them encouraged invented arguments. Removing them helped tool routing, but the model could still invent a BMI in its final prose without making a call.
>
> That led to a targeted intervention: retain those inputs and teach explicit abstention. I evaluated this with paired probes: remove the measurements and require abstention, then keep the original inputs and require a valid call. On 34 probes, fabrication fell from 25 cases to zero, while intact-input tool success stayed at 33. The same pattern replicated on Qwen3.5, from 29 cases to zero.
>
> I also discovered that my original scorer penalized correct paraphrases, while the improved scorer still missed false extra numeric claims. So I separate strong behavioral evidence from uncertain numerical accuracy. My main conclusion is that explicit supervision for missing information mattered more than simply filtering data or changing the backbone. The remaining work is seed replication, a claim-level numeric audit, and recovery of the selected adapter for the next experiments.

If interrupted after the first thirty seconds, the interviewer should already know the task, constraints and central technical problem.

## 2. The ten-minute narrative: seven connected decisions

### Decision 1 — Define success as useful grounding, not just answer similarity

**Question:** What would count as a good system?

The assignment provides 2,000 training examples, 250 validation examples and 400 test examples. It requires four behaviors and two tools, with no external Core training data. I translated those requirements into a behavior contract: answer supported questions; invoke the correct tool with grounded arguments when required; incorporate the executed result; and explicitly identify missing information.

A model that always refuses can avoid invented tool arguments while being useless. A model that always calls can complete many valid tasks while fabricating inputs. I therefore measure both sides of the decision boundary.

**What to say:** “I needed to distinguish correctness of the answer from correctness of the behavior that produced it.”

**Transition:** “Once I had that contract, I checked whether the training targets actually taught it.”

### Decision 2 — Audit the supervision before treating gold as truth

The important defect was Q5: 78 of 2,000 training records supplied BMI tool arguments without the necessary measurements in the note, table or question. That is 3.9% of all training rows, not 3.9% of tool rows; it is 15.6% of the 500 tool rows.

A detector generated review candidates. Grounding checks included the question and unit conversions, because some valid examples put measurements only in the question. The original data remained unchanged. Filtering and relabeling produced versioned derived training views.

The supervision conflict is concrete: a prompt says “do not invent missing information,” while a training target rewards a specific weight and height absent from the context. Cross-entropy trains toward the latter whenever that example appears. This is an explanation consistent with the experiments, not a direct measurement of an internal mechanism.

**What to say:** “I treated quality flags as hypotheses to review, then made the data intervention explicit and reproducible.”

**Transition:** “Before comparing those policies, I needed confidence that the model was trained on the same interaction format that it would see during execution.”

### Decision 3 — Make the training and inference contracts match

I serialized the note, Markdown table and question consistently and used the model's native chat template. Tool schemas appear on all examples, including no-call examples, so schema availability does not leak the target class. `answer_type` stays outside model-visible text.

A tool example has an assistant call, a tool result and an assistant final answer. Assistant-only masks supervise the call and answer, including appropriate end tokens; user content, tool-result text and padding are not prediction targets. Tool results still provide context for the answer.

The runtime actually parses, validates and executes the call, then asks the model for its final answer. This catches errors that teacher-forced validation cannot: wrong tool choice, malformed calls, ungrounded arguments, failure to use the result, and budget violations.

I started with Qwen3-4B-Instruct-2507 and a fixed QLoRA recipe: NF4 with bf16 compute, rank 16, alpha 32, dropout 0.05, LR 1e-4, two epochs and effective batch 16. This was a feasible baseline, not a claim of optimal hyperparameters.

**Transition:** “The pipeline produced a large initial score improvement, but inspecting the failures showed that the score itself was part of the problem.”

### Decision 4 — Validate the measurement before optimizing it

The first scorer depended on gold numbers, phrasing and local direction words. It often failed valid base-model paraphrases. SFT made outputs resemble the target style, so some apparent improvement was evaluator compatibility rather than a new capability.

The replacement computed typed expectations from the input and question and used explicit numerical tolerances. Critically, I tested the scorer against blinded LLM adjudication. Version 2.0 performed well on its development set but reached only 76.7% agreement on the first holdout. I treated that as a failure. After revision, v2.1 reached 92.6% agreement on a second holdout under the recorded rubric, versus 65.7% for v1.

That is agreement with an LLM-adjudicated reference, not physician-validated clinical accuracy. Those holdouts used previously held-out task data, so the original test is no longer an untouched final benchmark.

Later models exposed another weakness: the scorer could find a correct requested number while ignoring a contradictory extra sentence. The measurement had improved but was still incomplete.

**What to say:** “I stopped treating a higher score as sufficient evidence and checked what errors the scorer could and could not detect.”

**Transition:** “With that limitation visible, I focused on a behavioral failure I could inspect directly: missing measurements.”

### Decision 5 — Compare removing bad examples with teaching the correct alternative

Filtering removed the 78 unsupported targets, leaving 1,922 rows. It reduced wrong calls but did not adequately teach what to say when both BMI inputs were missing. The model could decline to call and still invent a numerical answer.

Relabeling retained those 78 inputs and changed their targets to explicit uncertainty, restoring a 2,000-row view. Review covered compound questions so supported parts were retained where appropriate. This is a documented training-label correction and changes the derived answer-type distribution; it does not silently preserve the original 40/20/25/15 mixture.

P1 probes remove measurements from 34 validation BMI inputs; six other candidates were excluded because the question itself states the measurements. Each probe is paired with its original intact input.

For Qwen3, filtered versus relabeled gives:

- fabricated measurements or BMI: **25/34 → 0/34**;
- valid calls on intact partners: **33/34 → 33/34**;
- correct grounded tool tasks: **54/55 → 54/55**.

The 25 filtered errors were in prose, with no actual tool calls. That observation explains why routing metrics alone were insufficient.

**Transition:** “The result was large, but the intervention also changed row count and steps. I needed controls and replication before giving it a broader interpretation.”

### Decision 6 — Strengthen the explanation without claiming perfect causal identification

A raw-data control uses 2,000 rows and 250 steps, like the relabeled model, and fabricates on 32/34 probes. This weakens the explanation that the relabel benefit came merely from more training rows or steps. Differences in runtime history and the single seed still limit a pure causal claim.

The policy contrast replicates on Qwen3.5: filtered data yields 29/34 fabricated P1 responses; relabeled data yields 0/34. Both have 33/34 intact-partner successes and 54/55 grounded tool successes, though not exactly the same failed partner.

This is cross-family replication on the same small probe set. It is neither independent-cohort validation nor a substitute for multiple training seeds.

Prompting did not provide an equivalent solution in the tested arms. Under the Wave 3 strict tool checks, base v1 completed 29/55 grounded tool tasks, v3 completed 9/55 and v3 plus four demonstrations completed 29/55. SFT completed 54/55. These are conclusions about those particular prompts, not all possible prompting methods.

**Transition:** “Once both backbones had the corrected supervision, the selection question became whether the newer model added enough value to justify its complexity.”

### Decision 7 — Make a provisional engineering choice and preserve unresolved questions

On relabeled data, Qwen3.5's diagnostic macro exceeds Qwen3 by 0.25 percentage points, with a paired interval of −2.15 to +3.10. This provides no clear superiority evidence; it does not establish equivalence. Both pass the aggregate grounding outcomes discussed above, while numeric superiority remains unresolved.

The recorded choice is to continue with Qwen3 for the next extension. It has a simpler already-working integration, shorter serialized sequences and fewer additional dependencies. The observed speed and memory figures come from different GPUs and cannot establish a controlled cost advantage. Longer token sequences are a measured difference, not a proven sole cause of slower training.

The Qwen3.5 gate had a strict partner regression on `val_104`: 68.9 inches became 174.9 cm instead of 175.006. BMI and category were correct, but the argument tolerance failed. The recorded reviewer approved continuation with that exception explained. This is an approved review decision, not a clean automatic pass.

Finally, the original F adapter is no longer available. The next stage needs a named refit, F′, with its own hashes and output comparison. Historical F outputs remain evidence about F; they cannot be presented as measurements of F′.

## 3. Technical deep dives

### Training objective and effective batch

The objective is the sum of negative log probabilities of supervised assistant tokens divided by the number of supervised tokens in the accumulated batch:

`L = sum(-log p(target_token | preceding_context)) / number_of_supervised_tokens`

Averaging already-averaged micro-batch losses can overweight shorter batches. The implementation uses the accumulated supervised-token count through the Trainer loss contract. The relevant unit is supervised tokens, not total padded sequence length.

Micro-batch 1 with accumulation 16 and micro-batch 4 with accumulation 4 target the same effective record batch. They need not be numerically identical: padding, dropout, reduction order and kernels can differ. Similar observed loss curves are a sanity check, not proof of correct accumulation. A stronger check compares loss and gradients on the same fixed examples with dropout disabled and controlled precision.

The reported type shares are supervised-token shares. They do not measure gradient norm, optimizer influence or causal contribution. There is no evidence here that reweighting could never help; it was simply not the most justified next intervention.

### Why QLoRA, and what remains unmeasured

The frozen quantized base limits trainable state; low-rank adapters receive updates. This made a 4B model practical within the original GPU budget and left room for runtime overhead. Actual memory includes activations, optimizer state, temporary buffers and allocator behavior, not only model weights.

Do not claim QLoRA was necessary if bf16 LoRA might fit. It was a headroom tradeoff. The accuracy and speed tradeoff against bf16 LoRA was not measured. Rank 16 was a reasonable fixed starting point; the study did not establish an optimal rank.

Micro-batch 4 was slower in the observed run. Padding and attention-kernel selection are plausible explanations; allocator pressure was also visible earlier. Without profiling, none is a confirmed cause. Packing is a candidate experiment, not a guaranteed fix, and must preserve masks and isolation between examples.

### Template parity and model migration

“Same messages” does not imply “same tokens.” Templates can insert or omit historical thinking blocks, tool delimiters or generation prefixes. The Qwen3-8B template required segmented supervision so the historical call turn matched what inference actually rendered. Qwen3.5 uses XML calls and needed architecture-specific LoRA target coverage.

An audit should check exact rendered prefixes, supervised spans, terminators, tool parsing round-trips and truncation. A migration that silently leaves most token-mixing layers unadapted is not a fair fixed-recipe comparison.

Keep these details available for a deep engineering question; do not let them obscure the main experimental result.

### Evaluation denominators

Remember the difference between:

- **62** annotated validation tool records, including seven unsupported Q5 labels;
- **55** grounded tool tasks where a real call is appropriate;
- **34** missing-input P1 probes and their intact partners;
- **50** numeric validation questions;
- **38** originally annotated uncertain validation questions.

The diagnostic tool metric over 62 treats Q5 abstention differently from legacy tool-call matching. Therefore “61/62 tool score” is not “61 successful calls.” Call and text fabrication categories can overlap; do not add them without checking unique cases.

The historical walkthrough has 29/55 for R0-v1 under Wave 3 strict checks, while the later family report lists 30/55 under its reporting path. Do not blend these counts. Name the report and definition; reconcile the per-item discrepancy before using a single unified presentation table. The principal SFT 54/55 result is consistent across those reports.

### Why a correct tool result does not finish the evaluation

Tool quality has several stages: deciding to call, selecting the tool, schema validity, input grounding, successful execution, incorporation of the returned result, and factual context in the final answer. A system may pass one and fail another.

Exact argument checks and outcome checks answer different questions. In `val_104`, the argument check detects conversion drift even though rounding leaves the BMI unchanged. Keep both visible rather than moving the tolerance to admit a preferred model.

A further gap is whether the model uses the returned result or independently recomputes it. A controlled altered-result probe could test this, but blindly copying an inconsistent tool result is not itself desirable. The intended behavior and probe rubric would need to be defined first.

### Scorer validation and numeric reasoning

The move from v1 to v2.1 reduces surface-form bias. It does not make the evaluator a complete semantic verifier. In `val_062`, an answer can state that 99 is within 60–100 and also claim that it exceeds the upper limit. Matching one correct clause is insufficient.

A stronger numeric audit checks the selected quantity, reference range, units, sign, arithmetic, ranking criterion, and every material extra assertion. Ambiguous questions stay ambiguous. The v2.2 guards remain a prototype because they were developed after inspecting these outputs and lack a fresh evaluation set.

A valid scorer-development holdout is also not automatically a valid untouched model-evaluation test. Test-derived scorer development can influence later selection indirectly. That exposure must remain disclosed.

### Statistical interpretation and confidence

Zero fabricated responses out of 34 gives an upper Wilson 95% bound of about 10.15%. It supports an observed improvement; it does not certify a population fabrication rate below 5%.

Paired bootstrap intervals capture item variation conditional on the sampled outputs. They do not capture training-seed variance, correct an invalid scorer, remove adaptive selection bias or establish generalization to real clinical notes.

An interval containing zero does not show equality. Likewise, a statistically positive exploratory difference does not automatically establish a reliable winner after many comparisons. Do not call every small LR difference “noise”; call the ranking insufficiently established given evaluator uncertainty, adaptivity and one seed.

Call-prefix probability is a probability of a token event. It can be used as a proxy for a call decision when the format aligns the two. It exists under sampled or greedy decoding; greedy decoding does not make it intrinsically valid. Call ECE does not measure factual calibration. Mean answer log probability is a ranking diagnostic, not a calibrated probability that the whole answer is correct.

## 4. Difficult interview questions and defensible answers

### “What is the actual contribution beyond running SFT?”

“I built the runnable pipeline, then used it to identify a supervision failure and a measurement failure. The useful experimental result is the distinction between deleting unsupported targets and teaching the missing-input behavior, measured jointly with intact-input utility and replicated across two families.”

Avoid presenting an established method such as QLoRA as novel research.

### “Did SFT improve knowledge, or just formatting?”

“The original aggregate score mixed both. The strongest evidence is improved executed tool behavior and missing-input responses. Extractive capability was already much stronger than v1 suggested. Numerical reasoning remains unresolved under a complete claim-level rubric.”

### “Why relabel rather than filter?”

“Filtering removes an incorrect positive target but may leave little coverage of the desired negative behavior. Relabeling teaches the alternative on exactly those inputs. The experiments support that explanation, while the raw-data control reduces the row-count objection.”

### “Did you violate the fixed-data requirement?”

“The original files and splits are preserved, and no external Core examples were added. I used explicit derived views. Relabeling changes 78 targets and their answer types, so it is a documented experimental deviation from preserving the original distribution, not something I would hide as ordinary formatting.”

### “Could the model simply memorize the abstention sentence?”

“Yes. Reusing a sentence can satisfy these probes without proving broad uncertainty understanding. The intact partners show it did not simply refuse everything, but stronger evidence would require varied missingness, paraphrases and an independent distribution.”

### “Are 34 probes enough?”

“They are useful for exposing a large failure missed by seven natural examples, not for a tight population guarantee. They share sources and a removal procedure. I would add seed replication and independent reviewed cases before claiming general robustness.”

### “Why not train longer or raise the learning rate?”

“The largest verified failure had a supervision explanation. Sweeping optimization against an imperfect scorer risks optimizing the wrong behavior. I would first stabilize the numerical endpoint and replicate the data-policy result, then run a bounded optimization experiment if the residual evidence warrants it.”

### “Why not RL or a reward model?”

“The demonstrated behavior was teachable with corrected SFT targets. An evaluator that misses false extra claims would also be an unreliable reward and could encourage exploitation. RL becomes a justified experiment only after a sufficiently reliable reward and a residual problem that SFT has not addressed.”

### “Why keep the older model?”

“We did not establish a meaningful advantage for the newer one under matched relabel data. I chose the simpler integration provisionally. I am not claiming equivalent capability or a controlled speed advantage, and a completed numeric audit could change the decision.”

### “What did you get wrong?”

“I initially trusted the task labels and scorer too much. The first selection rule excluded the very unsupported cases that exposed the selected model's weakness. I also failed to preserve the final F checkpoint remotely. The correction is both methodological—explicit behavioral gates—and operational—verify uploaded checkpoint bytes before deleting a pod.”

### “Can you claim a clean final test?”

“No. Test data was inspected and used during scorer development. Another frozen run is useful as a disclosed reused-test evaluation, but cannot restore an untouched holdout. A fresh external evaluation would require agreement to expand the original assessment.”

### “How much was your work versus AI assistance?”

“AI assistance contributed to implementation, review and documentation, including label-review support. I distinguish those reviews from clinician adjudication. I would explain the decisions I made, show the checks I can reproduce, and avoid claiming a manual or clinical review that did not happen.”

Adapt this answer to your actual contribution; do not memorize an inaccurate ownership claim.

### “Was this proportionate to a four-hour assignment?”

“The end-to-end Core and this extended investigation are different scopes. I would present a runnable minimal submission first, then identify the additional experiments as work beyond the suggested budget. In a time-boxed setting, I would stop after the baseline, one high-value data intervention and a clear limitations report.”

Do not claim the extended work was completed within four hours. State actual time if asked.

## 5. Current status: completed, conditional and pending

Completed evidence includes the Core pipeline, scorer revisions, Qwen3 policy comparison, Qwen3.5 filtered/relabel comparison, raw-data probe control and Wave 4 parity regeneration. The parity report records 250/250 identical core trajectories for the compared code states; it does not imply every possible future code change is behavior-preserving.

The recorded Qwen3.5 gate is approved by Harry, with the strict `val_104` failure explained. Backbone selection is separately recorded as Qwen3. This supersedes the walkthrough's “gate review pending” text. The later backbone choice was made after observing results and deviates from the original conditional selection plan; disclose that without claiming its bias is known to be low.

The original F checkpoint is unavailable. D-097 specifies a separately named refit, F′, for the subsequent extension and final evaluation. No refit success is asserted here. Check its manifest and reports before the interview if new results arrive.

Stretch A adds `calculate_egfr` to test limited third-tool adaptation. The design and reviewed evaluation examples are prepared; this guide does not claim a completed transfer result. Adding 52 training examples is supervised tool adaptation, not proof of broad unseen-tool generalization. Keep the optional extension distinct from the two-tool Core.

The full numeric claim audit, seed robustness and a clean external evaluation remain unresolved. A gate approval does not resolve them.

## 6. Suggested demo and evidence route

Use three examples rather than scrolling through hundreds of metrics:

1. Show one unsupported training BMI target and its corrected abstention target. Point out precisely which required inputs are absent.
2. Show a filtered-model P1 answer that invents a BMI without a call, followed by the relabeled answer and its intact partner's successful call.
3. Show `val_062` to explain why the numeric scorer's pass is insufficient.

Then show the implementation path: formatting and loss masks → training config and manifest → raw rollout with actual execution → per-item scoring and aggregate report. This makes the scientific claim traceable to a runnable system.

Primary local evidence, relative to the repository root:

- `docs/ASSIGNMENT.md`: original requirements.
- `docs/PROJECT_WALKTHROUGH.md`: full chronology, with the caveats corrected in this guide.
- `docs/DECISIONS.md`: D-094 through D-097 for current choices and missing-checkpoint recovery.
- `reports/w4/r1/family_compare/TABLE.md`: latest cross-family score definitions and counts.
- `reports/w4/r1/p1/`: paired grounding behavior.
- `reports/w4/r1/parity_q35_filter_ep2.json`: code-state parity comparison.
- `configs/w4/gate_q35.json`: actual gate approval and selected extension backbone.
- `reports/scorer_v2/VALIDATION.md`: evaluator validation and its remaining limitations.
- `src/clinqa/formatting.py`, `src/clinqa/train.py`: representation and optimization contracts.

These are repository paths for preparation, not claims that every result was independently recomputed for this writing task.

## 7. Final rehearsal checklist

Be able to explain each of these without reading the document:

- Why no-call is different from no-hallucination.
- Why a missing-input probe must have an intact partner.
- Why filtering and relabeling teach different things.
- Why supervised-token normalization matters under accumulation.
- Why model migration requires a new template and mask audit.
- Why 46/50 scorer passes are not 92% verified clinical accuracy.
- Why 0/34 is not a population safety guarantee.
- Why the model comparison does not establish equivalence or a clean cost ranking.
- Why an approved gate can coexist with a documented strict-check failure.
- Why F′ must not inherit F's results merely because its configuration matches.

A strong closing sentence is:

> “The main lesson was that reliable fine-tuning depends on aligning the supervision, the execution contract and the evaluation. When those disagreed, a better score could hide a worse behavior. The most convincing improvement came from identifying that disagreement and testing a targeted correction.”
