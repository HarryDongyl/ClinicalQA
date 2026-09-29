# Assignment Specification (verbatim)

Moved verbatim from the original `requirements.txt` so that `requirements.txt` can hold pinned dependencies. Content is unchanged.

```text
Technical Assignment
Clinical QA Fine-Tuning — Technical Assignment
Time: Aim for ~4 hours total.
Overview
This assignment focuses on supervised fine-tuning (SFT) a language model to answer clinical questions grounded in a single encounter note plus an embedded structured table. Your goal is to build an end-to-end fine-tuning setup (data formatting → training → evaluation) such that the model reliably:
Extracts facts directly from the note/table;
Performs simple numeric reasoning over values in the note/table;
Learns tool-use behavior: deciding when to call a deterministic tool and producing valid tool-call arguments;
Produces calibrated responses by explicitly stating uncertainty when the note lacks sufficient information (no hallucinations).
The project is structured as Core (expected) and Stretch (optional extensions). Complete the Core track first. If you have extra time, choose one Stretch option.
Dataset
We provide a pre-generated dataset in JSONL format:
File
Examples
Purpose
train.jsonl
2,000
Training
val.jsonl
250
Validation / hyperparameter tuning
test.jsonl
400
Held-out evaluation (do not train on this)
train.jsonl
val.jsonl
test.jsonl
Use these splits as-is. Do not add external data for the Core track.
Schema
Each line is a JSON object with these fields:
Field
Type
Description
id
string
Unique example identifier, e.g. "train_042"
note
string
Clinical encounter note (150–300 words). Sections typically include CC, HPI, PMH, Medications, Exam, Assessment/Plan. May contain vitals, weight, and height inline.
table
object
One embedded structured table per example (see below).
question
string
A single clinical question about the note.
answer
string
Gold-standard answer (1–3 sentences).
answer_type
string
One of: "extractive", "numeric_reasoning", "tool_call", "uncertain".
tool_calls
array (optional)
Present only when answer_type == "tool_call". Array of tool invocations (see Tool Calling section).
Table format
table is always an object with:
{
 "type": "labs" | "vitals",
 "headers": ["Test", "Value", "Unit", "Reference Range"],   // labs
 //      or: ["Vital", "Value", "Unit"],                     // vitals
 "rows": [["Glucose (fasting)", "150.4", "mg/dL", "70-100"], ...]
}
Lab panels include: metabolic, lipid, CBC, liver, thyroid, coagulation, iron panel, and renal. Vitals include: blood pressure, heart rate, temperature, respiratory rate, and SpO2.
Answer types
Type
Distribution
What the model should do
extractive
~40%
Answer is directly stated in the note or table. No calculation needed.
numeric_reasoning
~20%
Requires comparing, subtracting, or checking thresholds across values in the note/table. No tool call needed.
tool_call
~25%
Requires calling one of the provided tools. The tool_calls field contains the gold-standard invocation and expected result.
uncertain
~15%
The note deliberately omits information needed to fully answer (e.g., missing weight, missing medication dose, no lab timestamp, no allergy documentation). The model must state what is available and what is missing — not hallucinate.
Tool Calling
Available tools
Implement exactly 2 tools with the following deterministic signatures:
1. unit_convert
unit_convert(value: float, from_unit: str, to_unit: str, substance: str | null) -> float
Example conversions are below:
From → To
Substance
Formula
mg/dL → mmol/L
"glucose"
value × 0.0555
mg/dL → μmol/L
"creatinine"
value × 88.42
mg/dL → mmol/L
"cholesterol"
value × 0.0259
lb → kg
null
value × 0.4536
kg → lb
null
value × 2.2046
in → cm
null
value × 2.54
cm → in
null
value × 0.3937
°F → °C
null
(value − 32) × 5/9
°C → °F
null
value × 9/5 + 32
Returns an error string for unsupported conversions.
2. calculate_bmi
calculate_bmi(weight_kg: float, height_cm: float) -> float
Formula: weight_kg / (height_cm / 100)², rounded to 1 decimal.
Tool call format in the dataset
When answer_type == "tool_call", the example includes:
"tool_calls": [
 {
   "tool": "calculate_bmi",
   "arguments": {"weight_kg": 104.2, "height_cm": 163.4},
   "result": 39.0
 }
]
What we evaluate:
Did the model decide to call a tool (vs. answering directly)?
Did it pick the correct tool?
Are the arguments valid and match the values in the note?
Did the final answer incorporate the tool result with brief clinical context?
Important note on units
Some notes record weight in lb and height in in (imperial). The tool_calls field always contains arguments in metric (kg, cm). The model must handle the implicit conversion from imperial → metric when constructing tool arguments. This is intentional — it tests whether the model correctly reads the note and prepares the right inputs.
Submission Guidelines
What to submit: A zipped repository sent back to HR or a GitHub repo link.
Include a clear report documenting your fine-tuning approach and engineering trade-offs, ending with key findings. Please include:
Setup instructions (env + dependencies)
How to run end-to-end: data formatting, fine-tuning, evaluation (exact commands)
Hardware assumptions (GPU/CPU, memory) and expected runtime
Where artifacts are written (e.g., outputs/, checkpoints/, reports/)
What you tried (base model choice, prompt/chat formatting, LoRA/QLoRA config, ablations)
What worked / didn’t work, and why (brief)
Final results + limitations + next steps
Reproducibility:
Pin package versions (e.g., requirements.txt / poetry.lock)
Set and document random seeds
Keep training and evaluation deterministic where feasible
Project structure (suggested):
data/ (provided JSONL; do not modify originals)
src/ (formatting + training + eval code)
configs/ (training/eval configs)
reports/ (data analysis + findings)
Do not include the held-out test.jsonl in fine-tuning.
Make it runnable: we should be able to execute your fine-tuning pipeline end-to-end with minimal manual steps.
Core Track (Expected)
Core deliverables
Submit a PR or repository containing:
1. Data formatting pipeline
A script that converts the provided JSONL files into SFT-ready examples in a chat/instruct format (e.g., OpenAI chat format, Alpaca, or ChatML). Must handle:
Serializing the note + table into the user message (choose a clear, consistent text representation for the table)
Formatting tool_call examples so the model learns the intended tool invocation pattern
Preserving the answer_type distribution across splits
2. Fine-tuning script
A runnable SFT training script (LoRA/QLoRA or full fine-tuning) that we can execute. We do not prescribe a specific base model — choose what fits the task and explain your choice. Acceptable frameworks: HuggingFace Transformers + PEFT, Axolotl, LLaMA-Factory, etc.
3. Data quality analysis
Provide a notebook or report showing:
At least 5 formatted examples (one per answer type, plus one additional), showing exactly what the model sees as input and what the expected output is
Basic statistics: example counts per split, answer type distribution, average note length (words), tool-call frequency, table type distribution
Any data quality issues you noticed and how you addressed them
4. Evaluation script
A runnable evaluation script that reports, at minimum, how your fine-tuned model performs on each answer type:
Metric
What it measures
Extractive accuracy
Fuzzy/exact match of extracted facts against gold answer
Numeric reasoning accuracy
Correctness of numeric comparisons or calculations
Tool selection accuracy
% of tool_call examples where the model chose the right tool
Tool argument accuracy
% of tool calls with valid, correct arguments
Uncertainty detection rate
% of uncertain examples where the model correctly refused to hallucinate
Simple implementations are fine (e.g., string overlap, regex parsing of tool calls). Explain any limitations.
Stretch Track (Optional — pick one)
Attempt if you prefer, not compulsory.
Stretch A: Add a third tool
Add one additional tool and expand your pipeline + evaluation:
Tool
Signature
calculate_egfr
calculate_egfr(creatinine_mg_dl: float, age: int, sex: str) -> float
clinical_score
clinical_score(score_name: str, inputs: dict) -> float (limit to one score, e.g., CHA₂DS₂-VASc)
Annotate a small set of new examples (10–20) for the new tool and show evaluation results.
Stretch B: Offline reference lookup
Add a provided reference.jsonl file and a deterministic lookup tool:
reference_lookup(key: str) -> str
The reference file contains drug interaction data or clinical guidelines (we provide it, versioned in the repo). Show how the model learns when to consult the reference vs. answer from the note alone.
reference.jsonl
Stretch C: Multi-turn (minimal)
Support 2-turn interactions only:
Turn 1: Answer from note + question
Turn 2: Answer a follow-up that references Turn 1's answer
Demonstrate with 10–20 curated multi-turn examples. Explain how you would scale this.
Evaluation Criteria
We optimize for clear fine-tuning thinking and good engineering trade-offs (data formatting, training setup, and evaluation design), not perfection.
Mindset: ship an end-to-end baseline first, then iterate with small, measurable changes.
Training approach: keep the setup reproducible (seeds + pinned deps) and justify key choices (format, base model, LoRA/QLoRA/Full, seq length, LR/epochs).
Evaluation-driven: use your evaluation to guide decisions and briefly summarize common failure modes (especially tool calls + uncertainty).
Weight
What we look for
10%
Data formatting quality — correct chat/instruct formatting for SFT, handling of all 4 answer types, reproducibility
40%
Fine-tuning setup — reasonable base model and hyperparameter choices, stability, clear code, can we run it?
10%
Evaluation — meaningful metrics for a fine-tuned + tool-using model, honest analysis of strengths/weaknesses
20%
Code quality — clean, documented, version-controlled
20%
Stretch / going beyond — optional extensions, thoughtful next-steps (e.g., better tool-call parsing, error analysis, data ablations)
A well-executed Core submission with clear documentation is a strong submission. If you hit the time cap, stop and describe what you would do next — that write-up is valuable.
Quick Start
# Inspect the data
head -1 data/train.jsonl | python -m json.tool
# Check split sizes
wc -l data/*.jsonl
# View answer type distribution
python -c "
import json, collections
data = [json.loads(l) for l in open('data/train.jsonl')]
print(collections.Counter(d['answer_type'] for d in data))
"
```
