# Validation-only scorer v2 report

No test records or predictions were opened. Legacy scores remain unchanged.

Pass/fail/review refer to bounded task contracts. Review is not counted as wrong; full contract accuracy remains unavailable when unresolved examples exist. Intervals describe unresolved coverage conditional on automatic labels, not statistical confidence intervals or validated clinical accuracy. Tool-call rows measure input-grounded policy success (including appropriate no-call behavior), not the original tool E2E metric; unchanged behavior counters are stored separately in summary.json.

| Run | Task | Legacy correct | Pass | Fail | Review | Possible pass fraction |
|---|---|---:|---:|---:|---:|---|
| base | extractive | 39/100 | 53 | 5 | 42 | 53.0%–95.0% |
| base | numeric_reasoning | 20/50 | 19 | 0 | 31 | 38.0%–100.0% |
| base | tool_call | 29/62 | 36 | 26 | 0 | 58.1%–58.1% |
| base | uncertain | 25/38 | 25 | 10 | 3 | 65.8%–73.7% |
| raw_lr1e4_step000125 | extractive | 99/100 | 62 | 1 | 37 | 62.0%–99.0% |
| raw_lr1e4_step000125 | numeric_reasoning | 30/50 | 20 | 2 | 28 | 40.0%–96.0% |
| raw_lr1e4_step000125 | tool_call | 55/62 | 56 | 6 | 0 | 90.3%–90.3% |
| raw_lr1e4_step000125 | uncertain | 35/38 | 35 | 2 | 1 | 92.1%–94.7% |
| raw_lr1e4_step000250 | extractive | 99/100 | 63 | 0 | 37 | 63.0%–100.0% |
| raw_lr1e4_step000250 | numeric_reasoning | 27/50 | 21 | 4 | 25 | 42.0%–92.0% |
| raw_lr1e4_step000250 | tool_call | 54/62 | 55 | 7 | 0 | 88.7%–88.7% |
| raw_lr1e4_step000250 | uncertain | 36/38 | 36 | 1 | 1 | 94.7%–97.4% |
| raw_lr5e5_step000125 | extractive | 98/100 | 63 | 0 | 37 | 63.0%–100.0% |
| raw_lr5e5_step000125 | numeric_reasoning | 26/50 | 19 | 3 | 28 | 38.0%–94.0% |
| raw_lr5e5_step000125 | tool_call | 54/62 | 53 | 8 | 1 | 85.5%–87.1% |
| raw_lr5e5_step000125 | uncertain | 34/38 | 34 | 2 | 2 | 89.5%–94.7% |
| raw_lr5e5_step000250 | extractive | 98/100 | 63 | 0 | 37 | 63.0%–100.0% |
| raw_lr5e5_step000250 | numeric_reasoning | 26/50 | 21 | 4 | 25 | 42.0%–92.0% |
| raw_lr5e5_step000250 | tool_call | 55/62 | 55 | 6 | 1 | 88.7%–90.3% |
| raw_lr5e5_step000250 | uncertain | 33/38 | 33 | 3 | 2 | 86.8%–92.1% |
| q5filtered_lr5e5_step000121 | extractive | 99/100 | 63 | 0 | 37 | 63.0%–100.0% |
| q5filtered_lr5e5_step000121 | numeric_reasoning | 23/50 | 16 | 7 | 27 | 32.0%–86.0% |
| q5filtered_lr5e5_step000121 | tool_call | 55/62 | 61 | 0 | 1 | 98.4%–100.0% |
| q5filtered_lr5e5_step000121 | uncertain | 35/38 | 35 | 1 | 2 | 92.1%–97.4% |
| q5filtered_lr5e5_step000242 | extractive | 98/100 | 63 | 0 | 37 | 63.0%–100.0% |
| q5filtered_lr5e5_step000242 | numeric_reasoning | 24/50 | 19 | 4 | 27 | 38.0%–92.0% |
| q5filtered_lr5e5_step000242 | tool_call | 54/62 | 59 | 1 | 2 | 95.2%–98.4% |
| q5filtered_lr5e5_step000242 | uncertain | 36/38 | 36 | 0 | 2 | 94.7%–100.0% |

Complete the blinded adjudication packet before selecting a model by full semantic accuracy. Keep the identity key separate from reviewers. The packet includes all statuses, so false positives can be audited alongside false negatives.
