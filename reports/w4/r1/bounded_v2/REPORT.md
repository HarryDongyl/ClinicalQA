# Validation-only scorer v2 report

No test records or predictions were opened. Legacy scores remain unchanged.

Pass/fail/review refer to bounded task contracts. Review is not counted as wrong; full contract accuracy remains unavailable when unresolved examples exist. Intervals describe unresolved coverage conditional on automatic labels, not statistical confidence intervals or validated clinical accuracy. Tool-call rows measure input-grounded policy success (including appropriate no-call behavior), not the original tool E2E metric; unchanged behavior counters are stored separately in summary.json.

| Run | Task | Legacy correct | Pass | Fail | Review | Possible pass fraction |
|---|---|---:|---:|---:|---:|---|
| w4_q35_4b_relabel_lr1e4_step000125 | extractive | 99/100 | 63 | 0 | 37 | 63.0%–100.0% |
| w4_q35_4b_relabel_lr1e4_step000125 | numeric_reasoning | 36/50 | 22 | 2 | 26 | 44.0%–96.0% |
| w4_q35_4b_relabel_lr1e4_step000125 | tool_call | 55/62 | 62 | 0 | 0 | 100.0%–100.0% |
| w4_q35_4b_relabel_lr1e4_step000125 | uncertain | 37/38 | 37 | 0 | 1 | 97.4%–100.0% |
| w4_q35_4b_relabel_lr1e4_step000250 | extractive | 100/100 | 63 | 0 | 37 | 63.0%–100.0% |
| w4_q35_4b_relabel_lr1e4_step000250 | numeric_reasoning | 36/50 | 23 | 0 | 27 | 46.0%–100.0% |
| w4_q35_4b_relabel_lr1e4_step000250 | tool_call | 54/62 | 61 | 1 | 0 | 98.4%–98.4% |
| w4_q35_4b_relabel_lr1e4_step000250 | uncertain | 37/38 | 37 | 0 | 1 | 97.4%–100.0% |
| w4_q35_4b_filtered_lr1e4_step000242 | extractive | 100/100 | 63 | 0 | 37 | 63.0%–100.0% |
| w4_q35_4b_filtered_lr1e4_step000242 | numeric_reasoning | 34/50 | 23 | 3 | 24 | 46.0%–94.0% |
| w4_q35_4b_filtered_lr1e4_step000242 | tool_call | 54/62 | 55 | 1 | 6 | 88.7%–98.4% |
| w4_q35_4b_filtered_lr1e4_step000242 | uncertain | 36/38 | 36 | 1 | 1 | 94.7%–97.4% |
| w3_c_filtered_s42 | extractive | 99/100 | 63 | 0 | 37 | 63.0%–100.0% |
| w3_c_filtered_s42 | numeric_reasoning | 25/50 | 18 | 4 | 28 | 36.0%–92.0% |
| w3_c_filtered_s42 | tool_call | 54/62 | 57 | 1 | 4 | 91.9%–98.4% |
| w3_c_filtered_s42 | uncertain | 35/38 | 35 | 1 | 2 | 92.1%–97.4% |
| w3_relabel_lr1e4_s42_step000250 | extractive | 98/100 | 63 | 0 | 37 | 63.0%–100.0% |
| w3_relabel_lr1e4_s42_step000250 | numeric_reasoning | 29/50 | 21 | 2 | 27 | 42.0%–96.0% |
| w3_relabel_lr1e4_s42_step000250 | tool_call | 54/62 | 61 | 1 | 0 | 98.4%–98.4% |
| w3_relabel_lr1e4_s42_step000250 | uncertain | 37/38 | 37 | 0 | 1 | 97.4%–100.0% |
| w3_q35_4b_filtered_lr1e4_step000242 | extractive | 100/100 | 63 | 0 | 37 | 63.0%–100.0% |
| w3_q35_4b_filtered_lr1e4_step000242 | numeric_reasoning | 34/50 | 23 | 3 | 24 | 46.0%–94.0% |
| w3_q35_4b_filtered_lr1e4_step000242 | tool_call | 54/62 | 55 | 1 | 6 | 88.7%–98.4% |
| w3_q35_4b_filtered_lr1e4_step000242 | uncertain | 36/38 | 36 | 1 | 1 | 94.7%–97.4% |
| w3_r0_q35_4b | extractive | 81/100 | 55 | 0 | 45 | 55.0%–100.0% |
| w3_r0_q35_4b | numeric_reasoning | 17/50 | 14 | 11 | 25 | 28.0%–78.0% |
| w3_r0_q35_4b | tool_call | 51/62 | 55 | 6 | 1 | 88.7%–90.3% |
| w3_r0_q35_4b | uncertain | 27/38 | 19 | 16 | 3 | 50.0%–57.9% |

Complete the blinded adjudication packet before selecting a model by full semantic accuracy. Keep the identity key separate from reviewers. The packet includes all statuses, so false positives can be audited alongside false negatives.
