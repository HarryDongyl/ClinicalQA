# Validation-only scorer v2 report

No test records or predictions were opened. Legacy scores remain unchanged.

Pass/fail/review refer to bounded task contracts. Review is not counted as wrong; full contract accuracy remains unavailable when unresolved examples exist. Intervals describe unresolved coverage conditional on automatic labels, not statistical confidence intervals or validated clinical accuracy. Tool-call rows measure input-grounded policy success (including appropriate no-call behavior), not the original tool E2E metric; unchanged behavior counters are stored separately in summary.json.

| Run | Task | Legacy correct | Pass | Fail | Review | Possible pass fraction |
|---|---|---:|---:|---:|---:|---|
| w3_c_filtered_s42 | extractive | 99/100 | 63 | 0 | 37 | 63.0%–100.0% |
| w3_c_filtered_s42 | numeric_reasoning | 25/50 | 18 | 4 | 28 | 36.0%–92.0% |
| w3_c_filtered_s42 | tool_call | 54/62 | 57 | 1 | 4 | 91.9%–98.4% |
| w3_c_filtered_s42 | uncertain | 35/38 | 35 | 1 | 2 | 92.1%–97.4% |
| w3_q35_4b_filtered_lr1e4_step000121 | extractive | 99/100 | 63 | 0 | 37 | 63.0%–100.0% |
| w3_q35_4b_filtered_lr1e4_step000121 | numeric_reasoning | 35/50 | 22 | 1 | 27 | 44.0%–98.0% |
| w3_q35_4b_filtered_lr1e4_step000121 | tool_call | 53/62 | 60 | 2 | 0 | 96.8%–96.8% |
| w3_q35_4b_filtered_lr1e4_step000121 | uncertain | 36/38 | 36 | 0 | 2 | 94.7%–100.0% |
| w3_q35_4b_filtered_lr1e4_step000242 | extractive | 100/100 | 63 | 0 | 37 | 63.0%–100.0% |
| w3_q35_4b_filtered_lr1e4_step000242 | numeric_reasoning | 34/50 | 23 | 3 | 24 | 46.0%–94.0% |
| w3_q35_4b_filtered_lr1e4_step000242 | tool_call | 54/62 | 55 | 1 | 6 | 88.7%–98.4% |
| w3_q35_4b_filtered_lr1e4_step000242 | uncertain | 36/38 | 36 | 1 | 1 | 94.7%–97.4% |
| w3_r0_8b | extractive | 82/100 | 59 | 0 | 41 | 59.0%–100.0% |
| w3_r0_8b | numeric_reasoning | 18/50 | 13 | 9 | 28 | 26.0%–82.0% |
| w3_r0_8b | tool_call | 46/62 | 46 | 16 | 0 | 74.2%–74.2% |
| w3_r0_8b | uncertain | 17/38 | 17 | 19 | 2 | 44.7%–50.0% |
| w3_r0_q35_4b | extractive | 81/100 | 55 | 0 | 45 | 55.0%–100.0% |
| w3_r0_q35_4b | numeric_reasoning | 17/50 | 14 | 11 | 25 | 28.0%–78.0% |
| w3_r0_q35_4b | tool_call | 51/62 | 55 | 6 | 1 | 88.7%–90.3% |
| w3_r0_q35_4b | uncertain | 27/38 | 19 | 16 | 3 | 50.0%–57.9% |
| w3_r0_q35_9b | extractive | 86/100 | 54 | 0 | 46 | 54.0%–100.0% |
| w3_r0_q35_9b | numeric_reasoning | 23/50 | 16 | 9 | 25 | 32.0%–82.0% |
| w3_r0_q35_9b | tool_call | 49/62 | 52 | 9 | 1 | 83.9%–85.5% |
| w3_r0_q35_9b | uncertain | 18/38 | 16 | 17 | 5 | 42.1%–55.3% |
| w3_r0_v1 | extractive | 39/100 | 53 | 5 | 42 | 53.0%–95.0% |
| w3_r0_v1 | numeric_reasoning | 20/50 | 19 | 0 | 31 | 38.0%–100.0% |
| w3_r0_v1 | tool_call | 29/62 | 36 | 26 | 0 | 58.1%–58.1% |
| w3_r0_v1 | uncertain | 25/38 | 25 | 10 | 3 | 65.8%–73.7% |
| w3_r0_v3 | extractive | 40/100 | 56 | 2 | 42 | 56.0%–98.0% |
| w3_r0_v3 | numeric_reasoning | 19/50 | 19 | 6 | 25 | 38.0%–88.0% |
| w3_r0_v3 | tool_call | 9/62 | 15 | 46 | 1 | 24.2%–25.8% |
| w3_r0_v3 | uncertain | 27/38 | 27 | 8 | 3 | 71.1%–78.9% |
| w3_r0_v3_fs4 | extractive | 85/100 | 59 | 3 | 38 | 59.0%–97.0% |
| w3_r0_v3_fs4 | numeric_reasoning | 22/50 | 18 | 6 | 26 | 36.0%–88.0% |
| w3_r0_v3_fs4 | tool_call | 29/62 | 34 | 26 | 2 | 54.8%–58.1% |
| w3_r0_v3_fs4 | uncertain | 28/38 | 28 | 6 | 4 | 73.7%–84.2% |
| w3_relabel_lr1e4_s42_step000125 | extractive | 100/100 | 63 | 0 | 37 | 63.0%–100.0% |
| w3_relabel_lr1e4_s42_step000125 | numeric_reasoning | 26/50 | 18 | 4 | 28 | 36.0%–92.0% |
| w3_relabel_lr1e4_s42_step000125 | tool_call | 53/62 | 60 | 2 | 0 | 96.8%–96.8% |
| w3_relabel_lr1e4_s42_step000125 | uncertain | 36/38 | 36 | 0 | 2 | 94.7%–100.0% |
| w3_relabel_lr1e4_s42_step000250 | extractive | 98/100 | 63 | 0 | 37 | 63.0%–100.0% |
| w3_relabel_lr1e4_s42_step000250 | numeric_reasoning | 29/50 | 21 | 2 | 27 | 42.0%–96.0% |
| w3_relabel_lr1e4_s42_step000250 | tool_call | 54/62 | 61 | 1 | 0 | 98.4%–98.4% |
| w3_relabel_lr1e4_s42_step000250 | uncertain | 37/38 | 37 | 0 | 1 | 97.4%–100.0% |

Complete the blinded adjudication packet before selecting a model by full semantic accuracy. Keep the identity key separate from reviewers. The packet includes all statuses, so false positives can be audited alongside false negatives.
