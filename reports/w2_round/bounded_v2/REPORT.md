# Validation-only scorer v2 report

No test records or predictions were opened. Legacy scores remain unchanged.

Pass/fail/review refer to bounded task contracts. Review is not counted as wrong; full contract accuracy remains unavailable when unresolved examples exist. Intervals describe unresolved coverage conditional on automatic labels, not statistical confidence intervals or validated clinical accuracy. Tool-call rows measure input-grounded policy success (including appropriate no-call behavior), not the original tool E2E metric; unchanged behavior counters are stored separately in summary.json.

| Run | Task | Legacy correct | Pass | Fail | Review | Possible pass fraction |
|---|---|---:|---:|---:|---:|---|
| w2_filtered_lr1e4_mb1_step000121 | extractive | 99/100 | 63 | 0 | 37 | 63.0%–100.0% |
| w2_filtered_lr1e4_mb1_step000121 | numeric_reasoning | 27/50 | 20 | 2 | 28 | 40.0%–96.0% |
| w2_filtered_lr1e4_mb1_step000121 | tool_call | 54/62 | 58 | 1 | 3 | 93.5%–98.4% |
| w2_filtered_lr1e4_mb1_step000121 | uncertain | 36/38 | 36 | 0 | 2 | 94.7%–100.0% |
| w2_filtered_lr1e4_mb1_step000242 | extractive | 100/100 | 63 | 0 | 37 | 63.0%–100.0% |
| w2_filtered_lr1e4_mb1_step000242 | numeric_reasoning | 25/50 | 19 | 4 | 27 | 38.0%–92.0% |
| w2_filtered_lr1e4_mb1_step000242 | tool_call | 54/62 | 57 | 1 | 4 | 91.9%–98.4% |
| w2_filtered_lr1e4_mb1_step000242 | uncertain | 35/38 | 35 | 1 | 2 | 92.1%–97.4% |
| w2_filtered_lr1p5e4_mb1_step000121 | extractive | 99/100 | 63 | 0 | 37 | 63.0%–100.0% |
| w2_filtered_lr1p5e4_mb1_step000121 | numeric_reasoning | 30/50 | 18 | 4 | 28 | 36.0%–92.0% |
| w2_filtered_lr1p5e4_mb1_step000121 | tool_call | 54/62 | 58 | 1 | 3 | 93.5%–98.4% |
| w2_filtered_lr1p5e4_mb1_step000121 | uncertain | 36/38 | 36 | 1 | 1 | 94.7%–97.4% |
| w2_filtered_lr1p5e4_mb1_step000242 | extractive | 98/100 | 63 | 0 | 37 | 63.0%–100.0% |
| w2_filtered_lr1p5e4_mb1_step000242 | numeric_reasoning | 29/50 | 19 | 3 | 28 | 38.0%–94.0% |
| w2_filtered_lr1p5e4_mb1_step000242 | tool_call | 54/62 | 57 | 1 | 4 | 91.9%–98.4% |
| w2_filtered_lr1p5e4_mb1_step000242 | uncertain | 37/38 | 37 | 0 | 1 | 97.4%–100.0% |
| w2_filtered_lr2e4_mb1_step000121 | extractive | 99/100 | 63 | 0 | 37 | 63.0%–100.0% |
| w2_filtered_lr2e4_mb1_step000121 | numeric_reasoning | 29/50 | 19 | 3 | 28 | 38.0%–94.0% |
| w2_filtered_lr2e4_mb1_step000121 | tool_call | 55/62 | 57 | 1 | 4 | 91.9%–98.4% |
| w2_filtered_lr2e4_mb1_step000121 | uncertain | 35/38 | 35 | 1 | 2 | 92.1%–97.4% |
| w2_filtered_lr2e4_mb1_step000242 | extractive | 99/100 | 63 | 0 | 37 | 63.0%–100.0% |
| w2_filtered_lr2e4_mb1_step000242 | numeric_reasoning | 27/50 | 20 | 2 | 28 | 40.0%–96.0% |
| w2_filtered_lr2e4_mb1_step000242 | tool_call | 54/62 | 59 | 1 | 2 | 95.2%–98.4% |
| w2_filtered_lr2e4_mb1_step000242 | uncertain | 37/38 | 37 | 0 | 1 | 97.4%–100.0% |
| w2p_v1_base | extractive | 38/100 | 53 | 4 | 43 | 53.0%–96.0% |
| w2p_v1_base | numeric_reasoning | 19/50 | 19 | 2 | 29 | 38.0%–96.0% |
| w2p_v1_base | tool_call | 31/62 | 38 | 24 | 0 | 61.3%–61.3% |
| w2p_v1_base | uncertain | 26/38 | 25 | 7 | 6 | 65.8%–81.6% |
| w2p_v1_q5filtered_lr5e5 | extractive | 99/100 | 63 | 0 | 37 | 63.0%–100.0% |
| w2p_v1_q5filtered_lr5e5 | numeric_reasoning | 23/50 | 16 | 6 | 28 | 32.0%–88.0% |
| w2p_v1_q5filtered_lr5e5 | tool_call | 55/62 | 61 | 0 | 1 | 98.4%–100.0% |
| w2p_v1_q5filtered_lr5e5 | uncertain | 35/38 | 35 | 1 | 2 | 92.1%–97.4% |
| w2p_v1_raw_lr1e4 | extractive | 99/100 | 62 | 1 | 37 | 62.0%–99.0% |
| w2p_v1_raw_lr1e4 | numeric_reasoning | 29/50 | 20 | 2 | 28 | 40.0%–96.0% |
| w2p_v1_raw_lr1e4 | tool_call | 55/62 | 56 | 6 | 0 | 90.3%–90.3% |
| w2p_v1_raw_lr1e4 | uncertain | 35/38 | 35 | 2 | 1 | 92.1%–94.7% |
| w2p_v1_raw_lr5e5 | extractive | 98/100 | 63 | 0 | 37 | 63.0%–100.0% |
| w2p_v1_raw_lr5e5 | numeric_reasoning | 26/50 | 19 | 2 | 29 | 38.0%–96.0% |
| w2p_v1_raw_lr5e5 | tool_call | 54/62 | 53 | 8 | 1 | 85.5%–87.1% |
| w2p_v1_raw_lr5e5 | uncertain | 34/38 | 34 | 2 | 2 | 89.5%–94.7% |
| w2p_v2_base | extractive | 35/100 | 52 | 7 | 41 | 52.0%–93.0% |
| w2p_v2_base | numeric_reasoning | 19/50 | 19 | 6 | 25 | 38.0%–88.0% |
| w2p_v2_base | tool_call | 34/62 | 41 | 21 | 0 | 66.1%–66.1% |
| w2p_v2_base | uncertain | 24/38 | 24 | 11 | 3 | 63.2%–71.1% |
| w2p_v2_q5filtered_lr5e5 | extractive | 99/100 | 63 | 0 | 37 | 63.0%–100.0% |
| w2p_v2_q5filtered_lr5e5 | numeric_reasoning | 24/50 | 17 | 6 | 27 | 34.0%–88.0% |
| w2p_v2_q5filtered_lr5e5 | tool_call | 55/62 | 61 | 0 | 1 | 98.4%–100.0% |
| w2p_v2_q5filtered_lr5e5 | uncertain | 35/38 | 35 | 1 | 2 | 92.1%–97.4% |
| w2p_v2_raw_lr1e4 | extractive | 99/100 | 62 | 1 | 37 | 62.0%–99.0% |
| w2p_v2_raw_lr1e4 | numeric_reasoning | 30/50 | 20 | 2 | 28 | 40.0%–96.0% |
| w2p_v2_raw_lr1e4 | tool_call | 54/62 | 55 | 7 | 0 | 88.7%–88.7% |
| w2p_v2_raw_lr1e4 | uncertain | 34/38 | 34 | 2 | 2 | 89.5%–94.7% |
| w2p_v2_raw_lr5e5 | extractive | 99/100 | 63 | 0 | 37 | 63.0%–100.0% |
| w2p_v2_raw_lr5e5 | numeric_reasoning | 26/50 | 19 | 3 | 28 | 38.0%–94.0% |
| w2p_v2_raw_lr5e5 | tool_call | 54/62 | 54 | 7 | 1 | 87.1%–88.7% |
| w2p_v2_raw_lr5e5 | uncertain | 34/38 | 34 | 2 | 2 | 89.5%–94.7% |

Complete the blinded adjudication packet before selecting a model by full semantic accuracy. Keep the identity key separate from reviewers. The packet includes all statuses, so false positives can be audited alongside false negatives.
