# Paired comparison — w2p_v1_q5filtered_lr5e5 vs w2p_v2_q5filtered_lr5e5 / val / full

Paired by id; bootstrap 2000 resamples (seed 42).

| answer type | n | w2p_v1_q5filtered_lr5e5 | w2p_v2_q5filtered_lr5e5 | diff (b-a) | 95% CI | fixed | broken | both wrong |
|---|---|---|---|---|---|---|---|---|
| extractive | 100 | 99 | 99 | +0.000 | [+0.000, +0.000] | 0 | 0 | 1 |
| numeric_reasoning | 50 | 23 | 24 | +0.020 | [+0.000, +0.060] | 1 | 0 | 26 |
| tool_call | 62 | 55 | 55 | +0.000 | [+0.000, +0.000] | 0 | 0 | 7 |
| uncertain | 38 | 35 | 35 | +0.000 | [+0.000, +0.000] | 0 | 0 | 3 |

Broken examples (correct in a, wrong in b):

