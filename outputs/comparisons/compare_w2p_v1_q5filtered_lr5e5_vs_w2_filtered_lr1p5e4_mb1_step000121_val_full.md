# Paired comparison — w2p_v1_q5filtered_lr5e5 vs w2_filtered_lr1p5e4_mb1_step000121 / val / full

Paired by id; bootstrap 2000 resamples (seed 42).

| answer type | n | w2p_v1_q5filtered_lr5e5 | w2_filtered_lr1p5e4_mb1_step000121 | diff (b-a) | 95% CI | fixed | broken | both wrong |
|---|---|---|---|---|---|---|---|---|
| extractive | 100 | 99 | 99 | +0.000 | [+0.000, +0.000] | 0 | 0 | 1 |
| numeric_reasoning | 50 | 23 | 30 | +0.140 | [+0.060, +0.240] | 7 | 0 | 20 |
| tool_call | 62 | 55 | 54 | -0.016 | [-0.048, +0.000] | 0 | 1 | 7 |
| uncertain | 38 | 35 | 36 | +0.026 | [-0.053, +0.105] | 2 | 1 | 1 |

Broken examples (correct in a, wrong in b):

- tool_call `val_052`: correct -> wrong_args_value
- uncertain `val_200`: correct -> fabricated_value
