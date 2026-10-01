# Paired comparison — w2p_v1_q5filtered_lr5e5 vs w2_filtered_lr1e4_mb1_step000121 / val / full

Paired by id; bootstrap 2000 resamples (seed 42).

| answer type | n | w2p_v1_q5filtered_lr5e5 | w2_filtered_lr1e4_mb1_step000121 | diff (b-a) | 95% CI | fixed | broken | both wrong |
|---|---|---|---|---|---|---|---|---|
| extractive | 100 | 99 | 99 | +0.000 | [+0.000, +0.000] | 0 | 0 | 1 |
| numeric_reasoning | 50 | 23 | 27 | +0.080 | [+0.020, +0.160] | 4 | 0 | 23 |
| tool_call | 62 | 55 | 54 | -0.016 | [-0.048, +0.000] | 0 | 1 | 7 |
| uncertain | 38 | 35 | 36 | +0.026 | [+0.000, +0.079] | 1 | 0 | 2 |

Broken examples (correct in a, wrong in b):

- tool_call `val_105`: correct -> wrong_args_imperial
