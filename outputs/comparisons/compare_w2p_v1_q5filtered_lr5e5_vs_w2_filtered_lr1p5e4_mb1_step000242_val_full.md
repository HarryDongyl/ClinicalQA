# Paired comparison — w2p_v1_q5filtered_lr5e5 vs w2_filtered_lr1p5e4_mb1_step000242 / val / full

Paired by id; bootstrap 2000 resamples (seed 42).

| answer type | n | w2p_v1_q5filtered_lr5e5 | w2_filtered_lr1p5e4_mb1_step000242 | diff (b-a) | 95% CI | fixed | broken | both wrong |
|---|---|---|---|---|---|---|---|---|
| extractive | 100 | 99 | 98 | -0.010 | [-0.030, +0.000] | 0 | 1 | 1 |
| numeric_reasoning | 50 | 23 | 29 | +0.120 | [+0.020, +0.220] | 7 | 1 | 20 |
| tool_call | 62 | 55 | 54 | -0.016 | [-0.048, +0.000] | 0 | 1 | 7 |
| uncertain | 38 | 35 | 37 | +0.053 | [+0.000, +0.132] | 2 | 0 | 1 |

Broken examples (correct in a, wrong in b):

- extractive `val_191`: correct -> number_mismatch
- numeric_reasoning `val_085`: correct -> number_mismatch
- tool_call `val_105`: correct -> wrong_args_imperial
