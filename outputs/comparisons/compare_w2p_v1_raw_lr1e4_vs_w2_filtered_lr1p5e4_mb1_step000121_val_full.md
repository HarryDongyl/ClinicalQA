# Paired comparison — w2p_v1_raw_lr1e4 vs w2_filtered_lr1p5e4_mb1_step000121 / val / full

Paired by id; bootstrap 2000 resamples (seed 42).

| answer type | n | w2p_v1_raw_lr1e4 | w2_filtered_lr1p5e4_mb1_step000121 | diff (b-a) | 95% CI | fixed | broken | both wrong |
|---|---|---|---|---|---|---|---|---|
| extractive | 100 | 99 | 99 | +0.000 | [-0.030, +0.030] | 1 | 1 | 0 |
| numeric_reasoning | 50 | 29 | 30 | +0.020 | [-0.100, +0.140] | 5 | 4 | 16 |
| tool_call | 62 | 55 | 54 | -0.016 | [-0.048, +0.000] | 0 | 1 | 7 |
| uncertain | 38 | 35 | 36 | +0.026 | [+0.000, +0.079] | 1 | 0 | 2 |

Broken examples (correct in a, wrong in b):

- extractive `val_068`: correct -> direction_mismatch
- numeric_reasoning `val_075`: correct -> number_mismatch
- numeric_reasoning `val_081`: correct -> number_mismatch
- numeric_reasoning `val_114`: correct -> number_mismatch
- numeric_reasoning `val_239`: correct -> number_mismatch
- tool_call `val_052`: correct -> wrong_args_value
