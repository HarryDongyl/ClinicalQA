# Paired comparison — w2p_v1_raw_lr1e4 vs w2_filtered_lr1p5e4_mb1_step000242 / val / full

Paired by id; bootstrap 2000 resamples (seed 42).

| answer type | n | w2p_v1_raw_lr1e4 | w2_filtered_lr1p5e4_mb1_step000242 | diff (b-a) | 95% CI | fixed | broken | both wrong |
|---|---|---|---|---|---|---|---|---|
| extractive | 100 | 99 | 98 | -0.010 | [-0.040, +0.020] | 1 | 2 | 0 |
| numeric_reasoning | 50 | 29 | 29 | +0.000 | [-0.100, +0.120] | 4 | 4 | 17 |
| tool_call | 62 | 55 | 54 | -0.016 | [-0.048, +0.000] | 0 | 1 | 7 |
| uncertain | 38 | 35 | 37 | +0.053 | [+0.000, +0.132] | 2 | 0 | 1 |

Broken examples (correct in a, wrong in b):

- extractive `val_068`: correct -> direction_mismatch
- extractive `val_191`: correct -> number_mismatch
- numeric_reasoning `val_054`: correct -> number_mismatch
- numeric_reasoning `val_075`: correct -> number_mismatch
- numeric_reasoning `val_081`: correct -> number_mismatch
- numeric_reasoning `val_114`: correct -> number_mismatch
- tool_call `val_105`: correct -> wrong_args_imperial
