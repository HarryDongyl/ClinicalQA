# Paired comparison — w2p_v1_raw_lr1e4 vs w2_filtered_lr1e4_mb1_step000121 / val / full

Paired by id; bootstrap 2000 resamples (seed 42).

| answer type | n | w2p_v1_raw_lr1e4 | w2_filtered_lr1e4_mb1_step000121 | diff (b-a) | 95% CI | fixed | broken | both wrong |
|---|---|---|---|---|---|---|---|---|
| extractive | 100 | 99 | 99 | +0.000 | [-0.030, +0.030] | 1 | 1 | 0 |
| numeric_reasoning | 50 | 29 | 27 | -0.040 | [-0.140, +0.060] | 2 | 4 | 19 |
| tool_call | 62 | 55 | 54 | -0.016 | [-0.048, +0.000] | 0 | 1 | 7 |
| uncertain | 38 | 35 | 36 | +0.026 | [-0.053, +0.105] | 2 | 1 | 1 |

Broken examples (correct in a, wrong in b):

- extractive `val_068`: correct -> direction_mismatch
- numeric_reasoning `val_075`: correct -> number_mismatch
- numeric_reasoning `val_081`: correct -> number_mismatch
- numeric_reasoning `val_114`: correct -> number_mismatch
- numeric_reasoning `val_239`: correct -> number_mismatch
- tool_call `val_105`: correct -> wrong_args_imperial
- uncertain `val_110`: correct -> missing_field_not_named
