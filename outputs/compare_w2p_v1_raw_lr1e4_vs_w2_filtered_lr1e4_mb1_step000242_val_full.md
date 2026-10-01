# Paired comparison — w2p_v1_raw_lr1e4 vs w2_filtered_lr1e4_mb1_step000242 / val / full

Paired by id; bootstrap 2000 resamples (seed 42).

| answer type | n | w2p_v1_raw_lr1e4 | w2_filtered_lr1e4_mb1_step000242 | diff (b-a) | 95% CI | fixed | broken | both wrong |
|---|---|---|---|---|---|---|---|---|
| extractive | 100 | 99 | 100 | +0.010 | [+0.000, +0.030] | 1 | 0 | 0 |
| numeric_reasoning | 50 | 29 | 25 | -0.080 | [-0.180, +0.020] | 2 | 6 | 19 |
| tool_call | 62 | 55 | 54 | -0.016 | [-0.048, +0.000] | 0 | 1 | 7 |
| uncertain | 38 | 35 | 35 | +0.000 | [-0.105, +0.105] | 2 | 2 | 1 |

Broken examples (correct in a, wrong in b):

- numeric_reasoning `val_054`: correct -> number_mismatch
- numeric_reasoning `val_075`: correct -> number_mismatch
- numeric_reasoning `val_081`: correct -> number_mismatch
- numeric_reasoning `val_114`: correct -> number_mismatch
- numeric_reasoning `val_117`: correct -> number_mismatch
- tool_call `val_105`: correct -> wrong_args_imperial
- uncertain `val_025`: correct -> fabricated_value
- uncertain `val_110`: correct -> missing_field_not_named
