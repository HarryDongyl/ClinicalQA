# Paired comparison — w2p_v1_raw_lr1e4 vs w2_filtered_lr2e4_mb1_step000121 / val / full

Paired by id; bootstrap 2000 resamples (seed 42).

| answer type | n | w2p_v1_raw_lr1e4 | w2_filtered_lr2e4_mb1_step000121 | diff (b-a) | 95% CI | fixed | broken | both wrong |
|---|---|---|---|---|---|---|---|---|
| extractive | 100 | 99 | 99 | +0.000 | [-0.030, +0.030] | 1 | 1 | 0 |
| numeric_reasoning | 50 | 29 | 29 | +0.000 | [-0.080, +0.080] | 2 | 2 | 19 |
| tool_call | 62 | 55 | 55 | +0.000 | [+0.000, +0.000] | 0 | 0 | 7 |
| uncertain | 38 | 35 | 35 | +0.000 | [-0.079, +0.079] | 1 | 1 | 2 |

Broken examples (correct in a, wrong in b):

- extractive `val_068`: correct -> direction_mismatch
- numeric_reasoning `val_081`: correct -> number_mismatch
- numeric_reasoning `val_114`: correct -> number_mismatch
- uncertain `val_110`: correct -> missing_field_not_named
