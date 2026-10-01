# Paired comparison — w2p_v1_raw_lr1e4 vs w2p_v2_raw_lr1e4 / val / full

Paired by id; bootstrap 2000 resamples (seed 42).

| answer type | n | w2p_v1_raw_lr1e4 | w2p_v2_raw_lr1e4 | diff (b-a) | 95% CI | fixed | broken | both wrong |
|---|---|---|---|---|---|---|---|---|
| extractive | 100 | 99 | 99 | +0.000 | [+0.000, +0.000] | 0 | 0 | 1 |
| numeric_reasoning | 50 | 29 | 30 | +0.020 | [+0.000, +0.060] | 1 | 0 | 20 |
| tool_call | 62 | 55 | 54 | -0.016 | [-0.048, +0.000] | 0 | 1 | 7 |
| uncertain | 38 | 35 | 34 | -0.026 | [-0.079, +0.000] | 0 | 1 | 3 |

Broken examples (correct in a, wrong in b):

- tool_call `val_104`: correct -> wrong_args_imperial
- uncertain `val_110`: correct -> missing_field_not_named
