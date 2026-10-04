# Paired comparison — w2p_v1_raw_lr5e5 vs w2p_v2_raw_lr5e5 / val / full

Paired by id; bootstrap 2000 resamples (seed 42).

| answer type | n | w2p_v1_raw_lr5e5 | w2p_v2_raw_lr5e5 | diff (b-a) | 95% CI | fixed | broken | both wrong |
|---|---|---|---|---|---|---|---|---|
| extractive | 100 | 98 | 99 | +0.010 | [+0.000, +0.030] | 1 | 0 | 1 |
| numeric_reasoning | 50 | 26 | 26 | +0.000 | [-0.060, +0.060] | 1 | 1 | 23 |
| tool_call | 62 | 54 | 54 | +0.000 | [+0.000, +0.000] | 0 | 0 | 8 |
| uncertain | 38 | 34 | 34 | +0.000 | [+0.000, +0.000] | 0 | 0 | 4 |

Broken examples (correct in a, wrong in b):

- numeric_reasoning `val_235`: correct -> number_mismatch
