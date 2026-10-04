# Paired comparison — w2p_v1_q5filtered_lr5e5 vs w2_filtered_lr2e4_mb1_step000121 / val / full

Paired by id; bootstrap 2000 resamples (seed 42).

| answer type | n | w2p_v1_q5filtered_lr5e5 | w2_filtered_lr2e4_mb1_step000121 | diff (b-a) | 95% CI | fixed | broken | both wrong |
|---|---|---|---|---|---|---|---|---|
| extractive | 100 | 99 | 99 | +0.000 | [+0.000, +0.000] | 0 | 0 | 1 |
| numeric_reasoning | 50 | 23 | 29 | +0.120 | [+0.000, +0.240] | 8 | 2 | 19 |
| tool_call | 62 | 55 | 55 | +0.000 | [+0.000, +0.000] | 0 | 0 | 7 |
| uncertain | 38 | 35 | 35 | +0.000 | [-0.079, +0.079] | 1 | 1 | 2 |

Broken examples (correct in a, wrong in b):

- numeric_reasoning `val_085`: correct -> number_mismatch
- numeric_reasoning `val_235`: correct -> number_mismatch
- uncertain `val_200`: correct -> fabricated_value
