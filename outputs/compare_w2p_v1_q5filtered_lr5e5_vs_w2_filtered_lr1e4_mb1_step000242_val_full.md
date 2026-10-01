# Paired comparison — w2p_v1_q5filtered_lr5e5 vs w2_filtered_lr1e4_mb1_step000242 / val / full

Paired by id; bootstrap 2000 resamples (seed 42).

| answer type | n | w2p_v1_q5filtered_lr5e5 | w2_filtered_lr1e4_mb1_step000242 | diff (b-a) | 95% CI | fixed | broken | both wrong |
|---|---|---|---|---|---|---|---|---|
| extractive | 100 | 99 | 100 | +0.010 | [+0.000, +0.030] | 1 | 0 | 0 |
| numeric_reasoning | 50 | 23 | 25 | +0.040 | [-0.040, +0.120] | 3 | 1 | 24 |
| tool_call | 62 | 55 | 54 | -0.016 | [-0.048, +0.000] | 0 | 1 | 7 |
| uncertain | 38 | 35 | 35 | +0.000 | [-0.079, +0.079] | 1 | 1 | 2 |

Broken examples (correct in a, wrong in b):

- numeric_reasoning `val_085`: correct -> number_mismatch
- tool_call `val_105`: correct -> wrong_args_imperial
- uncertain `val_025`: correct -> fabricated_value
