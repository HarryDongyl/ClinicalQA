# Paired comparison — w2p_v1_q5filtered_lr5e5 vs w2_filtered_lr2e4_mb1_step000242 / val / full

Paired by id; bootstrap 2000 resamples (seed 42).

| answer type | n | w2p_v1_q5filtered_lr5e5 | w2_filtered_lr2e4_mb1_step000242 | diff (b-a) | 95% CI | fixed | broken | both wrong |
|---|---|---|---|---|---|---|---|---|
| extractive | 100 | 99 | 99 | +0.000 | [+0.000, +0.000] | 0 | 0 | 1 |
| numeric_reasoning | 50 | 23 | 27 | +0.080 | [-0.020, +0.180] | 6 | 2 | 21 |
| tool_call | 62 | 55 | 54 | -0.016 | [-0.048, +0.000] | 0 | 1 | 7 |
| uncertain | 38 | 35 | 37 | +0.053 | [+0.000, +0.132] | 2 | 0 | 1 |

Broken examples (correct in a, wrong in b):

- numeric_reasoning `val_085`: correct -> number_mismatch
- numeric_reasoning `val_227`: correct -> number_mismatch
- tool_call `val_105`: correct -> wrong_args_imperial
