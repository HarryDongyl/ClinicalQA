# Paired comparison — w2p_v1_base vs w2p_v2_base / val / full

Paired by id; bootstrap 2000 resamples (seed 42).

| answer type | n | w2p_v1_base | w2p_v2_base | diff (b-a) | 95% CI | fixed | broken | both wrong |
|---|---|---|---|---|---|---|---|---|
| extractive | 100 | 38 | 35 | -0.030 | [-0.100, +0.030] | 4 | 7 | 58 |
| numeric_reasoning | 50 | 19 | 19 | +0.000 | [-0.080, +0.080] | 2 | 2 | 29 |
| tool_call | 62 | 31 | 34 | +0.048 | [-0.048, +0.145] | 6 | 3 | 25 |
| uncertain | 38 | 26 | 24 | -0.053 | [-0.132, +0.000] | 0 | 2 | 12 |

Broken examples (correct in a, wrong in b):

- extractive `val_045`: correct -> direction_mismatch
- extractive `val_058`: correct -> direction_mismatch
- extractive `val_061`: correct -> direction_mismatch
- extractive `val_111`: correct -> direction_mismatch
- extractive `val_206`: correct -> direction_mismatch
- numeric_reasoning `val_085`: correct -> over_refusal
- numeric_reasoning `val_089`: correct -> number_mismatch
- tool_call `val_006`: correct -> no_call
- tool_call `val_109`: correct -> no_call
- tool_call `val_165`: correct -> no_call
- uncertain `val_127`: correct -> fabricated_value
- uncertain `val_170`: correct -> fabricated_value
