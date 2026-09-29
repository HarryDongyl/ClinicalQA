# Paired comparison — base_test vs raw_lr1e4_test / test / full

Paired by id; bootstrap 2000 resamples (seed 42).

| answer type | n | base_test | raw_lr1e4_test | diff (b-a) | 95% CI | fixed | broken | both wrong |
|---|---|---|---|---|---|---|---|---|
| extractive | 160 | 56 | 150 | +0.588 | [+0.500, +0.669] | 96 | 2 | 8 |
| numeric_reasoning | 80 | 20 | 45 | +0.312 | [+0.212, +0.425] | 26 | 1 | 34 |
| tool_call | 100 | 40 | 83 | +0.430 | [+0.330, +0.530] | 43 | 0 | 17 |
| uncertain | 60 | 46 | 58 | +0.200 | [+0.083, +0.333] | 14 | 2 | 0 |

Broken examples (correct in a, wrong in b):

- extractive `test_248`: correct -> low_overlap
- extractive `test_304`: correct -> low_overlap
- numeric_reasoning `test_034`: correct -> number_mismatch
- uncertain `test_020`: correct -> fabricated_value
- uncertain `test_366`: correct -> missing_field_not_named
