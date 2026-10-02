# C10 confidence diagnostics (validation)

| label | C10a n | C10a AUROC | C10b AUROC annotated | C10b AUROC grounded | Brier grounded | ECE grounded | token identity |
|---|---|---|---|---|---|---|---|
| w3_r0_v1 | 231 | 0.5848 | 0.9581 | 0.9644 | 0.0983 | 0.1043 | {'consistent': 250} |
| w3_r0_v3 | 243 | 0.5003 | 0.9451 | 0.9546 | 0.183 | 0.1878 | {'consistent': 250} |
| w3_r0_v3_fs4 | 248 | 0.6704 | 0.9816 | 0.9872 | 0.0735 | 0.074 | {'consistent': 250} |
| w3_c_filtered_s42 | 250 | 0.7503 | 0.942 | 1.0 | 0.0007 | 0.0025 | {'consistent': 250} |
| w3_relabel_lr1e4_s42_step000125 | 250 | 0.7478 | 0.9165 | 1.0 | 0.0032 | 0.0047 | {'consistent': 250} |
| w3_relabel_lr1e4_s42_step000250 | 250 | 0.6396 | 0.9108 | 1.0 | 0.0004 | 0.0018 | {'consistent': 250} |
| w3_r0_8b | 224 | 0.6659 | 0.9671 | 0.9638 | 0.1276 | 0.1454 | {'consistent': 250} |

C10a ranks correctness only; no ECE is computed from raw log-probabilities (D-071). C10b Brier/ECE/bins describe the first-token prefix event under the stated label policy. Checkpoints and seeds on the same questions are repeated measures, not independent samples.
