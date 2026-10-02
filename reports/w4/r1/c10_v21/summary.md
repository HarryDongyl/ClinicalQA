# C10 confidence diagnostics (validation)

| label | C10a n | C10a AUROC | C10b AUROC annotated | C10b AUROC grounded | Brier grounded | ECE grounded | token identity |
|---|---|---|---|---|---|---|---|
| w4_q35_4b_relabel_lr1e4_step000125 | 250 | 0.7951 | 0.9931 | 1.0 | 0.003 | 0.0043 | {'consistent': 250} |
| w4_q35_4b_relabel_lr1e4_step000250 | 250 | 0.6817 | 0.9928 | 1.0 | 0.0005 | 0.0016 | {'consistent': 250} |
| w4_q35_4b_filtered_lr1e4_step000242 | 250 | 0.8571 | 0.9949 | 1.0 | 0.001 | 0.0045 | {'consistent': 250} |
| w3_c_filtered_s42 | 250 | 0.7503 | 0.942 | 1.0 | 0.0007 | 0.0025 | {'consistent': 250} |
| w3_relabel_lr1e4_s42_step000250 | 250 | 0.6396 | 0.9108 | 1.0 | 0.0004 | 0.0018 | {'consistent': 250} |
| w3_q35_4b_filtered_lr1e4_step000242 | 250 | 0.8571 | 0.9949 | 1.0 | 0.001 | 0.0045 | {'consistent': 250} |
| w3_r0_q35_4b | 241 | 0.8144 | 0.8731 | 0.9357 | 0.1388 | 0.2032 | {'consistent': 248, 'near_tie': 2} |

C10a ranks correctness only; no ECE is computed from raw log-probabilities (D-071). C10b Brier/ECE/bins describe the first-token prefix event under the stated label policy. Checkpoints and seeds on the same questions are repeated measures, not independent samples.
