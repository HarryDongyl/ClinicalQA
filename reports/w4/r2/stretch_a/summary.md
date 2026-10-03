# Stretch A results (validation; n = 19 positives + 19 probes; capability demonstration, not accuracy)

| label | e2e | called | selected | grounded | outcome | reported | KDIGO | probe fab. | probe intended | partner | core eGFR over-call (table eGFR) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| w4_sa_zs_base_q3 | 0/19 | 19/19 | 19/19 | 19/19 | 19/19 | 19/19 | 0/19 | 13/19 | 6/19 | 0/19 | 1/250 (0/21) |
| w4_sa_zs_v1_q3 | 3/19 | 19/19 | 19/19 | 19/19 | 19/19 | 19/19 | 3/19 | 5/19 | 14/19 | 3/19 | 0/250 (0/21) |
| w4_sa_zs_v1e_q3 | 3/19 | 19/19 | 19/19 | 19/19 | 19/19 | 19/19 | 3/19 | 3/19 | 16/19 | 3/19 | 0/250 (0/21) |
| w4_q3_relabel_egfr_lr1e4_step000258 | 9/19 | 19/19 | 19/19 | 19/19 | 19/19 | 19/19 | 9/19 | 16/19 | 3/19 | 9/19 | 0/250 (0/21) |

Pre-registered criteria (STRETCH_A_PLAN.md section 8): A-sft e2e >= 15/19, probe fabrication <= 2/19, core eGFR over-call on table-eGFR records <= 1/21; zero-shot arms are reported without a threshold.
