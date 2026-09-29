# Training supervision policy

Generated from train only; canonical splits and targets are unchanged.

Q5 filtering is a heuristic quarantine, not clinical adjudication. No missing gold measurements are inserted into inputs.

| Run | Train rows | Excluded | Remaining Q5 |
|---|---:|---:|---:|
| raw_lr1e4 | 2000 | 0 | 78 |
| raw_lr5e5 | 2000 | 0 | 78 |
| q5filtered_lr5e5 | 1922 | 78 | 0 |

Raw controls explicitly retain unsupported targets. The q5filtered run is the mitigated candidate.
Compare raw_lr1e4 with raw_lr5e5 for LR; compare raw_lr5e5 with q5filtered_lr5e5 for filtering.
Filtering changes sample count, type distribution and optimizer-step count; it is a data-policy ablation, not a pure LR comparison.

## Quarantined train candidates

- `train_009`: calculate_bmi {"weight_kg": 88.8, "height_cm": 169.1}: weight, height not found in input; no such measurement in input
- `train_023`: calculate_bmi {"weight_kg": 103.9, "height_cm": 177.9}: weight, height not found in input; no such measurement in input
- `train_105`: calculate_bmi {"weight_kg": 71.7, "height_cm": 190.2}: weight, height not found in input; no such measurement in input
- `train_135`: calculate_bmi {"weight_kg": 111.1, "height_cm": 173.5}: weight, height not found in input; no such measurement in input
- `train_146`: calculate_bmi {"weight_kg": 74.8, "height_cm": 169.8}: weight, height not found in input; no such measurement in input
- `train_168`: calculate_bmi {"weight_kg": 100.0, "height_cm": 161.6}: weight, height not found in input; no such measurement in input
- `train_175`: calculate_bmi {"weight_kg": 102.5, "height_cm": 158.7}: weight, height not found in input; no such measurement in input
- `train_193`: calculate_bmi {"weight_kg": 89.3, "height_cm": 191.8}: weight, height not found in input; no such measurement in input
- `train_269`: calculate_bmi {"weight_kg": 54.0, "height_cm": 177.4}: weight, height not found in input; no such measurement in input
- `train_271`: calculate_bmi {"weight_kg": 117.4, "height_cm": 169.7}: weight, height not found in input; no such measurement in input
- `train_290`: calculate_bmi {"weight_kg": 81.7, "height_cm": 150.4}: weight, height not found in input; no such measurement in input
- `train_321`: calculate_bmi {"weight_kg": 62.3, "height_cm": 193.9}: weight, height not found in input; no such measurement in input
- `train_456`: calculate_bmi {"weight_kg": 103.9, "height_cm": 153.1}: weight, height not found in input; no such measurement in input
- `train_491`: calculate_bmi {"weight_kg": 111.4, "height_cm": 166.1}: weight, height not found in input; no such measurement in input
- `train_516`: calculate_bmi {"weight_kg": 72.4, "height_cm": 193.8}: weight, height not found in input; no such measurement in input
- `train_521`: calculate_bmi {"weight_kg": 68.3, "height_cm": 183.7}: weight, height not found in input; no such measurement in input
- `train_552`: calculate_bmi {"weight_kg": 79.3, "height_cm": 161.6}: weight, height not found in input; no such measurement in input
- `train_556`: calculate_bmi {"weight_kg": 75.3, "height_cm": 168.6}: weight, height not found in input; no such measurement in input
- `train_610`: calculate_bmi {"weight_kg": 62.6, "height_cm": 181.4}: weight, height not found in input; no such measurement in input
- `train_708`: calculate_bmi {"weight_kg": 110.8, "height_cm": 165.2}: weight, height not found in input; no such measurement in input
- `train_718`: calculate_bmi {"weight_kg": 77.0, "height_cm": 178.8}: weight, height not found in input; no such measurement in input
- `train_746`: calculate_bmi {"weight_kg": 58.5, "height_cm": 179.7}: weight, height not found in input; no such measurement in input
- `train_780`: calculate_bmi {"weight_kg": 98.7, "height_cm": 181.4}: weight, height not found in input; no such measurement in input
- `train_827`: calculate_bmi {"weight_kg": 82.5, "height_cm": 157.6}: weight, height not found in input; no such measurement in input
- `train_830`: calculate_bmi {"weight_kg": 102.2, "height_cm": 155.0}: weight, height not found in input; no such measurement in input
- `train_903`: calculate_bmi {"weight_kg": 88.5, "height_cm": 177.3}: weight, height not found in input; no such measurement in input
- `train_918`: calculate_bmi {"weight_kg": 83.7, "height_cm": 154.2}: weight, height not found in input; no such measurement in input
- `train_961`: calculate_bmi {"weight_kg": 55.2, "height_cm": 193.2}: weight, height not found in input; no such measurement in input
- `train_984`: calculate_bmi {"weight_kg": 67.3, "height_cm": 186.1}: weight, height not found in input; no such measurement in input
- `train_997`: calculate_bmi {"weight_kg": 104.5, "height_cm": 166.6}: weight, height not found in input; no such measurement in input
- `train_998`: calculate_bmi {"weight_kg": 67.5, "height_cm": 184.1}: weight, height not found in input; no such measurement in input
- `train_1038`: calculate_bmi {"weight_kg": 65.2, "height_cm": 167.4}: weight, height not found in input; no such measurement in input
- `train_1064`: calculate_bmi {"weight_kg": 93.8, "height_cm": 167.2}: weight, height not found in input; no such measurement in input
- `train_1072`: calculate_bmi {"weight_kg": 55.6, "height_cm": 183.6}: weight, height not found in input; no such measurement in input
- `train_1123`: calculate_bmi {"weight_kg": 66.3, "height_cm": 170.0}: weight, height not found in input; no such measurement in input
- `train_1127`: calculate_bmi {"weight_kg": 55.0, "height_cm": 172.8}: weight, height not found in input; no such measurement in input
- `train_1153`: calculate_bmi {"weight_kg": 88.3, "height_cm": 179.0}: weight, height not found in input; no such measurement in input
- `train_1163`: calculate_bmi {"weight_kg": 88.6, "height_cm": 156.6}: weight, height not found in input; no such measurement in input
- `train_1179`: calculate_bmi {"weight_kg": 80.9, "height_cm": 161.5}: weight, height not found in input; no such measurement in input
- `train_1190`: calculate_bmi {"weight_kg": 55.9, "height_cm": 174.2}: weight, height not found in input; no such measurement in input
- `train_1203`: calculate_bmi {"weight_kg": 60.3, "height_cm": 181.2}: weight, height not found in input; no such measurement in input
- `train_1212`: calculate_bmi {"weight_kg": 80.6, "height_cm": 160.8}: weight, height not found in input; no such measurement in input
- `train_1225`: calculate_bmi {"weight_kg": 106.9, "height_cm": 153.7}: weight, height not found in input; no such measurement in input
- `train_1296`: calculate_bmi {"weight_kg": 62.3, "height_cm": 166.8}: weight, height not found in input; no such measurement in input
- `train_1350`: calculate_bmi {"weight_kg": 62.4, "height_cm": 182.8}: weight, height not found in input; no such measurement in input
- `train_1364`: calculate_bmi {"weight_kg": 59.4, "height_cm": 172.5}: weight, height not found in input; no such measurement in input
- `train_1384`: calculate_bmi {"weight_kg": 117.3, "height_cm": 185.4}: weight, height not found in input; no such measurement in input
- `train_1385`: calculate_bmi {"weight_kg": 73.1, "height_cm": 154.7}: weight, height not found in input; no such measurement in input
- `train_1387`: calculate_bmi {"weight_kg": 66.2, "height_cm": 182.9}: weight, height not found in input; no such measurement in input
- `train_1416`: calculate_bmi {"weight_kg": 84.0, "height_cm": 168.5}: weight, height not found in input; no such measurement in input
- `train_1452`: calculate_bmi {"weight_kg": 80.2, "height_cm": 167.4}: weight, height not found in input; no such measurement in input
- `train_1464`: calculate_bmi {"weight_kg": 116.9, "height_cm": 166.0}: weight, height not found in input; no such measurement in input
- `train_1478`: calculate_bmi {"weight_kg": 87.6, "height_cm": 150.8}: weight, height not found in input; no such measurement in input
- `train_1483`: calculate_bmi {"weight_kg": 111.6, "height_cm": 175.7}: weight, height not found in input; no such measurement in input
- `train_1500`: calculate_bmi {"weight_kg": 95.1, "height_cm": 154.9}: weight, height not found in input; no such measurement in input
- `train_1514`: calculate_bmi {"weight_kg": 118.9, "height_cm": 185.9}: weight, height not found in input; no such measurement in input
- `train_1515`: calculate_bmi {"weight_kg": 57.2, "height_cm": 186.1}: weight, height not found in input; no such measurement in input
- `train_1556`: calculate_bmi {"weight_kg": 89.3, "height_cm": 190.6}: weight, height not found in input; no such measurement in input
- `train_1584`: calculate_bmi {"weight_kg": 80.6, "height_cm": 168.3}: weight, height not found in input; no such measurement in input
- `train_1588`: calculate_bmi {"weight_kg": 101.7, "height_cm": 151.5}: weight, height not found in input; no such measurement in input
- `train_1594`: calculate_bmi {"weight_kg": 61.0, "height_cm": 183.4}: weight, height not found in input; no such measurement in input
- `train_1621`: calculate_bmi {"weight_kg": 50.4, "height_cm": 167.3}: weight, height not found in input; no such measurement in input
- `train_1683`: calculate_bmi {"weight_kg": 102.5, "height_cm": 179.3}: weight, height not found in input; no such measurement in input
- `train_1688`: calculate_bmi {"weight_kg": 71.1, "height_cm": 182.0}: weight, height not found in input; no such measurement in input
- `train_1714`: calculate_bmi {"weight_kg": 84.5, "height_cm": 160.9}: weight, height not found in input; no such measurement in input
- `train_1716`: calculate_bmi {"weight_kg": 63.7, "height_cm": 185.3}: weight, height not found in input; no such measurement in input
- `train_1740`: calculate_bmi {"weight_kg": 67.5, "height_cm": 165.3}: weight, height not found in input; no such measurement in input
- `train_1748`: calculate_bmi {"weight_kg": 105.0, "height_cm": 155.5}: weight, height not found in input; no such measurement in input
- `train_1767`: calculate_bmi {"weight_kg": 107.8, "height_cm": 169.3}: weight, height not found in input; no such measurement in input
- `train_1823`: calculate_bmi {"weight_kg": 74.2, "height_cm": 171.8}: weight, height not found in input; no such measurement in input
- `train_1833`: calculate_bmi {"weight_kg": 74.4, "height_cm": 170.1}: weight, height not found in input; no such measurement in input
- `train_1923`: calculate_bmi {"weight_kg": 110.9, "height_cm": 151.6}: weight, height not found in input; no such measurement in input
- `train_1943`: calculate_bmi {"weight_kg": 91.4, "height_cm": 162.5}: weight, height not found in input; no such measurement in input
- `train_1946`: calculate_bmi {"weight_kg": 67.9, "height_cm": 187.3}: weight, height not found in input; no such measurement in input
- `train_1954`: calculate_bmi {"weight_kg": 83.4, "height_cm": 151.1}: weight, height not found in input; no such measurement in input
- `train_1960`: calculate_bmi {"weight_kg": 87.4, "height_cm": 174.1}: weight, height not found in input; no such measurement in input
- `train_1970`: calculate_bmi {"weight_kg": 71.7, "height_cm": 194.3}: weight, height not found in input; no such measurement in input
- `train_1973`: calculate_bmi {"weight_kg": 75.2, "height_cm": 169.4}: weight, height not found in input; no such measurement in input
