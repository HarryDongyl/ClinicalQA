# Training metrics

| run | model | n | steps | LoRA params | train loss ep1 / ep2 | val loss ep1 / ep2 (all) | grad norm med / max | s/step | train min | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| w2_filtered_lr1e4_mb1 | Qwen3-4B-Instruct-2507 | 1922 | 242 | 33.0M | 0.5306 / 0.2655 | 0.3561 / 0.3414 | 0.6764 / 4.0803 | 10.212 | 42.7 | 7.81 |
| w3_relabel_lr1e4_s42 | Qwen3-4B-Instruct-2507 | 2000 | 250 | 33.0M | 0.5214 / 0.266 | 0.3594 / 0.347 | 0.6882 / 3.9688 | 10.129 | 43.9 | 7.81 |

Val loss is teacher-forced on assistant tokens of the 250 validation conversations (by type in each run's JSON). It is a per-token loss under each model's own tokenizer: compare within a model family only. Train loss by epoch is the mean of logged step losses.
