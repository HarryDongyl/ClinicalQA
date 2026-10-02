# Training metrics

| run | model | n | steps | LoRA params | train loss ep1 / ep2 | val loss ep1 / ep2 (all) | grad norm med / max | s/step | train min | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| w4_q35_4b_relabel_lr1e4 | Qwen3.5-4B | 2000 | 250 | 30.5M | 0.3499 / 0.1896 | 0.2732 / 0.2685 | 0.7981 / 4.0476 | 11.659 | 56.6 | 11.3 |
| w3_q35_4b_filtered_lr1e4 | Qwen3.5-4B | 1922 | 242 | 30.5M | 0.3501 / 0.1914 | 0.274 / 0.2663 | 0.7497 / 3.7189 | 11.762 | 63.3 | 11.3 |
| w3_relabel_lr1e4_s42 | Qwen3-4B-Instruct-2507 | 2000 | 250 | 33.0M | 0.5214 / 0.266 | 0.3594 / 0.347 | 0.6882 / 3.9688 | 10.129 | 43.9 | 7.81 |

Val loss is teacher-forced on assistant tokens of the 250 validation conversations (by type in each run's JSON). It is a per-token loss under each model's own tokenizer: compare within a model family only. Train loss by epoch is the mean of logged step losses.
