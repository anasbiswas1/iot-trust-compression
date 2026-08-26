# SABER 768-channel causal semantic-harm benchmark: data dictionary

File: `SABER_768_channel_causal_harm_benchmark.csv`

| Column | Meaning |
|---|---|
| architecture | `CNN1D-2block` or `DeepCNN1D-4block`. |
| benchmark_stage | Shallow development benchmark or frozen depth replication. |
| g1b_split | Shallow channel-development half, shallow held-out channel half, or external depth replication. |
| group_id | Stable structured channel identifier, e.g. `conv.0:out:17`. |
| module_path | Source Conv1d module. |
| channel_index | Original output-channel index. |
| flops_cost | First-order direct removable FLOP estimate used by the initial selector; not the realised set-level reduction. |
| magnitude | L1 channel magnitude; larger means retain. |
| taylor | First-order Taylor saliency; larger means retain. |
| fisher | Squared first-order/Fisher-style saliency; larger means retain. |
| random | Frozen random ranking control. |
| v_c | Two-stage V-C score: within-layer ASVG semantic rank multiplied by mean layer Fisher scale. |
| harm_awbir | Increase in Action-Weighted Boundary Inversion Rate after ablating the group. |
| harm_fine_macro_f1 | Teacher fine macro-F1 minus ablated-model fine macro-F1. |
| harm_family_macro_f1 | Teacher family macro-F1 minus ablated-model family macro-F1. |
| harm_hsr_balanced_soc | Increase in normalized Hierarchical Semantic Risk under the balanced-SOC cost profile. |

## Scope

- Each row is one physically coupled structured group: source Conv1d output, paired BatchNorm channel, and the corresponding downstream input slice.
- The shallow table contains 192 groups; the depth replication contains 576; total = 768.
- All harms are measured on validation data. The benchmark does not use the final SABER-held-out test audit.
- A high saliency score means the group should be retained. A high harm value means ablating it is damaging.
- The benchmark validates channel-level causal saliency. It does not by itself validate joint-set selection after recovery.
