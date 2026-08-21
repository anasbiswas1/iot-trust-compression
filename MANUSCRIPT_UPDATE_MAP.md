# Map new notebook outputs to the Computer Networks manuscript

Do not paste a result into the paper merely because a notebook completed. First check the run log, class order, checkpoint identity, and CSV schema. Keep the current bounded wording whenever a new result is incomplete or inconsistent.

## Notebook 09 — network-security semantics and recovery audit

Primary outputs:

- `security_semantics_by_cell.csv`
- `family_metrics_by_cell.csv`
- `substitution_semantics_by_cell.csv`
- `calibration_extended_by_cell.csv`
- `security_semantics_head_refit.csv`
- `family_metrics_head_refit.csv`
- `substitution_semantics_head_refit.csv`
- `calibration_head_refit.csv`
- `ciciot2023_alert_family_mapping.csv`

Use these to add one compact main-text table reporting binary attack detection, benign false-alert rate, family macro-F1, and fine-type substitution categories. Put the complete per-family table in the supplement. Update the abstract only if the attack-to-benign and benign-to-attack rates have been independently checked against the fine-label confusion matrix.

## Notebook 10 — validation-frozen tiers and stronger diagnostic

Primary outputs:

- `validation_defined_tiers.csv`
- `validation_tier_threshold_sensitivity.csv`
- `validation_architecture_gate_cnn.csv`
- `validation_architecture_gate.csv`
- `prediction_features_validation_only.csv`
- `test_recall_loss_targets.csv`
- `prediction_validation_features_family_heldout.csv`

Replace the current test-informed tier limitation only after the validation-defined class set is frozen. Report overlap with the previous measurable/robust set. Restore a pre-deployment prediction claim only if validation-only covariates consistently beat the mean and frequency baselines under family-held-out evaluation; otherwise retain the retrospective-diagnostic wording.

## Notebook 11 — independent paired baseline/pruning seeds

Primary outputs:

- `paired_seed_run_summary.csv`
- `paired_seed_per_class_effects.csv`
- `paired_seed_per_class_summary.csv`
- `paired_seed_mask_jaccard.csv`
- `paired_seed_macro_f1_wide.csv`
- `paired_seed_macro_f1_summary.csv`

Replace the fixed-mask sensitivity paragraph with paired baseline-to-compression effects. The main paper should report mean paired macro-F1 change, the frequency of material class deterioration, and the stable core of affected classes. Put full seed-by-class values and mask-overlap results in the supplement.

## Notebook 12 — strict probes and causal controls

Primary outputs:

- `strict_probe_trainval_to_test.csv`
- `strict_pairwise_probe_trainval_to_test.csv`
- `batchnorm_recalibration_summary.csv`
- `dense_vs_sparse_head_refit.csv`
- associated per-class CSVs

Make the strict train/validation-to-test probe the primary crux table. Keep the older within-test probe only as a sensitivity analysis. If BatchNorm-only recalibration materially restores performance, qualify the immediate one-shot pruning interpretation. Use the sparse-head comparison to distinguish decision-boundary re-optimisation from dense head-capacity restoration.

## Notebook 13 — realised deployment measurements

Primary outputs:

- `deployment_size_and_sparsity.csv`
- `deployment_cpu_latency_throughput.csv`
- `deployment_environment.json`

Report actual serialized bytes and dense-backend CPU measurements exactly as measured. Do not call Colab CPU hardware a physical edge gateway. Do not infer sparse acceleration from zeros unless a sparse execution backend is separately measured.

## Final reconciliation

After all accepted outputs are inserted:

1. update abstract, Results, Discussion, Limitations, and Conclusion together;
2. update the supplement and all table/figure cross-references;
3. update the README to the same numerical claims;
4. add the release tag and commit hash to Data and Code Availability;
5. render the DOCX page by page and re-audit every table and figure;
6. archive the exact manuscript-facing release.
