# Computer Networks extension run order

## Before running anything

1. Apply the update package in dry-run mode.
2. Rotate any credential that may have been exposed and remove the credential file from Git history.
3. Confirm the dataset and model checkpoints exist in the repository's configured Drive paths.
4. Install `requirements.txt` and `requirements-comnet.txt`.
5. Create a clean Git branch, for example `computer-networks-revision`.

## Recommended order

### 1. Notebook 09 - security semantics and recovery audit

Run first. It normally reuses checkpoints.

Required outputs:

- `security_semantics_by_cell.csv`
- `family_metrics_by_cell.csv`
- `substitution_semantics_by_cell.csv`
- `calibration_extended_by_cell.csv`
- recovered-head equivalents
- `ciciot2023_alert_family_mapping.csv`

**Gate:** inspect the family mapping before interpreting same-family substitutions as response-equivalent.

### 2. Notebook 10 - validation-frozen tiers and safe diagnostic

Normally reuses baseline checkpoints and archived compression targets.

Required outputs:

- `validation_defined_tiers.csv`
- `validation_tier_threshold_sensitivity.csv`
- `validation_architecture_gate.csv` (CNN/MLP checkpoints found; missing seeds recorded)
- `prediction_features_validation_only.csv`
- `test_recall_loss_targets.csv`
- `prediction_validation_features_family_heldout.csv`

**Gate:** restore a pre-deployment diagnostic claim only if validation-only covariates beat mean and frequency baselines consistently under family-held-out evaluation.

### 3. Notebook 12 - strict probes and controls

Run before the heavy independent-seed experiment if resources allow.

Required outputs:

- `strict_probe_trainval_to_test.csv`
- `strict_pairwise_probe_trainval_to_test.csv`
- `batchnorm_recalibration_summary.csv`
- `dense_vs_sparse_head_refit.csv`

**Gate:** keep the probe conclusion at “retained relative decodability” if any class shows meaningful strict-protocol degradation.

### 4. Notebook 13 - deployment benchmark

Run on the actual hardware you intend to describe. A Colab CPU is an edge proxy, not a physical gateway.

Required outputs:

- `deployment_size_and_sparsity.csv`
- `deployment_cpu_latency_throughput.csv`
- `deployment_environment.json`

**Gate:** do not claim sparse speedup unless a genuinely sparse execution backend is measured.

### 5. Notebook 11 - independent paired seeds

This is the heaviest and most important new training run.

Required outputs:

- `paired_seed_run_summary.csv`
- `paired_seed_per_class_effects.csv`
- `paired_seed_per_class_summary.csv`
- `paired_seed_mask_jaccard.csv`
- `paired_seed_macro_f1_summary.csv`

**Gate:** replace the fixed-mask sensitivity wording only after every paired seed completes and checkpoint/class ordering is verified.

## Manuscript update order

1. Insert binary/family/substitution and deployment results.
2. Replace test-informed tiers with validation-frozen results.
3. Replace the current probe table with the strict protocol if complete.
4. Update recovery with post-refit calibration and alert semantics.
5. Update the independent-seed paragraph and uncertainty claims.
6. Re-render the manuscript and supplement page by page.
7. Freeze a release tag and add tag/commit information to Data and Code Availability.
