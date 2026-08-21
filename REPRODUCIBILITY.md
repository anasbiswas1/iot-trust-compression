# Reproducibility map and interpretation boundaries

This repository contains two evidence layers that must not be conflated.

## 1. Archived manuscript evidence

| Component | Canonical code | Representative output |
|---|---|---|
| Baseline and compression matrix | `src/train.py`, `src/compression.py` | `results/tables/compression/cnn1d_per_class_recall_matrix.csv` |
| Class deterioration and calibration | `src/metrics.py` and archived notebooks | `results/tables/explain/global_metrics_by_cell.csv`, `overall_ece_by_cell.csv` |
| Archived relative OvR probe | `src/explain.py::crux_probe` | `results/tables/explain/cnn_crux_probe_prune80.csv` |
| Leakage-safer OvR point-estimate replication | archived audit code/output | `results/tables/explain/crux_probe_leakage_safe.csv` |
| Pairwise probe | `src/mitigate.py::pairwise_vs_ovr_separability` | `results/tables/explain/pairwise_vs_ovr_prune80.csv` |
| Consolidation | `src/explain.py::confusability_reallocation` | `results/tables/explain/prune80_confusability.csv` |
| Rank | `src/explain.py::rank_collapse_analysis` | `results/tables/explain/rank_collapse_global.csv` |
| Archived baseline diagnostic | `src/predict.py` | `results/tables/explain/cnn_predict_table.csv` |
| Recovery | `src/mitigate.py` | `results/tables/explain/mitigate_per_class_recovery.csv` |
| TON_IoT check | archived dataset-specific pipeline | `results/tables/explain/ton_crux_probe.csv` |

### Archived diagnostic qualification

`src/predict.py::assemble_features` defaults to `which="test"`, and the archived predictor covariates were assembled on the evaluation partition. The current manuscript therefore treats this result as an **exploratory retrospective diagnostic**, not validated pre-deployment forecasting. Notebook 10 performs the stronger validation-only covariate analysis.

### Archived seed qualification

The archived five-run prune80 analysis uses one deterministic magnitude mask from one anchor baseline and varies fine-tuning seeds. It measures optimisation-trajectory sensitivity under a fixed mask. Notebook 11 is the independent baseline-mask replication.

### Archived probe qualification

The original probe randomly splits one held-out partition and is useful for a relative M0-versus-compressed comparison. The stricter point-estimate replication fits on training/validation representations and evaluates on the untouched provenance-ordered test partition. Notebook 12 makes this protocol reproducible, adds paired bootstrap intervals, and repeats the pairwise probes.

## 2. Computer Networks extension evidence

All notebooks below write only under `results/tables/comnet/`.

| Notebook | New output family |
|---|---|
| 09 | `security_semantics_by_cell.csv`, `family_metrics_by_cell.csv`, `substitution_semantics_by_cell.csv`, extended calibration and recovered-head audits |
| 10 | validation-defined tiers, tier-threshold sensitivity, validation-only CNN/MLP architecture gate, validation-only features, test-only targets, family-held-out predictor results |
| 11 | independent paired seed summaries, per-class effects, mask Jaccard tables |
| 12 | strict OvR/pairwise probes with intervals, BatchNorm recalibration, dense-versus-sparse head controls |
| 13 | actual serialization and dense CPU deployment measurements |

These outputs do not become manuscript evidence automatically. Inspect the run logs, check class ordering, and reconcile every manuscript number against the exact CSV before insertion.

## 3. No-rerun code and documentation corrections

The update package safely:

- replaces stale repository claims with the bounded manuscript interpretation;
- removes dead public compatibility stubs;
- corrects the int8 method metadata to dynamic Linear-layer quantisation;
- renames the benign recovery metric as a delta in false-positive rate while retaining backward-compatible aliases;
- adds credential exclusions and security guidance;
- adds tested analysis helpers and journal-facing notebooks.

It does **not** change archived numerical result files.

## Environment and release

Run:

```bash
python -m pytest -q tests/test_compat_modules.py tests/test_comnet_audit.py
```

Before submission, create a release tag and record:

- Git tag;
- commit hash;
- Python and PyTorch versions;
- hardware used for Notebook 13;
- any notebook configuration changed from the defaults.
