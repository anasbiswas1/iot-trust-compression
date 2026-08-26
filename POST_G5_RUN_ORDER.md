# Post-G5 run order

## Phase A — provenance repair

1. Apply the source files in `src/saber/`.
2. Run `17b_realized_flop_calibration_and_checkpoint_freeze.ipynb`.
3. Run `20b_freeze_depth_students.ipynb`.
4. Inspect both registry audit JSON files and reconcile validation metrics.

## Phase B — one-shot test audit

5. Open `21_one_shot_saber_test_audit.ipynb`.
6. Review the complete frozen model registry.
7. Set `CONFIRM_ONE_SHOT_TEST = True` once.
8. Run the notebook once. Do not tune from, overwrite, or rerun the test results.

## Phase C — resolve the failed transfer

9. Run `22_set_level_interaction_and_recovery_reordering.ipynb`.
10. Read `mechanism_verdict.json`:
    - additive raw harm + recovery reordering => recovery-mediated dissociation;
    - poor raw additivity + reordering => interaction-plus-recovery dissociation.

## Phase D — optional method rescue

11. Run the shallow arm of `23_alert_semantic_lookahead_selection.ipynb`.
12. Enable the depth arm only if the shallow gate is credible.
13. Run `24_boundary_aligned_hsbd_v2.ipynb`.

## Phase E — confirmation

14. If neither method-rescue gate passes, write the benchmark/methodology paper and do not claim a new superior pruning method.
15. If one passes, run three independent screening seeds, then five confirmatory seeds.
16. Add a fresh external dataset arm before a strong Q1 submission.

## Important

The CICIoT2023 test partition was used in the earlier diagnostic paper. It is held out from SABER development, but not globally unseen. Use the phrase **SABER-held-out test audit** and add a fresh external confirmation.
