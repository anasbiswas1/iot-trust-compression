# SABER-IDS first-phase run order

## Notebook 14 — taxonomy and ASVG

Expected duration: light to medium, mostly one validation forward pass.

Manual checks:
- class order is exact;
- all labels map to a family;
- source coverage is near complete;
- top directed edges are operationally plausible;
- cost profiles are frozen before Notebook 16.

## Notebook 15 — baseline channel scores

Expected duration: light to medium.

Manual checks:
- every Conv1d channel has a downstream dependency;
- physical one-channel-per-layer smoke test succeeds;
- Taylor and Fisher are finite and nonzero.

## Notebook 16 — R-SBL causal score validation

Expected duration: medium. This is the most important early notebook.

The single-group loop is resumable. It uses a deterministic balanced validation
subset, not the test set.

Proceed only if `G1_score_gate.json` is credible. Also inspect the scatter plots;
a pass based on one outlier is not sufficient.

## Notebook 17 — structured pruning screen

Expected duration: medium to heavy depending on fine-tuning epochs.

This is validation-only method elimination at three budgets. It creates
physically smaller models and compares random, magnitude, Taylor, Fisher and
R-SBL under the same training budget.

## Notebook 18 — hierarchical boundary distillation

Expected duration: heavy.

Compares CE+KD, fixed HSBD and dual-constrained HSBD. Do not tune against the
test partition. Notebook 19 will later repeat shortlisted variants over
independent seeds and datasets.

## Later notebooks, created after the early gates

- 19: independent multi-seed, multi-dataset confirmation
- 20: deployment Pareto and hardware measurement
- 21: ablations and confirmatory statistics
- 22: hierarchical selective fallback
- 23: distribution-shift stress
- 24: deterministic final figures/tables
