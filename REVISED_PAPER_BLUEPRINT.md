# Revised paper blueprint after G5

## Recommended primary paper identity

**From Channel Harm to Recovered Pruning Sets: An Alert-Semantic Causal Benchmark for IoT Intrusion Detection**

This framing survives whether or not the optional method-rescue experiments succeed.

## Central question

> When does a saliency score that predicts the causal harm of removing one channel transfer to a physically pruned, recovered deployment at matched realised cost?

## Contribution order

### C1. Alert-semantic causal benchmark

Release 768 physically defined single-group ablations across a shallow and deeper CNN, with:

- AWBIR;
- HSR under three frozen cost profiles;
- fine- and family-level macro-F1 harm;
- layer/dependency metadata;
- magnitude, Taylor, Fisher, V-C, and random scores.

### C2. Validation-built semantic graph with destination validity

Show that ASVG top edges recover later test-time prune80 absorber destinations, including high-mass consolidation sinks. This validates the semantic object independently of channel saliency.

### C3. V-C as a two-stage diagnostic

Present V-C exactly:

1. curvature/Fisher allocates importance across layers;
2. ASVG boundary leverage ranks channels within each layer.

Report shallow selection/holdout development and the fully frozen depth replication. State uncertainty; do not claim universal superiority.

### C4. Realised-cost calibration protocol

Show that direct channel-cost budgeting can overshoot nominal budgets by 19–42 percentage points and method-dependently confound comparisons. Introduce binary-search prefix calibration and verify all method–budget cells within 1.5 percentage points of target realised FLOPs.

### C5. Channel-to-set-to-recovery regime map

Separate:

- one-channel causal validity;
- joint raw-set harm;
- minimal recovery;
- full recovery.

The core new finding should be stated conditionally after Notebook 22:

- **recovery-mediated dissociation** if additive raw harm predicts joint damage but recovery reorders sets;
- **interaction-plus-recovery dissociation** if additive confirmation also fails.

### C6. Optional method contribution

Include Alert-Semantic Lookahead Selection or boundary-aligned HSBD-v2 only if its frozen gate passes and multi-seed confirmation supports it. Otherwise keep it as a negative ablation in the supplement.

## Proposed main-text structure

### 1. Introduction

- Edge compression and class/alert semantics.
- Diagnostic paper as motivation, not reused contribution.
- Saliency-validation problem: one-channel validity is usually assumed to transfer to a selected/recovered model.
- Contributions C1–C5; C6 only if successful.

### 2. Related work

- Channel saliency taxonomies and metric design.
- Structural dependencies and group Fisher/Domino metrics.
- Oracle/lookahead and joint/combinatorial channel selection.
- Recovery/reconstruction after pruning.
- Cost-sensitive and hierarchical IDS.
- Explicit novelty boundary: alert-semantic causal benchmark and stage-dissociation, not first lookahead or first class-aware pruning.

### 3. Alert-semantic risk framework

- Binary–family–fine taxonomy.
- Cost profiles and robust aggregation.
- ASVG construction.
- AWBIR and HSR.
- ASVG absorber-destination validation.

### 4. Causal channel benchmark and V-C

- Physically coupled group definition.
- Single-group ablation protocol.
- G1 failure and why normalisation/cancellation mattered.
- V-C formulation.
- Shallow held-out channel subset.
- Layer decomposition.
- Frozen depth replication with bootstrap intervals.

### 5. Realised-cost set construction

- Why direct-cost budgets fail.
- Prefix calibration algorithm.
- Parameter/FLOP verification.
- Reproducibility/checkpoint registry.

### 6. From channels to recovered sets

- Random set interaction benchmark.
- Additive discovery/confirmation results.
- Raw set damage.
- Minimal/full recovery reordering.
- Shallow/deep regime map.
- G5 negative result reported plainly.

### 7. Optional algorithmic extensions

- Alert-Semantic Lookahead Selection.
- Boundary-aligned HSBD-v2.
- Only retained in the main paper if they pass frozen criteria.

### 8. SABER-held-out and external evaluation

- One-shot test audit.
- Three/five independent seeds for headline cells.
- Fresh external dataset arm.
- Hardware proxy or real edge hardware.

### 9. Discussion and limitations

- What single-channel ablation validates and what it does not.
- Recovery as part of the pruning algorithm.
- Historical use of CICIoT2023 test data disclosed.
- One-dataset/one-seed screening boundaries.
- Cost-profile sensitivity.

### 10. Conclusion and artifact release

## Principal figures

1. ASVG and observed absorber recovery.
2. 768-group causal benchmark design.
3. V-C shallow holdout with bootstrap uncertainty.
4. Frozen depth replication.
5. Nominal versus realised FLOP calibration.
6. Additive predicted versus observed joint raw harm.
7. Raw-to-recovered rank/recovery regime map.
8. Channel validity versus recovered-set ranking.
9. Optional ASL/HSBD-v2 result if successful.
10. Test and external Pareto curves.

## Principal tables

1. Benchmark architectures, groups, outcomes, and splits.
2. ASVG absorber recovery summary.
3. V-C versus baseline channel validity with intervals.
4. Realised-cost calibration audit.
5. Set interaction confirmation metrics.
6. Recovery rank-reordering summary.
7. Final test multi-metric results.
8. Multi-seed/external confirmation.

## Submission gate

Do not write the final Q1 method claim until:

- Notebook 21 test audit is locked;
- Notebook 22 identifies the dissociation mechanism;
- at least three seeds confirm the final primary result;
- at least one fresh external dataset reproduces the direction;
- if an algorithm is claimed, it beats Taylor/Fisher at matched realised cost on AWBIR plus an operational metric without worsening attack misses.
