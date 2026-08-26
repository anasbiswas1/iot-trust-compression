# SABER-IDS post-G5 forensic audit and recovery plan

**Repository basis:** uploaded `saber-ids-method` branch ZIP, through Notebook 20.  
**Audit date:** 26 August 2026.  
**Scope:** Notebooks 14–20, `src/saber/`, all committed `results/saber/` CSV/JSON outputs, and the historical diagnostic tables used for absorber validation.

## Executive verdict

G5 did fail, but the project did **not** collapse into a null result. What failed is the strongest original algorithmic claim:

> A channel score that predicts one-channel alert-semantic harm will, when sorted statically, produce the safest recovered pruning set.

The branch supports a different and scientifically valuable result:

1. The validation-only Alert-Semantic Vulnerability Graph (ASVG) independently anticipates later test-time prune80 absorber destinations.
2. The two-stage V-C score is a strong **channel-level boundary-harm diagnostic**, especially for AWBIR, and this replicates in a deeper architecture.
3. Direct additive-cost budgeting is invalid for physical channel surgery; realised-FLOP calibration is necessary.
4. Static channel validity and recovered set quality are different objects. Recovery can compress, reorder, or reverse method differences.
5. The original HSBD dual is misaligned with the actual benign false-alert event; its surrogate is not a boundary-crossing proxy.
6. The branch now contains a 768-group causal semantic-harm benchmark across two architectures. That artifact is substantial and publishable if paired with set-level, multi-seed, and final held-out evidence.

The scientifically correct response is **not** to hide G5 or keep tuning V-C until it wins. The response is to make the failed transfer itself the central methodological question and run experiments that distinguish set interaction from recovery dynamics.

## 1. Gate audit

### G1: original R-SBL failed legitimately

The original cost-normalised R-SBL score did not predict measured one-channel harm:

| Outcome | R-SBL Spearman | Best magnitude/Taylor comparator | Difference |
|---|---:|---:|---:|
| AWBIR harm | 0.030 | 0.233 | -0.203 |
| HSR harm | -0.082 | 0.017 | -0.099 |
| Fine macro-F1 harm | -0.084 | 0.062 | -0.146 |
| Family macro-F1 harm | -0.099 | 0.065 | -0.165 |

This failure is consistent with four implementation-object mismatches in the first score:

- gradients were averaged before taking the absolute value, permitting cancellation;
- the score did not cover every tensor removed by the physical dependency group;
- the score population did not match AWBIR's teacher-positive eligibility set;
- harm prediction was mixed with FLOP/parameter normalisation intended for budget selection.

### G1b: V-C passed the pre-specified point-estimate gate, but the uncertainty is weaker than “4/4 superiority”

V-C combines within-layer semantic rank with mean layer Fisher scale. On the 96-channel holdout half, its point-estimate correlations exceed Fisher on all four harms:

| Harm | V-C rho | Fisher rho | Difference | Stratified bootstrap 95% interval for difference |
|---|---:|---:|---:|---:|
| AWBIR | 0.782 | 0.711 | +0.072 | [-0.017, 0.164] |
| Fine macro-F1 | 0.652 | 0.542 | +0.110 | [-0.002, 0.219] |
| Family macro-F1 | 0.634 | 0.571 | +0.063 | [-0.043, 0.174] |
| HSR | 0.438 | 0.393 | +0.045 | [-0.077, 0.176] |

Therefore the defensible shallow claim is:

> V-C has higher holdout point-estimate rank correlation than Fisher across all four outcomes, but the 96-group holdout is not large enough to establish a precise superiority margin under layer-stratified bootstrap uncertainty.

Do not write “V-C significantly beats Fisher 4/4” from the current shallow experiment.

### Why V-C works at shallow depth

The shallow architecture has a severe layer-allocation confound:

| Layer | Channels | Mean AWBIR harm | Mean fine-F1 harm | Mean family-F1 harm | Mean Fisher |
|---|---:|---:|---:|---:|---:|
| conv.0 | 64 | 0.192 | 0.077 | 0.063 | 1.437 |
| conv.3 | 128 | 0.034 | 0.002 | 0.001 | 0.068 |

V-C uses Fisher to allocate importance between layers and semantic rank to order channels within a layer. This is a coherent, interpretable two-stage saliency construction. It is not evidence that the raw first-order SBL quantity is globally valid.

### G2: the committed gate JSON is stale and invalid

`G2_screen_gate.json` is false with an empty details list because the gate code searches for method `r_sbl`, while the revised method is stored as `saber_v2`. This JSON must not be cited as evidence either for or against the method.

The more serious protocol issue is that the first screen matched nominal direct channel cost, not actual FLOPs. A nominal 55% budget produced 81.4%–96.7% realised dense-FLOP reduction depending on the method. Those cells were not matched-budget comparisons.

### Realised-FLOP calibration succeeds

The later binary-search prefix calibration fixes the systems problem:

- uncalibrated realised range: 43.8%–96.7%;
- maximum uncalibrated absolute target error: 41.7 percentage points;
- calibrated mean absolute error: 0.52 percentage points;
- calibrated maximum absolute error: 1.45 percentage points;
- all 15 method–budget cells are within 1.5 percentage points.

This is a strong methodological contribution. The code that generated the calibrated CSV is missing from the committed notebook, however, so the result is not yet independently reproducible. Notebook 17b in this recovery pack repairs that omission and freezes checkpoints.

### G3: HSBD gives a real calibration gain but violates the alert-risk objective

| Variant | Fine macro-F1 | Family macro-F1 | AWBIR | HSR | Benign→attack | ECE15 |
|---|---:|---:|---:|---:|---:|---:|
| CE+KD | 0.546 | 0.592 | 0.138 | 0.129 | 0.0146 | 0.0573 |
| HSBD fixed | 0.560 | 0.586 | 0.140 | 0.123 | 0.1272 | 0.0185 |
| HSBD dual | 0.563 | 0.589 | 0.140 | 0.124 | 0.1023 | 0.0277 |

HSBD fixed reduces ECE by roughly threefold and improves fine macro-F1/HSR, but it creates a large false-alert increase and does not improve AWBIR or family macro-F1.

The cause is not merely “bad hyperparameters.” The original benign-false-alert dual uses mean attack probability on benign examples. That surrogate is not the argmax boundary event that defines benign→attack. Notebook 24 replaces it with a binary logit-margin surrogate and adds an ASVG edge-margin constraint.

### G4: recovery equalises shallow selection methods

Without recovery, method choice matters greatly. Across the three budgets, raw AWBIR spreads are 0.423, 0.406, and 0.302. After one minimal recovery epoch, those spreads shrink to 0.039, 0.028, and 0.050. Full recovery also compresses differences.

This is not evidence that saliency is useless. It shows that **selection and recovery form a coupled system**. A paper that validates a channel score only through one-channel ablation and then evaluates a fully recovered pruning set can conflate two different causal stages.

### G5a: depth replication validates the AWBIR/family diagnostic, not a universal harm score

Across 576 deeper-CNN groups:

| Harm | V-C rho | Fisher rho | Difference | Stratified bootstrap 95% interval |
|---|---:|---:|---:|---:|
| AWBIR | 0.754 | 0.640 | +0.114 | [0.062, 0.166] |
| Fine macro-F1 | 0.061 | 0.062 | -0.001 | [-0.063, 0.061] |
| Family macro-F1 | 0.158 | 0.082 | +0.075 | [0.011, 0.139] |
| HSR | 0.103 | 0.053 | +0.050 | [-0.014, 0.115] |

The strong, replicated conclusion is:

> V-C predicts single-channel ASVG boundary inversions, and to a lesser extent family-level harm, more directly than Fisher in the deeper architecture. It is not a general predictor of every damage measure.

### G5b: static V-C selection fails after recovery

At approximately 40% realised FLOP reduction with minimal recovery:

| Method | AWBIR (lower) | Fine macro-F1 | Family macro-F1 |
|---|---:|---:|---:|
| Taylor | **0.034** | **0.553** | **0.628** |
| Magnitude | 0.096 | 0.529 | 0.599 |
| V-C | 0.115 | 0.497 | 0.558 |
| Random | 0.139 | 0.491 | 0.554 |
| Fisher | 0.180 | 0.506 | 0.566 |

V-C is therefore not currently a competitive static pruning selector at depth.

The ranking dissociates sharply from channel-level validity. Across V-C, Fisher, Taylor, and magnitude, channel-level AWBIR validity versus minimal-recovery deployed AWBIR has Spearman -0.60 (n=4; descriptive only). Fisher is a strong channel predictor but the worst recovered pruner; magnitude has negative channel validity yet is near-best after recovery; Taylor is the best recovered method.

## 2. Strong positive result omitted from the proposed outline: ASVG predicts absorber destinations

The validation-built ASVG was compared with the top absorber observed later under anchor prune80 on the diagnostic test partition:

- Top-1 recovery: 8/22 = 36.4%;
- Top-2 recovery: 11/22 = 50.0%;
- Top-3 recovery: 15/22 = 68.2%;
- mean reciprocal rank: 0.492;
- absorber-fraction-weighted Top-3 recovery: 75.3%;
- among eight sources whose observed absorber took at least 75% of traffic, Top-3 recovery is 7/8 = 87.5%.

A naive random Top-3 rate among 33 competing classes is approximately 9.1%. A simple binomial comparison gives p≈2.2e-11, although source cases are not independent and the diagnostic test set has historical prior use. This is still compelling external validity for the graph object.

This should become a principal result, not a footnote.

## 3. The proposed “channel validity does not guarantee deployed safety” thesis needs one correction

For the 15 shallow raw selected sets, the sum of measured one-channel harms predicts joint raw set damage strongly:

- summed AWBIR harm versus raw-set AWBIR: rho = 0.843;
- summed AWBIR harm versus raw-set HSR: rho = 0.921;
- summed AWBIR harm versus raw-set fine-F1 loss: rho = 0.932;
- summed AWBIR harm versus raw-set family-F1 loss: rho = 0.921.

So the current data do **not** support a blanket statement that single-channel causal validity is unrelated to set damage. The sharper, evidence-compatible thesis is:

> Single-channel causal harm can predict the immediate raw damage of a pruning set, but it does not by itself identify the best **recovered** deployment. Recovery dynamics and architecture depth can reorder or erase the static-selection advantage.

Deep raw-set metrics were not saved, so the current branch cannot distinguish deep set interaction from recovery-mediated reordering. Notebook 22 is designed specifically to resolve that ambiguity.

## 4. Recommended paper identity

### If the new method rescue fails

**Recommended title:**

> **From Channel Harm to Recovered Pruning Sets: An Alert-Semantic Causal Benchmark for IoT Intrusion Detection**

Alternative:

> **When Channel-Level Saliency Fails to Transfer to Recovered Pruning Sets: An Alert-Semantic Benchmark for IoT IDS Compression**

The paper would contribute:

1. ASVG and its absorber-recovery validation;
2. the 768-group causal semantic-harm benchmark;
3. V-C as a replicated AWBIR-oriented diagnostic, with uncertainty stated honestly;
4. realised-FLOP calibration and the failure of nominal direct-cost matching;
5. a set-interaction/recovery-reordering benchmark;
6. a regime map showing when selection or recovery dominates.

This is a benchmark/methodology paper, not a claim that V-C is the best pruning algorithm.

### If Alert-Semantic Lookahead or HSBD-v2 passes

A method paper remains possible, but the new method must be the load-bearing contribution:

> **Alert-Semantic Set-Conditional Pruning for Reliable IoT Intrusion Detection**

The paper must compare against static Taylor/Fisher/V-C at matched realised FLOPs and show multi-seed, test, and external-dataset superiority. Generic lookahead/oracle pruning is established in prior literature, so novelty must be restricted to the directed alert-semantic objective and recovery-aware evaluation.

## 5. Required experiments before writing Results

### Mandatory

1. **Notebook 17b:** reproduce realised-FLOP calibration and freeze 15 shallow checkpoints.
2. **Notebook 20b:** freeze/reconcile the ten depth students.
3. **Notebook 21:** one-shot SABER-held-out test evaluation of all frozen models.
4. **Notebook 22:** random-set additive/interaction benchmark with a discovery/confirmation split and minimal-recovery subset.
5. **At least three independent seeds** for the final headline architecture–method cells.

### Method-rescue experiments

6. **Notebook 23:** V-C-shortlisted, alert-semantic lookahead set selection.
7. **Notebook 24:** boundary-aligned HSBD-v2 constraints.
8. If either passes, run a three-seed validation screen and then a five-seed confirmation.

### External validity

9. Repeat the compact score/set/recovery comparison on at least one genuinely fresh dataset arm. Because the CICIoT2023 test split was used in the earlier diagnostic study, describe Notebook 21 as SABER-held-out rather than globally untouched. A fresh Edge-IIoTset or CICIoMT2024 confirmation would materially improve the paper.

## 6. Claims that are currently safe

- The ASVG recovers a large fraction of later high-mass absorber destinations.
- V-C's two-stage layer/channel design predicts single-channel AWBIR harm strongly and replicates at depth.
- The original R-SBL formulation failed.
- Direct-cost budgets do not ensure matched realised FLOPs under coupled channel surgery.
- Minimal/full recovery can substantially compress and reorder shallow method differences.
- Static V-C is not the best recovered pruner in the depth probe; Taylor is.
- The original HSBD dual's probability surrogate does not control the actual benign false-alert rate.

## 7. Claims that are not safe yet

- “V-C significantly beats Fisher on all four harms.”
- “Channel-level validity never predicts set damage.”
- “V-C is a superior pruning method.”
- “The test set is completely untouched.”
- “Deployed safety” without physical hardware, multi-seed confirmation, and an external dataset.
- “Four preregistered variants” unless a timestamped commit or frozen record predates computation; otherwise say “pre-specified variants.”

## Bottom line

G5 is disappointing only if the project is forced to remain the original SABER method paper. Scientifically, it exposed a more interesting problem: the field commonly treats channel saliency, set selection, recovery, and deployment as though they were one object. Your results show they are not.

The strongest path is now dual:

- preserve the benchmark/negative-methodology paper regardless of method rescue;
- run a focused set-conditional selector and a boundary-aligned recovery objective to determine whether a genuine new algorithm can still be built on top of the benchmark.
