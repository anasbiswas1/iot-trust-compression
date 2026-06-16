# Research Plan — Trust-Preserving Compression for IoT Intrusion Detection

**A framework to measure, explain, predict, and mitigate per-class trustworthiness collapse in compressed network intrusion detectors.**

Author: Md Anas Biswas, School of Computing, University of Portsmouth (up2082724@myport.ac.uk). Sole-authored.
Target: Q1 journal — Expert Systems with Applications (ESWA) or Computers & Security.
Status: **v2.0** (restructured after the lane-scoped novelty review). Q1 novelty bar is the priority, not a submission date.

> **What changed from v1.0.** A literature review scoped strictly to the network IoT-IDS / tabular-security lane established that the three novel stages are *not* co-equal. The predictive diagnostic is the clean open gap; the mechanism is partially open and is most valuable as the *justification* for the diagnostic's features; per-class trustworthiness measurement under compression is itself a gap; rare-class mitigation is crowded. This version re-ranks the contribution accordingly and positions against the specific competitors the review surfaced. (v1.0 retained as `PLAN_v1_archived.md`.)

---

## 1. Contribution hierarchy (the core change in v2.0)

The narrative arc is still **measure → explain → predict → mitigate**, but the four stages carry very different contribution weight. Lead the abstract and introduction with the diagnostic; present the body in narrative order.

**PRIMARY — PREDICT (the headline).** A pre-deployment, per-class diagnostic that forecasts *which* attack classes will lose recall, calibration, or explanation-trust (stability + faithfulness) under compression, computed from the **baseline (uncompressed) model and data alone, before any compression is applied.** The review found no IoT-IDS / tabular-security paper that does this, and — as a *deployable per-class forecaster* — no general-ML equivalent either. **v2.1 upgrade — the diagnostic is mechanism-grounded, not correlational.** Its features are *derived from a stated theory of capacity allocation* (neural collapse / Minority Collapse, see Stage 2 and §2), so the claim is "we predict per-class collapse from a theory of why the tail is fragile," not "we found properties that happen to correlate." The thing that *earns* the causal framing is **cross-architecture generalisation**: a theory-derived predictor should transfer across MLP/CNN/Transformer; a curve-fit will not. Existing work that touches the idea is post-hoc by construction (Hooker's PIE/CIE require the compressed model to already exist) or backward-looking statistical regression on collected runs (Paganini). This is the reusable artefact a practitioner runs once on the baseline model to decide what is safe to compress.

**BACKBONE — EXPLAIN (supports the headline, not a separate claim).** The mechanism — information-loss vs. decision-rule-loss dissociation via linear probes, per-class effective-rank change, per-class margin/boundary geometry shift, and dissociation of pruning vs. quantisation vs. distillation by their per-class mechanism — is novel in the IoT-IDS lane. **v2.1 — it now has a named theoretical anchor.** Neural collapse, specifically *Minority Collapse* (Fang et al., PNAS 2021; §2), supplies the capacity-allocation account of *why* the tail is fragile at baseline: under imbalanced training, minority-class classifier vectors lose angular separation and collapse toward each other, sitting closest to the decision boundary with the least margin — so they fail first when compression removes representational capacity. This is the "why rarity → collapse" theory; the crux experiment and per-family dissection are then "what compression does to that already-fragile tail." Critically, the theory is the **scientific justification for the diagnostic's feature set**: the per-class neural-collapse geometry (ETF deviation, minority-collapse angle) becomes a *theory-derived predictor feature*, which is what makes the predictor principled rather than a fitted black box and tightens the measure→explain→predict chain.

**PROBLEM STATEMENT — MEASURE (a genuine but bounded gap).** Two distinct things sit under "measure": (a) *replicating that collapse happens at all* is Hooker-level prior art and is pure validation substrate — do not claim novelty for it; (b) *quantifying per-class calibration collapse and per-class explanation-trust change under compression — explanation stability and faithfulness measured and reported separately, drift distinguished from genuine unfaithfulness* — is itself understudied in IoT-IDS (the lane's compression papers report aggregate accuracy/F1/latency), so the per-class **trust** measurement is a legitimate, if secondary, contribution and is the quantified problem the rest of the paper addresses.

**CLOSING PAYOFF — MITIGATE (not the core novelty; droppable; now with a certified option).** Rare-class handling in IDS is crowded (focal loss, SMOTE/CTGAN, cost-sensitive learning, rare-class-aware distillation all exist). Do not lead with mitigation. The defensible angle is **predictor-guided selective protection**: use the diagnostic to target protection only at the classes it flags, with the *form* of protection chosen by the crux experiment (decision-layer recalibration if information survived; a compression-time intervention if information was lost). This closes the loop with the headline. **v2.1 stretch upgrade — a per-class recall *certificate*.** A class-conditional conformal / conformal-risk-control guarantee (Angelopoulos et al. 2022; §2) on the flagged classes would upgrade mitigation from "heuristic that beat baselines with CIs" to a *distribution-free guarantee* that a protected class will not silently drop below a recall threshold post-compression — strong for Computers & Security, and the formal backing for the contract's false-negative envelope. **Stretch with a pre-registered fallback:** the rarest classes may lack the calibration samples a tight distribution-free certificate needs; if so, downgrade to an empirical risk-control claim (observed per-class risk with CIs, no formal certificate). Per D3, the novel-method claim is still dropped if a plain focal-loss / class-weighted finetune closes the gap.

---

## 2. Positioning against prior art (write this into the introduction)

**Foundational, out-of-lane (cite as background, out-position cleanly):**

- Hooker et al., *What Do Compressed Deep Neural Networks Forget?* (arXiv:1911.05248) and *Characterising Bias in Compressed Models* (arXiv:2010.03058) — PIEs/CIEs; compression disproportionately harms the long-tail. **Post-hoc, vision, accuracy-only.** This paper is per-class (not per-exemplar), trust-based (not accuracy-only), predictive-before-compression (not post-hoc), and security-asymmetric (the forgotten tail is the attack class).

**The dangerous explain/predict precedents (engage head-on):**

- **Tran, Fioretto, Kim & Naidu, *Pruning has a disparate impact on model accuracy* (NeurIPS 2022, arXiv:2205.13574)** — the closest mechanistic account: ties disparate pruning harm to baseline gradient norm and distance to decision boundary. It is image/group-fairness, *causal-explanatory not deployable-predictive*, and not IDS. **This is the most likely "isn't your predictor just a transfer of a known idea?" objection.** Defence: (i) it is not packaged or validated as a pre-compression per-class screening tool; (ii) this paper's predictor uses IDS-specific features (per-attack-class attribution concentration, per-attack margin, flow-feature separability, effective rank) beyond margin/gradient-norm; (iii) it is validated to generalise *across compression families* (pruning + int8/float16 + distillation), not fit to one. The Tran & Fioretto margin/gradient-norm-only predictor is a **required baseline** the diagnostic must beat (see Stage 3).
- **Paganini, *Prune Responsibly* (arXiv:2009.09936, 2020)** — retrospective statistical model of post-pruning per-class accuracy on baseline covariates (class imbalance + complexity). Closest to a statistical predictor, but **backward-looking regression on collected pruning runs, vision, not a deployable pre-compression diagnostic.**
- **Good, Lin, Yu et al., *Recall Distortion in Neural Network Pruning* (arXiv:2206.02976, NeurIPS 2022)** — formalises a per-class recall "intensification effect" under pruning. **Characterisation, not forecasting**; directly relevant framing for the recall-collapse target.

**The theoretical anchor for the mechanistic predictor (cite as foundation, not competitor):**

- **Fang, He, Long & Su, *Layer-Peeled Model: Minority Collapse in imbalanced training* (PNAS 118, 2021)** — establishes the simplex-ETF geometry of neural collapse and *Minority Collapse*: beyond an imbalance threshold the classifiers of minority classes collapse onto each other. This is the capacity-allocation theory the PREDICT diagnostic is *derived* from. It is vision/theory, not IDS, not compression-prediction — so it is the ancestor the paper builds on, and the novelty is operationalising it into a pre-deployment per-class compression-harm forecaster for IDS. Engage explicitly so the mechanism reads as theory-grounded, not invented.

**The certificate tooling (cite as the method the certified-mitigation option uses):**

- **Angelopoulos, Bates et al., *Conformal Risk Control* (arXiv:2208.02814)** and class-conditional conformal prediction (Ding, Tibshirani et al.) — distribution-free per-class risk/FNR control. Domain-agnostic, not IDS, not compression. The novelty is delivering a per-class recall certificate on a *compressed* IDS's flagged classes.

**The closest structural match — the framework-scoop check (cite head-on to differentiate):**

- **Hong et al., *Decoding Compressed Trust* (arXiv:2403.15447, ICML 2024)** — the first thorough evaluation of trustworthiness under compression across multiple dimensions, finding 4-bit quantisation roughly preserves trust while pruning degrades it even at 50% sparsity. It is the nearest *structural* ancestor (trustworthiness-under-compression, multi-dimension) but is **LLM-domain, not per-class, with no pre-deployment predictor and no mechanism dissociation.** This is the single most important framework-level differentiation: the moat is the *integration* (measure→explain→predict→mitigate, per-class, in IDS), which no surveyed paper assembles.

**The dangerous in-lane competitor (cite head-on, differentiate sharply):**

| Paper (lane: IoT-IDS) | What it does | What it does NOT do (our room) |
|---|---|---|
| **SHAP-Guided Pruning + Kronecker-distilled nets** (arXiv:2512.19488, TON_IoT) — *most dangerous adjacent work* | SHAP feature pruning + KD + int8 for lightweight IoT-IDS | No prediction; aggregate macro-F1/latency only; **concedes rare-class macro-F1 drop**; no calibration, no attribution-drift, no mechanism |
| **KD-based lightweight IDS** (Cluster Computing 2025, DOI 10.1007/s10586-025-05597-2) | Ladder-structured KD + adaptive focal loss for minority classes | Mitigation-only; no prediction, no trust metrics, no mechanism — *this is a mitigation baseline, not a competitor to the headline* |
| Adaptive DNN pruning for embedded IDS (arXiv:2505.14592); Quantized AE-IDS (Cybersecurity 2023); dynamic-quant BiLSTM IDS (PeerJ CS 2023) | Pruning/quantisation for edge efficiency | Aggregate metrics only; no per-class trust, prediction, or mechanism |
| HED-ID explainable edge IDS (Sci. Reports 2025); XAI-IDS review (Sensors 26(2):363, 2026) | SHAP + edge framing; PRISMA review of the accuracy–efficiency–explainability trilemma | No actual compression studied under per-class trust; review proposes no method (useful framing citation) |

**The one-line reviewer rebuttal on imbalance** (keep from v1.0): *this is latent fragility unmasked by compression — the baseline model handles the rare class adequately, and compression is what breaks it* — distinct from standard imbalance, where the baseline already fails. The imbalance baselines in Stage 4 demonstrate this rather than asserting it.

No surveyed paper combines explain + predict + mitigate, and none does pre-compression per-class forecasting. Make the headline claim "to our knowledge"; no "first to."

---

## 3. Decision log (frozen; flip a single entry if revisited)

- **D1 — Crux experiment is the spine.** The information-loss-vs-decision-rule-loss probe runs *before* the mitigation is designed, because its outcome dictates the form of protection. Adopted.
- **D2 — Diagnostic target.** Continuous Δrecall is the primary target (report R² and a predicted-vs-actual calibration plot on held-out classes); a binary collapse label is secondary, for the ship/don't-ship framing. Both defined in the prereg before any compressed result is inspected. Adopted.
- **D3 — Novel-method claim is droppable.** If a plain focal-loss / class-weighted finetune closes the gap, the novel mitigation claim is dropped; the paper stands on predict + explain + per-class trust measurement. Committed in advance. Adopted.
- **D4 — Rarity-vs-separability subsampling ablation is IN.** Strongest single defence of the mechanism and of the predictor's "separability matters beyond rarity" claim. Adopted.
- **D5 — No human evaluation.** Trust is operationalised mathematically (calibration; faithfulness vs. anchor; deletion/AOPC), not perceptually; flow features are not human-interpretable as images are. Fully automated; stated as a strength. Adopted.
- **D6 — No physical devices.** All trust metrics are hardware-independent numerical properties of the model; "edge-scale" is satisfied by sizing models (params/MB/sparsity reported), not deployment. No latency/energy claims. Laptop/desktop + Colab. Adopted.

---

## 4. Scope boundaries (do not cross)

- **Lane:** network IoT-IDS only. The IoMT/medical paper (CICIoMT2024, selective prediction, "Trustworthy by How Much?") is a SEPARATE project. No medical datasets, no selective prediction here.
- **Conference paper:** cited as prior work, no verbatim text, majority-new experiments.
- **No human eval (D5), no physical hardware (D6).**

---

## 5. Validation substrate

**Architectures (size- and accuracy-matched, edge-scale, tens of thousands to low-hundred-thousands of params):**

- MLP (dense) — anchor.
- 1D-CNN (local/convolutional) — **build first; lowest risk; carries the Stage 1 gate and the crux experiment.**
- FT-Transformer (attention) — primary third arm. **Pre-registered fallback:** within ±0.02 macro-F1 of the MLP on *both* core datasets within a fixed tuning budget, else GRU (noting a GRU over feature-as-timestep vectors is a questionable inductive bias for non-sequential flow data).

Match on **macro-F1, not accuracy** (accuracy hiding collapse is the premise), within a pre-registered tolerance. Report per-class baseline recall so "architecture X collapses more" cannot secretly be "architecture X started weaker."

**Datasets:**

- CICIoT2023 (core) — Neto et al., *Sensors* 2023, 23(13):5941; 33 attacks over 105 real IoT devices, grouped to 7 categories (+benign), ~46.6M flows, 47 features — a concrete, highly imbalanced multi-class testbed ideal for per-class forecasting.
- TON_IoT network subset (core) — drop identity fields (addresses, ports, timestamps).
- Bot-IoT (optional stretch / robustness + the leave-one-dataset-out generalisation test) — inverted (benign-rare) imbalance tests whether collapse follows class *rarity* regardless of which class is rare.

**Deduplication and identity-field removal are mechanism-critical, not hygiene** (leaked identifiers create spurious separability that compression "loses," masquerading as capacity loss and invalidating Stage 2). Audit and document before modelling.

**Compression matrix:** M0 (full-precision baseline) | prune50 | prune80 | distillation | int8 | float16. The mix of post-training quantisation (no retraining: int8, float16) and finetune/retrain-based methods (prune, distillation) is load-bearing — it dissociates information loss from optimisation-during-finetuning (Stage 2), and the predictor must generalise across all of them (Stage 3).

**Evaluation protocol — split choice is an integrity requirement, not optional breadth.** The **primary split is grouped / source-aware** (no device/session/capture spans the train/test boundary) and it governs MEASURE, the crux experiment, and PREDICT — not just one stage. Random splits leak via near-duplicate flows, which inflates baseline per-class properties *and* makes "collapse under compression" partly loss-of-memorised-duplicates, so a random-only design could manufacture the headline finding. A **random split is retained as an explicit reference** so the leakage gap (how much random overstates baseline trust) can be reported as a *finding* that demonstrates the latent-fragility premise. Temporal split is a robustness layer (Stage 5); leave-one-attack-family-out is optional and separately framed (it tests novelty detection, a different question). The grouping variable, and the rule for fine-grained types that don't survive grouping, are fixed in the prereg before any metric. (Full spec: `anchor_preregistration.json` → `evaluation_protocol`.)

---

## 6. The crux experiment (the spine — D1)

For each class, freeze the compressed model's penultimate representation and train a **linear probe** (one-vs-rest) to recover that class; compare its AUC to the same probe on the baseline (M0) representation.

- **Probe survives (near-baseline AUC):** information survived; recall collapse is a **decision-layer artifact** (logit rescaling / margin shift across the argmax boundary). Protection will be near-free (decision-layer recalibration / re-thresholding).
- **Probe AUC collapses:** information is **genuinely lost**; true capacity loss. Protection must touch the compression procedure.

This makes the mechanism falsifiable, tells the diagnostic whether to weight information-theoretic vs. margin/geometry features, and sets the *form* of the Stage 4 mitigation. Runs at step 3 of the sequence.

---

## 7. Stages

### Stage 0 — Scaffolding (before any experiment)
- Repo + Drive structure; `.gitignore` for data/binaries.
- `config/`: seeds, locked stratified splits (scaler fit on train only), per-architecture hyperparameters.
- Dedup + identity-field audit (mechanism-critical).
- `logs/anchor_preregistration.json`: freeze full-precision anchors, seeds, macro-F1 matching tolerance, the Δrecall/collapse target definitions (D2), the diagnostic's candidate feature list, the diagnostic baselines (frequency-only and Tran & Fioretto margin/gradient-norm-only), and the mitigation baseline list — **before any trust metric is computed.**
- Per-class metric code: adaptive equal-mass ECE (10/15/20 bins), signed per-class over-confidence gap, explanation **stability** (per-instance Spearman vs. anchor) against an explainer-noise floor + a **redundancy/retraining null band** (two M0 seeds, no compression), **faithfulness** via top-k deletion AOPC vs. random-deletion (the sole faithfulness adjudicator), attribution-drift **stratified by prediction-change**, **perturbation-based attributions (KernelSHAP) across the whole compression matrix** (DeepSHAP gradients lie on int8), bootstrap 95% CIs (B=1000). (Four-case decomposition and the unfaithfulness decision rule: `anchor_preregistration.json` → `metrics.explanation_trust_decomposition`.)

### Stage 1 — MEASURE (mostly substrate; per-class trust is the secondary contribution)
- Start with **1D-CNN × both core datasets × full compression matrix**, fairness controls applied (macro-F1 matched at baseline; all arms in the same compact param band).
- Output per class: calibration gap (ECE), explanation **stability** drift and **faithfulness** (separated — drift is not unfaithfulness until the decision rule says so), recall collapse — all with bootstrap CIs, all under the primary grouped split.
- **Seed null band:** report seed-to-seed variability for recall *and* attribution, so every "collapse" claim is visibly outside the null.
- **Gate:** if the effect does not replicate cleanly here, stop and diagnose before scaling to three architectures. Non-replication is itself a reportable result.

### Stage 2 — EXPLAIN (the mechanistic backbone that justifies the predictor)
Run as a hypothesis tournament; the surviving mechanism defines the predictor's feature set.
1. **Information loss** — probe AUC collapses (crux experiment).
2. **Decision-rule shift** — probe survives but per-class signed margin collapses; recoverable by re-thresholding.
3. **Optimisation-during-finetuning** — distinguished by compression family: post-training int8/float16 involve no retraining, so collapse there implicates the representation/decision geometry, not finetuning dynamics.
4. **Rarity-vs-separability confound (D4)** — subsample a frequent class down to a rare class's count; collapse → rarity causal; robust → separability causal.

Per-family proximate causes (more precise, more defensible than one monolithic "capacity" claim): test whether magnitude pruning's removed units over-represent high-rare-contribution units (rank units by baseline ablation drop / gradient×activation on the rare class); frame quantisation as a margin-resolution argument tied to measurement #2. Tooling: linear-probe recoverability, per-class effective rank (e.g. singular-value entropy), per-class margin distributions — established outside IDS, novel here.

**Neural-collapse geometry (the theoretical layer, v2.1).** Measure each class's penultimate geometry against the ideal simplex ETF: per-class angular separation between classifier/mean vectors and the degree of *Minority Collapse* (how far rare-class vectors have collapsed toward each other) at baseline, and how it shifts under each compression level. This is the capacity-allocation account of *why* the tail is fragile (rare classes start with the least angular margin), and it yields the theory-derived predictor feature used in Stage 3. The test that earns the *causal* claim: this geometry should predict per-class collapse **consistently across all three architectures** — if it does, the mechanism generalises; if it predicts only within one architecture, it is an architecture-specific correlation and the causal framing is withdrawn (see §9).

**Measurement caution:** class-conditional effective rank / separability is noisy at rare-class sample counts; every such measure needs bootstrap CIs and a same-n subsampled frequent-class control, or it measures "few samples → noisy estimate."

Honesty check: distinguish correlation from cause; state what the evidence does and does not establish; report any architecture/class where the mechanism does not hold.

### Stage 3 — PREDICT (the headline contribution)
Goal: from M0 alone, rank attack classes by predicted post-compression collapse, before compressing.
- **All predictors computable from the baseline model only:** per-class sample count, per-class baseline margin, baseline linear separability in representation space, baseline attribution concentration, per-class effective-rank contribution, and **(v2.1) per-class neural-collapse geometry** (ETF deviation / minority-collapse angle). The mechanism in Stage 2 *derives* each — the predictor is theory-grounded, not a feature grab-bag.
- **Tiny-n discipline (main over-claim risk):** ~8 + ~10 class-rows (more at 34-type granularity, see prereg). Keep the diagnostic deliberately simple — 1–2 features, a monotone/threshold or low-capacity rule, not a learned multivariate black box.
- **Required baselines the diagnostic must beat:** (a) naive "smallest classes collapse" (frequency-only); (b) the **Tran & Fioretto margin/gradient-norm-only predictor**. If the IDS-feature diagnostic does not beat both, the honest finding is "rarity/margin necessary but not sufficient; [feature] explains the residual."
- **Evaluation:** rank correlation between predicted and actual per-class collapse (Spearman/Kendall, bootstrap CIs); precision@k of correctly flagged collapsing classes (the practitioner operating point); R² + predicted-vs-actual calibration plot for continuous Δrecall.
- **Generalisation — the test that earns the causal claim (v2.1).** Must hold *across architectures* (the theory-derived predictor transfers MLP→CNN→Transformer — this is what licenses the mechanistic, not merely correlational, framing), *across compression families* (fit on some, predict others), and *across datasets* (leave-one-dataset-out, including Bot-IoT's inverted structure). Report failure cases. If it generalises across families/datasets but **not** across architectures, the causal claim is withdrawn and the predictor is reported as a strong correlational tool with the mechanism as an interpretive lens (§9) — still publishable.

### Stage 4 — MITIGATE (closing payoff: predictor-guided selective protection)
Close the loop: apply protection **only to the classes the Stage 3 diagnostic flags**, with the *form* set by the crux result.
- **Information survived →** decision-layer protection (near-free): per-class threshold optimisation (cost-sensitive / Neyman-Pearson), a small re-fit head on the frozen compressed representation, or per-class temperature+bias. Show that scalar temperature scaling fixes calibration but cannot move argmax decisions, so it structurally cannot recover recall.
- **Information lost →** compression-time intervention motivated by the mechanism: rare-class-aware pruning saliency (protect high-rare-contribution units), distillation loss up-weighting rare-class logit/boundary matching, or mixed-precision keeping resolution on rare-critical units.

**Pre-registered baselines:** naive compression (floor); scalar temperature scaling; per-class threshold optimisation (strong cheap baseline); focal-loss / class-weighted finetune; SMOTE/CTGAN-rebalanced compressed model; rare-class-aware KD (Cluster Computing 2025 style); uncompressed model (upper reference).

**Report the trade-off frontier, not a point:** recall recovered vs. (a) compression ratio sacrificed, (b) benign FPR incurred, (c) macro-F1 — with bootstrap CIs on the deltas. **Bound the benign blind spot explicitly:** state where recall→0 is not closed and by how much. Honour D3.

**Certified option (v2.1 stretch).** On the flagged-and-protected classes, attempt a class-conditional conformal / conformal-risk-control certificate: a distribution-free guarantee that per-class FNR (1 − recall) stays below α with high probability post-compression. This is the formal version of the contract's false-negative envelope and upgrades the strongest version of the paper from "beats baselines" to "certified." **Pre-registered feasibility gate and fallback:** a distribution-free per-class certificate needs roughly 1/α calibration instances for the rarest classes even to define the quantile; the rarest attack types may not clear this. If they do not, downgrade to an empirical risk-control claim (observed per-class risk + CIs, explicitly *not* a formal certificate) and report which classes could and could not be certified — the boundary is itself a finding.

### Stage 5 — Robustness / optional breadth (only if 1–4 are done)
- Bot-IoT (inverted imbalance) as robustness + the leave-one-dataset-out generalisation test for the diagnostic.
- Temporal split as a deployment-realistic drift check (the grouped/source-aware split is no longer here — it is the *primary* regime, see §5). Leave-one-attack-family-out optional, separately framed as novelty detection.

### Stage 6 — Write-up, preprint, submission
- **Lead the abstract/intro with the diagnostic;** present the body in measure→explain→predict→mitigate order. Position the mechanism as the predictor's justification and breadth as substrate.
- Reproducibility statement matching the repo exactly; state the fully-automated, no-human, no-hardware design as a strength.
- arXiv cs.CR preprint (endorsement already held) when ready, to stake priority on the diagnostic.
- **Venue:** the diagnostic-as-usable-tool framing leans ESWA; the security-asymmetric-cost framing leans Computers & Security. Decide after Stages 3–4, on which of predict/mitigate is strongest.

---

## 8. Experiment sequence (with gate and branch point)

1. Stage 0 — scaffolding, dedup/leakage audit, freeze prereg (primary grouped split, grouping variable, diagnostic baselines and feature list, the explanation-trust decomposition).
2. Stage 1 — MEASURE on 1D-CNN × both datasets × full matrix. **Gate:** must replicate outside the seed null.
3. **Crux experiment** (probe + margin) on the 1D-CNN. **Branches the project** into information-loss vs decision-layer worlds — run before building the mitigation.
4. Stage 2 — complete the hypothesis tournament across all three architectures (FT-Transformer fallback checkpoint here); finalise the predictor's feature set from what the mechanism supports.
5. Stage 3 — diagnostic: M0 predictors, simple rule, beat frequency-only AND Tran & Fioretto baselines, cross-compression + leave-one-dataset-out validation incl. Bot-IoT, report failures.
6. Stage 4 — predictor-guided selective protection pointed to by step 3, full baseline set, trade-off frontier with CIs.
7. Stage 5 — Bot-IoT robustness; temporal-split drift check.
8. Stage 6 — write-up, arXiv preprint, ESWA/C&S.

---

## 9. Strategic contingencies (decided in advance, from the review)

- **If a pre-compression per-class predictor surfaces in the IoT-IDS lane during build/review** → pivot the lead to the explain mechanism + the pruning/quantisation/distillation dissociation, which the review found untouched in this lane; demote predict to validation of the mechanism.
- **If the diagnostic underperforms the naive frequency baseline** → fall back to MEASURE (per-class trust) + EXPLAIN as the primary contribution; report the predictor honestly as a negative/bounded result ("rarity dominates; richer features add little"). This is publishable if clearly framed.
- **If the novel mitigation does not beat focal-loss finetune** → invoke D3, drop the method claim, keep predictor-guided *targeting* as the framing (you still showed where to apply known fixes), and lean on predict + explain + measure.
- **If the neural-collapse mechanism does not generalise across architectures (v2.1)** → withdraw the *causal* framing of PREDICT; report the predictor as a strong correlational tool and present the mechanism as an interpretive lens. The headline survives, downgraded from "mechanistic" to "empirical"; still publishable, and pre-committing this stops the theoretical claim being defended past its evidence.
- **If a per-class conformal certificate is infeasible for the rarest classes (v2.1)** → downgrade the certified-mitigation option to empirical risk control (observed per-class risk + CIs), report the certifiable/non-certifiable boundary, and make no formal-guarantee claim. The certificate was always a stretch in the *secondary*, droppable stage, so losing it costs a stretch goal, not the paper.
- **Treat 2026 arXiv competitors as moving targets** — re-run the lane-scoped search immediately before submission; the closest items are unreviewed preprints whose status may change.

---

## 10. Non-negotiables

- Q1 novelty over speed.
- Honest, bounded claims; report nulls, failure cases, and where the mitigation does not help.
- Large-n test sets: effect sizes + bootstrap 95% CIs + retraining null band, not p-values.
- No "first to" claims; "to our knowledge" only. No deployability claims beyond what is measured.
- Network IoT-IDS lane only; medical/IoMT stays separate.
- **The mechanistic claim raises the bar (v2.1):** grounding PREDICT in a capacity-allocation theory means a reviewer will hold the theory to *generalising*, not just correlating. Stage 2's neural-collapse measurement and Stage 3's cross-architecture test are therefore load-bearing for the causal claim — if the mechanism does not hold across all three architectures, fall back to the correlational framing rather than over-defend.
- Reproducibility from day one: locked splits, seeded stochastic components, anchors pre-registered before metrics, public repo (code/notebooks/tables/logs/figures; data and binaries in Drive).
