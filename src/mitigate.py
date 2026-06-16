"""
src/mitigate.py — closing payoff (droppable per D3). Predictor-guided selective
protection of flagged classes; FORM set by the crux result.

  information survived -> decision-layer: per-class thresholds / re-fit head /
                          per-class temperature+bias (scalar temp can't move argmax).
  information lost     -> compression-time: rare-class-aware pruning saliency /
                          rare-class-weighted distillation / mixed precision.

v2.1 STRETCH: a class-conditional conformal / risk-control CERTIFICATE on flagged
classes (distribution-free per-class FNR <= alpha). Feasibility gate: ~1/alpha
calibration instances for the rarest classes; if unmet, DOWNGRADE to empirical
risk control and report the certifiable boundary.

Baselines to beat: naive compression, scalar temperature, per-class thresholds,
focal/class-weighted finetune, SMOTE/CTGAN, rare-class-aware KD, uncompressed ref.
Report the trade-off FRONTIER (recall vs compression / benign-FPR / macro-F1), CIs.
"""
from __future__ import annotations
import numpy as np


def per_class_threshold(probs, y_true, flagged, cost_ratio):
    """Cost-sensitive / Neyman-Pearson per-class thresholds (decision-layer fix)."""
    raise NotImplementedError("Optimise per-class thresholds with the security-asymmetric default.")


def conformal_recall_certificate(scores, y_true, flagged, alpha):
    """Class-conditional conformal / CRC: certify per-class FNR <= alpha, or return
    the empirical-risk-control fallback for classes that can't be certified."""
    raise NotImplementedError("Implement class-conditional conformal + feasibility gate.")
