"""
src/metrics.py — per-class trust metrics with the honesty controls baked in.

Implemented (data-independent): bootstrap CI, adaptive ECE, per-class recall,
AOPC. Scaffolded (need model internals/SHAP): stability, the four-case
explanation-trust decomposition, prediction-change stratification.

KEY DISCIPLINE (frozen): explanation STABILITY (Spearman vs anchor) is NOT
faithfulness; FAITHFULNESS is adjudicated by deletion/AOPC ALONE; attribution
drift is not unfaithfulness until the decision rule says so. Attributions use
KernelSHAP across the WHOLE compression matrix (DeepSHAP gradients lie on int8).
"""
from __future__ import annotations
from typing import Callable, Optional
import numpy as np

from .config import CFG


# --- inference: effect sizes + bootstrap CIs, NOT p-values -----------
def bootstrap_ci(values: np.ndarray, statistic: Callable = np.mean,
                 B: Optional[int] = None, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap 95% CI. Returns (point, lo, hi)."""
    B = B or CFG["metrics"]["bootstrap_B"]
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    stats = np.array([statistic(values[rng.integers(0, n, n)]) for _ in range(B)])
    return float(statistic(values)), float(np.percentile(stats, 100 * alpha / 2)), \
        float(np.percentile(stats, 100 * (1 - alpha / 2)))


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for c in np.unique(y_true):
        m = y_true == c
        out[int(c)] = float((y_pred[m] == c).mean()) if m.any() else float("nan")
    return out


# --- calibration: adaptive equal-mass ECE ----------------------------
def adaptive_ece(probs: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Equal-MASS binning (each bin ~equal count), reported at 10/15/20 bins."""
    probs, correct = np.asarray(probs), np.asarray(correct).astype(float)
    order = np.argsort(probs)
    probs, correct = probs[order], correct[order]
    bins = np.array_split(np.arange(len(probs)), n_bins)
    ece = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        conf, acc = probs[b].mean(), correct[b].mean()
        ece += (len(b) / len(probs)) * abs(conf - acc)
    return float(ece)


def signed_overconfidence_gap(probs: np.ndarray, correct: np.ndarray) -> float:
    """mean(confidence) - mean(accuracy); positive => over-confident."""
    return float(np.mean(probs) - np.mean(correct))


# --- faithfulness: deletion/AOPC (the SOLE faithfulness adjudicator) --
def aopc_deletion(predict_fn: Callable, x: np.ndarray, attribution: np.ndarray,
                  top_k: Optional[int] = None) -> float:
    """Area Over the Perturbation Curve for top-k feature deletion vs the model's
    OWN top features. Compare against a random-deletion baseline (caller).
    High AOPC => attributions are faithful to THIS model's computation."""
    raise NotImplementedError("Implement top-k deletion AOPC for the compressed model.")


# --- stability + the four-case decomposition (NOT faithfulness) -------
def stability_spearman(attr_anchor: np.ndarray, attr_compressed: np.ndarray) -> float:
    """Per-instance Spearman of attributions vs the M0 anchor. This is STABILITY,
    reported separately from faithfulness. Compare against the redundancy/retraining
    null band (two M0 seeds, no compression)."""
    raise NotImplementedError("Implement per-instance Spearman + redundancy null band.")


def stratify_drift_by_prediction_change(drift: np.ndarray, pred_flipped: np.ndarray) -> dict:
    """The grouping-free redundancy control. unchanged-prediction drift = C1/C2
    floor (redundancy + representational); flipped-prediction drift = C3
    (behavior-coupled). The DIFFERENCE is the real signal. No correlation graph,
    no threshold, no arbitrary cutoff."""
    drift, pred_flipped = np.asarray(drift), np.asarray(pred_flipped).astype(bool)
    return {
        "floor_unchanged": float(drift[~pred_flipped].mean()) if (~pred_flipped).any() else float("nan"),
        "coupled_flipped": float(drift[pred_flipped].mean()) if pred_flipped.any() else float("nan"),
    }


def call_unfaithful(stability_drifted: bool, deletion_degraded: bool) -> bool:
    """Frozen decision rule: genuine unfaithfulness (C4) requires BOTH stability
    drift AND deletion degradation. Drift alone (C2/C3) is faithful."""
    return bool(stability_drifted and deletion_degraded)
