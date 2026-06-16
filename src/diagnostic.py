"""
src/diagnostic.py — the headline. Mechanism-grounded, pre-deployment, per-class.

From M0 ALONE, rank attack classes by predicted post-compression collapse, using
theory-derived features (incl. neural-collapse geometry from src/geometry.py).
Keep the rule deliberately simple (1-2 features, monotone/threshold) — meta-n is
tiny even at 34-type granularity.

MUST beat: (a) frequency-only, (b) Tran & Fioretto margin/grad-norm-only.
Generalisation: across architectures (EARNS the causal claim), across compression
families (held-out, no pooling), across datasets (leave-one-out incl. Bot-IoT).
"""
from __future__ import annotations
import numpy as np


def build_features(baseline_bundle) -> "pd.DataFrame":
    """Assemble the per-class baseline feature table (M0 only): sample_count,
    margin, separability, attribution_concentration (FLAGGED redundancy-confounded),
    effective_rank, neural_collapse_geometry."""
    raise NotImplementedError("Assemble per-class M0 feature table from geometry/metrics.")


def fit_simple_rule(features, target):
    """Low-capacity predictor (monotone/threshold or 1-2 feature linear)."""
    raise NotImplementedError("Fit deliberately simple rule; resist multivariate black boxes.")


def baseline_frequency_only(features):
    """Dumb baseline: rank by 1/sample_count. The diagnostic must beat this."""
    return -features["sample_count"].to_numpy()


def baseline_tran_fioretto(features):
    """Margin/grad-norm-only predictor (NeurIPS 2022). Required comparison baseline."""
    raise NotImplementedError("Implement margin + gradient-norm-only predictor.")


def evaluate(pred, actual_delta_recall) -> dict:
    """Rank correlation (Spearman/Kendall + bootstrap CI), precision@k, R^2 +
    predicted-vs-actual calibration. Report failure cases."""
    raise NotImplementedError("Implement rank-corr / precision@k / calibration eval.")
