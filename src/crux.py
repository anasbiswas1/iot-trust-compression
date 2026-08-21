"""Array-level linear-probe compatibility utilities.

The canonical model/data pipeline lives in :mod:`src.explain` (`crux_probe`).
This module provides a small, tested array interface for callers that already
have anchor and compressed penultimate representations.

Important scope
---------------
The function below uses one shared random stratified split for both feature
matrices. It is useful for a *relative* anchor-versus-compressed comparison,
but it does not make an internally mixed partition leakage-aware. For a
stronger evaluation, fit probes on original train/validation representations
and evaluate once on the untouched provenance-ordered test partition.
"""
from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def _validate_inputs(
    feats_anchor: np.ndarray,
    feats_comp: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalise probe inputs without changing sample order."""
    anchor = np.asarray(feats_anchor)
    comp = np.asarray(feats_comp)
    y = np.asarray(labels)

    if anchor.ndim != 2 or comp.ndim != 2:
        raise ValueError("feats_anchor and feats_comp must both be 2-D arrays")
    if y.ndim != 1:
        raise ValueError("labels must be a 1-D array")
    if anchor.shape[0] != comp.shape[0] or anchor.shape[0] != y.shape[0]:
        raise ValueError(
            "anchor features, compressed features, and labels must contain "
            "the same number of samples"
        )
    if anchor.shape[0] < 10:
        raise ValueError("at least 10 samples are required for a held-out probe")
    if not np.isfinite(anchor).all() or not np.isfinite(comp).all():
        raise ValueError("feature arrays contain NaN or infinite values")
    if np.unique(y).size < 2:
        raise ValueError("labels must contain at least two classes")

    return anchor, comp, y


def _python_scalar(value: Any) -> Hashable:
    """Convert NumPy scalar labels to ordinary Python hashable values."""
    return value.item() if isinstance(value, np.generic) else value


def probe_recoverability(
    feats_anchor: np.ndarray,
    feats_comp: np.ndarray,
    labels: np.ndarray,
    seed: int = 0,
    *,
    test_size: float = 0.40,
    C: float = 1.0,
    max_iter: int = 500,
    min_train_positive: int = 5,
    min_test_positive: int = 2,
) -> dict[Hashable, dict[str, float | int]]:
    """Compare one-vs-rest linear decodability on two representations.

    Parameters
    ----------
    feats_anchor, feats_comp:
        Penultimate representations with matching rows. Feature dimensions may
        differ, but sample order must be identical.
    labels:
        One class label per row.
    seed:
        Random seed used for the single shared stratified split.
    test_size:
        Fraction assigned to the probe's held-out fold.
    C, max_iter:
        Logistic-regression settings.
    min_train_positive, min_test_positive:
        Per-class positive-count gates. Classes below a gate are returned with
        NaN AUC values instead of causing an opaque estimator failure.

    Returns
    -------
    dict
        ``{class_label: metrics}``, where metrics include anchor and compressed
        AUC, AUC drop, retention ratio, and split support. AUC drop is
        ``auc_anchor - auc_compressed``; positive values indicate reduced
        decodability after compression.

    Notes
    -----
    This helper deliberately reports a continuum. It does not label a class as
    "remembered" or "forgotten", and it does not interpret high absolute AUC
    as leakage-robust unless the caller supplied a genuinely external probe
    train/evaluation boundary.
    """
    anchor, comp, y = _validate_inputs(feats_anchor, feats_comp, labels)

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1")
    if C <= 0:
        raise ValueError("C must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    indices = np.arange(y.shape[0])
    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )
    except ValueError as exc:
        raise ValueError(
            "a shared stratified split could not be created; ensure every "
            "class has enough samples for both folds"
        ) from exc

    def fit_auc(features: np.ndarray, binary_y: np.ndarray) -> float:
        estimator = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight="balanced",
            solver="lbfgs",
        )
        estimator.fit(features[train_idx], binary_y[train_idx])
        scores = estimator.predict_proba(features[test_idx])[:, 1]
        return float(roc_auc_score(binary_y[test_idx], scores))

    results: dict[Hashable, dict[str, float | int]] = {}
    for class_value in np.unique(y):
        binary_y = (y == class_value).astype(np.int8)
        train_positive = int(binary_y[train_idx].sum())
        test_positive = int(binary_y[test_idx].sum())
        train_negative = int(train_idx.size - train_positive)
        test_negative = int(test_idx.size - test_positive)

        valid = (
            train_positive >= min_train_positive
            and test_positive >= min_test_positive
            and train_negative >= min_train_positive
            and test_negative >= min_test_positive
        )

        if valid:
            auc_anchor = fit_auc(anchor, binary_y)
            auc_compressed = fit_auc(comp, binary_y)
            auc_drop = auc_anchor - auc_compressed
            retention = (
                auc_compressed / auc_anchor if abs(auc_anchor) > 1e-12 else np.nan
            )
        else:
            auc_anchor = auc_compressed = auc_drop = retention = np.nan

        results[_python_scalar(class_value)] = {
            "auc_anchor": float(auc_anchor),
            "auc_compressed": float(auc_compressed),
            "auc_drop": float(auc_drop),
            "retention": float(retention),
            "support_total": int(binary_y.sum()),
            "support_train": train_positive,
            "support_test": test_positive,
            "n_train": int(train_idx.size),
            "n_test": int(test_idx.size),
        }

    return results
