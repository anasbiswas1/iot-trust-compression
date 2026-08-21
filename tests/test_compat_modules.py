"""Synthetic tests for public compatibility utilities.

These tests require no dataset, checkpoint, GPU, Drive mount, or Colab runtime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.crux import probe_recoverability
from src.diagnostic import (
    baseline_frequency_only,
    baseline_tran_fioretto,
    build_features,
    evaluate,
    fit_simple_rule,
)


def test_probe_recoverability_uses_matching_rows_and_reports_all_classes() -> None:
    rng = np.random.default_rng(12)
    n_per_class = 80
    y = np.repeat(np.arange(3), n_per_class)

    centres = np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    anchor = np.vstack(
        [rng.normal(loc=centres[c], scale=0.55, size=(n_per_class, 2)) for c in range(3)]
    )
    compressed = anchor + rng.normal(scale=0.08, size=anchor.shape)

    result = probe_recoverability(anchor, compressed, y, seed=3)

    assert set(result) == {0, 1, 2}
    for metrics in result.values():
        assert metrics["support_total"] == n_per_class
        assert metrics["support_train"] + metrics["support_test"] == n_per_class
        assert 0.0 <= metrics["auc_anchor"] <= 1.0
        assert 0.0 <= metrics["auc_compressed"] <= 1.0
        assert metrics["auc_anchor"] > 0.95
        assert metrics["auc_compressed"] > 0.95


def test_build_features_accepts_dataframe_and_mapping() -> None:
    frame = pd.DataFrame({"support": [10, 20], "margin": [0.2, 0.5]})
    direct = build_features(frame)
    mapped = build_features({"feature_df": frame})

    assert direct.equals(frame)
    assert mapped.equals(frame)
    assert direct is not frame
    assert mapped is not frame


def test_diagnostic_baselines_rule_and_evaluation() -> None:
    features = pd.DataFrame(
        {
            "support": [20, 40, 80, 160, 320, 640],
            "margin": [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
            "gradient_norm": [1.8, 1.5, 1.2, 0.9, 0.6, 0.3],
            "confusion_offdiag": [0.9, 0.8, 0.7, 0.5, 0.3, 0.1],
        }
    )
    actual = np.array([0.85, 0.72, 0.60, 0.42, 0.24, 0.10])

    frequency_score = baseline_frequency_only(features)
    margin_grad_score = baseline_tran_fioretto(features)

    assert frequency_score[0] > frequency_score[-1]
    assert margin_grad_score[0] > margin_grad_score[-1]

    model = fit_simple_rule(
        features,
        actual,
        feature_cols=["margin", "confusion_offdiag"],
        alpha=0.5,
    )
    prediction = model.predict(features[["margin", "confusion_offdiag"]])
    metrics = evaluate(prediction, actual, n_boot=100, seed=4)

    assert metrics["n"] == len(actual)
    assert -1.0 <= metrics["spearman"] <= 1.0
    assert metrics["mae"] >= 0.0
    assert 0.0 <= metrics["precision_at_k"] <= 1.0
