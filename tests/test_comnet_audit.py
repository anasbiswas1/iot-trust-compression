"""Synthetic tests for Computer Networks extension helpers.

No dataset, model checkpoint, Drive mount, GPU, or internet connection is needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.comnet_audit import (
    assign_validation_tiers,
    calibration_summary,
    family_mapping_table,
    infer_family,
    security_semantics,
    stratified_cap_indices,
    strict_ovr_probe_table,
    strict_pairwise_probe_table,
)


def test_security_semantics_basic() -> None:
    names = ["BenignTraffic", "DoS-UDP_Flood", "DDoS-UDP_Flood"]
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])
    summary, fam, sub = security_semantics(y_true, y_pred, names)

    assert summary.loc[0, "fn_attack_to_benign"] == 1
    assert summary.loc[0, "fp_benign_to_attack"] == 1
    assert 0 <= summary.loc[0, "binary_attack_recall"] <= 1
    assert "macro_average" in set(fam["family"])
    assert set(sub["error_type"]).issuperset(
        {"attack_to_benign", "benign_to_attack", "attack_to_same_family_attack"}
    )

    mapping = family_mapping_table(names)
    assert list(mapping["alert_family"]) == ["benign", "dos", "ddos"]


def test_calibration_summary() -> None:
    probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    y = np.array([0, 1, 1])
    out = calibration_summary(probs, y)
    assert out.loc[0, "n"] == 3
    assert out.loc[0, "nll"] >= 0
    assert out.loc[0, "multiclass_brier"] >= 0
    assert 0 <= out.loc[0, "ece_15"] <= 1


def test_validation_tiers_and_family_fallback() -> None:
    r = pd.DataFrame(
        {0: [0.01, 0.95, 0.50, 0.50], 1: [0.02, 0.96, 0.52, 0.10]},
        index=["floor", "robust", "measure", "unstable"],
    )
    out = assign_validation_tiers(r)
    assert out.loc["floor", "validation_tier"] == "floored"
    assert out.loc["robust", "validation_tier"] == "robust"
    assert infer_family("ransomware") == "malware_botnet"


def test_stratified_cap_is_deterministic_and_balanced() -> None:
    y = np.repeat(np.arange(3), [10, 20, 30])
    a = stratified_cap_indices(y, max_per_class=7, seed=4)
    b = stratified_cap_indices(y, max_per_class=7, seed=4)
    assert np.array_equal(a, b)
    assert np.array_equal(np.bincount(y[a], minlength=3), np.array([7, 7, 7]))


def test_strict_probes_retain_separable_information() -> None:
    rng = np.random.default_rng(8)
    n_fit, n_test = 90, 45
    y_fit = np.repeat(np.arange(3), n_fit)
    y_test = np.repeat(np.arange(3), n_test)
    centres = np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    fit0 = np.vstack([rng.normal(centres[c], 0.55, size=(n_fit, 2)) for c in range(3)])
    test0 = np.vstack([rng.normal(centres[c], 0.55, size=(n_test, 2)) for c in range(3)])
    fitc = fit0 + rng.normal(0, 0.06, size=fit0.shape)
    testc = test0 + rng.normal(0, 0.06, size=test0.shape)
    names = ["a", "b", "c"]

    ovr = strict_ovr_probe_table(
        fit0, test0, fitc, testc, y_fit, y_test, names,
        bootstrap_B=30, seed=2,
    )
    assert len(ovr) == 3
    assert (ovr["auc_compressed"] > 0.95).all()
    assert ovr["auc_drop_ci_low"].notna().all()

    pair = strict_pairwise_probe_table(
        fit0, test0, fitc, testc, y_fit, y_test, names,
        [("a", "b")], bootstrap_B=30, seed=2,
    )
    assert len(pair) == 1
    assert pair.loc[0, "pair_auc_compressed"] > 0.95
