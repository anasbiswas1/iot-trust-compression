"""Computer Networks submission audit helpers.

This module adds analyses requested by the journal-facing audit without changing
or overwriting any archived manuscript result.  All outputs should be written to
``results/tables/comnet`` or ``results/logs/comnet`` by the companion notebooks.

The functions are intentionally data/model agnostic and can be unit-tested on
small synthetic arrays.  Existing repository modules remain the source of truth
for model training, compression, feature extraction, and recovery.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import copy
import io
import json
import os
import platform
import time

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CICIoT2023 alert-family mapping
# ---------------------------------------------------------------------------

CICIOT2023_FAMILY_MAP: dict[str, str] = {
    "BenignTraffic": "benign",
    # DDoS
    "DDoS-RSTFINFlood": "ddos",
    "DDoS-PSHACK_Flood": "ddos",
    "DDoS-SYN_Flood": "ddos",
    "DDoS-UDP_Flood": "ddos",
    "DDoS-ICMP_Flood": "ddos",
    "DDoS-SynonymousIP_Flood": "ddos",
    "DDoS-ACK_Fragmentation": "ddos",
    "DDoS-UDP_Fragmentation": "ddos",
    "DDoS-ICMP_Fragmentation": "ddos",
    "DDoS-TCP_Flood": "ddos",
    "DDoS-HTTP_Flood": "ddos",
    "DDoS-SlowLoris": "ddos",
    # DoS
    "DoS-UDP_Flood": "dos",
    "DoS-SYN_Flood": "dos",
    "DoS-TCP_Flood": "dos",
    "DoS-HTTP_Flood": "dos",
    # Reconnaissance / scanning
    "Recon-PingSweep": "reconnaissance",
    "Recon-OSScan": "reconnaissance",
    "Recon-PortScan": "reconnaissance",
    "Recon-HostDiscovery": "reconnaissance",
    "VulnerabilityScan": "reconnaissance",
    # Spoofing / MITM
    "MITM-ArpSpoofing": "spoofing_mitm",
    "DNS_Spoofing": "spoofing_mitm",
    # Credential / brute-force
    "DictionaryBruteForce": "credential_attack",
    # Web/application attacks
    "SqlInjection": "web_application",
    "CommandInjection": "web_application",
    "XSS": "web_application",
    "BrowserHijacking": "web_application",
    # Malware / botnet behaviour
    "Backdoor_Malware": "malware_botnet",
    "Uploading_Attack": "malware_botnet",
    "Mirai-greip_flood": "malware_botnet",
    "Mirai-greeth_flood": "malware_botnet",
    "Mirai-udpplain": "malware_botnet",
}


def infer_family(label: str) -> str:
    """Return a conservative alert family for a class label.

    The explicit map is used first.  The fallback exists only to keep notebooks
    robust to harmless spelling variants in archived checkpoints.
    """
    if label in CICIOT2023_FAMILY_MAP:
        return CICIOT2023_FAMILY_MAP[label]
    low = label.lower().replace("-", "_")
    if "benign" in low or low == "normal":
        return "benign"
    if low.startswith("ddos"):
        return "ddos"
    if low.startswith("dos"):
        return "dos"
    if "recon" in low or "scan" in low or "hostdiscovery" in low:
        return "reconnaissance"
    if "spoof" in low or "mitm" in low or "arp" in low:
        return "spoofing_mitm"
    if "brute" in low or "password" in low or "credential" in low:
        return "credential_attack"
    if any(k in low for k in ("sql", "xss", "command", "browser", "injection")):
        return "web_application"
    if any(k in low for k in ("malware", "backdoor", "upload", "mirai", "ransom")):
        return "malware_botnet"
    return "other_attack"


def family_vector(class_names: Sequence[str], family_map: Mapping[str, str] | None = None) -> np.ndarray:
    mapping = dict(CICIOT2023_FAMILY_MAP)
    if family_map:
        mapping.update(family_map)
    return np.array([mapping.get(str(name), infer_family(str(name))) for name in class_names], dtype=object)


def family_mapping_table(
    class_names: Sequence[str], family_map: Mapping[str, str] | None = None
) -> pd.DataFrame:
    """Return the explicit fine-label to alert-family map used by the audit."""
    families = family_vector(class_names, family_map)
    return pd.DataFrame({
        "class_index": np.arange(len(class_names), dtype=int),
        "fine_label": [str(x) for x in class_names],
        "alert_family": families,
        "is_benign": families == "benign",
    })


def stratified_cap_indices(
    labels: Sequence, *, max_per_class: int | None = None, seed: int = 0
) -> np.ndarray:
    """Return deterministic, class-stratified row indices.

    ``max_per_class=None`` keeps all rows.  This helper is useful before feature
    extraction, avoiding multi-gigabyte probe matrices while preserving every
    class and using exactly the same sampled rows for M0 and compressed models.
    """
    y = np.asarray(labels)
    if y.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if max_per_class is None:
        return np.arange(len(y), dtype=int)
    if max_per_class < 1:
        raise ValueError("max_per_class must be positive or None")
    rng = np.random.default_rng(seed)
    parts = []
    for value in np.unique(y):
        idx = np.flatnonzero(y == value)
        if len(idx) > max_per_class:
            idx = np.sort(rng.choice(idx, max_per_class, replace=False))
        parts.append(idx)
    return np.sort(np.concatenate(parts)) if parts else np.array([], dtype=int)


# ---------------------------------------------------------------------------
# Security semantics and calibration
# ---------------------------------------------------------------------------

def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def security_semantics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    family_map: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute binary, family, and substitution-semantic metrics.

    Returns
    -------
    summary_df:
        One-row operational summary.
    family_df:
        Per-family precision/recall/F1/support.
    substitution_df:
        Counts and rates for attack->benign, benign->attack, same-family and
        cross-family attack substitutions.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_recall_fscore_support,
    )

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    names = np.asarray(class_names, dtype=object)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shape")
    if y_true.min(initial=0) < 0 or y_pred.min(initial=0) < 0:
        raise ValueError("Class indices must be non-negative")
    if max(y_true.max(initial=0), y_pred.max(initial=0)) >= len(names):
        raise ValueError("class_names does not cover all encoded labels")

    fam_by_class = family_vector(names, family_map)
    true_fam = fam_by_class[y_true]
    pred_fam = fam_by_class[y_pred]
    true_attack = true_fam != "benign"
    pred_attack = pred_fam != "benign"

    tp = int(np.sum(true_attack & pred_attack))
    fn = int(np.sum(true_attack & ~pred_attack))
    fp = int(np.sum(~true_attack & pred_attack))
    tn = int(np.sum(~true_attack & ~pred_attack))

    binary_precision = _safe_div(tp, tp + fp)
    binary_recall = _safe_div(tp, tp + fn)
    if np.isfinite(binary_precision) and np.isfinite(binary_recall) and (binary_precision + binary_recall) > 0:
        binary_f1 = float(2 * binary_precision * binary_recall / (binary_precision + binary_recall))
    else:
        binary_f1 = float("nan")
    binary_specificity = _safe_div(tn, tn + fp)
    binary_bal_acc = np.nanmean([binary_recall, binary_specificity])

    same_family_sub = true_attack & pred_attack & (y_true != y_pred) & (true_fam == pred_fam)
    cross_family_sub = true_attack & pred_attack & (true_fam != pred_fam)
    exact_attack_correct = true_attack & (y_true == y_pred)

    attack_n = int(true_attack.sum())
    benign_n = int((~true_attack).sum())
    error_n = int((y_true != y_pred).sum())

    summary = {
        "n": int(len(y_true)),
        "fine_accuracy": float(accuracy_score(y_true, y_pred)),
        "fine_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "family_accuracy": float(np.mean(true_fam == pred_fam)),
        "binary_attack_precision": binary_precision,
        "binary_attack_recall": binary_recall,
        "binary_attack_f1": binary_f1,
        "binary_balanced_accuracy": float(binary_bal_acc),
        "attack_to_benign_rate": _safe_div(fn, attack_n),
        "benign_to_attack_rate": _safe_div(fp, benign_n),
        "exact_attack_type_rate": _safe_div(int(exact_attack_correct.sum()), attack_n),
        "same_family_attack_substitution_rate": _safe_div(int(same_family_sub.sum()), attack_n),
        "cross_family_attack_substitution_rate": _safe_div(int(cross_family_sub.sum()), attack_n),
        "same_family_share_of_all_errors": _safe_div(int(same_family_sub.sum()), error_n),
        "cross_family_share_of_all_errors": _safe_div(int(cross_family_sub.sum()), error_n),
        "tp_attack": tp,
        "fn_attack_to_benign": fn,
        "fp_benign_to_attack": fp,
        "tn_benign": tn,
    }

    families = sorted(set(true_fam) | set(pred_fam))
    fam_to_i = {name: i for i, name in enumerate(families)}
    yt_f = np.array([fam_to_i[x] for x in true_fam], dtype=int)
    yp_f = np.array([fam_to_i[x] for x in pred_fam], dtype=int)
    p, r, f, s = precision_recall_fscore_support(
        yt_f, yp_f, labels=np.arange(len(families)), zero_division=0
    )
    family_df = pd.DataFrame({
        "family": families,
        "precision": p,
        "recall": r,
        "f1": f,
        "support": s.astype(int),
    })
    family_df.loc[len(family_df)] = {
        "family": "macro_average",
        "precision": float(np.mean(p)),
        "recall": float(np.mean(r)),
        "f1": float(np.mean(f)),
        "support": int(s.sum()),
    }

    substitution_df = pd.DataFrame([
        {"error_type": "attack_to_benign", "count": fn, "rate_over_true_attack": _safe_div(fn, attack_n)},
        {"error_type": "benign_to_attack", "count": fp, "rate_over_true_benign": _safe_div(fp, benign_n)},
        {"error_type": "attack_to_same_family_attack", "count": int(same_family_sub.sum()),
         "rate_over_true_attack": _safe_div(int(same_family_sub.sum()), attack_n)},
        {"error_type": "attack_to_cross_family_attack", "count": int(cross_family_sub.sum()),
         "rate_over_true_attack": _safe_div(int(cross_family_sub.sum()), attack_n)},
        {"error_type": "exact_attack_type_correct", "count": int(exact_attack_correct.sum()),
         "rate_over_true_attack": _safe_div(int(exact_attack_correct.sum()), attack_n)},
    ])
    return pd.DataFrame([summary]), family_df, substitution_df


def adaptive_ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    order = np.argsort(confidence)
    bins = np.array_split(order, n_bins)
    return float(sum(
        (len(b) / len(confidence)) * abs(confidence[b].mean() - correct[b].mean())
        for b in bins if len(b)
    ))


def calibration_summary(probs: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    """Return ECE sensitivity, NLL, multiclass Brier, and confident-error rates."""
    from sklearn.metrics import f1_score, log_loss

    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if probs.ndim != 2 or len(probs) != len(y_true):
        raise ValueError("probs must be (n, K) and align with y_true")
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == y_true
    one_hot = np.eye(probs.shape[1], dtype=float)[y_true]
    return pd.DataFrame([{
        "n": int(len(y_true)),
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "nll": float(log_loss(y_true, probs, labels=np.arange(probs.shape[1]))),
        "multiclass_brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "ece_10": adaptive_ece(conf, correct, 10),
        "ece_15": adaptive_ece(conf, correct, 15),
        "ece_20": adaptive_ece(conf, correct, 20),
        "mean_confidence": float(conf.mean()),
        "confidently_wrong_0.5": float(np.mean((~correct) & (conf > 0.5))),
        "confidently_wrong_0.7": float(np.mean((~correct) & (conf > 0.7))),
        "confidently_wrong_0.9": float(np.mean((~correct) & (conf > 0.9))),
    }])


# ---------------------------------------------------------------------------
# Validation-frozen class tiers
# ---------------------------------------------------------------------------

def assign_validation_tiers(
    recall_by_seed: pd.DataFrame,
    *,
    floored_max: float = 0.075,
    robust_min: float = 0.90,
    unstable_2sd_min: float = 0.20,
) -> pd.DataFrame:
    """Assign class tiers from validation recalls only.

    ``recall_by_seed`` must be indexed by class, with one column per independent
    baseline seed.  The thresholds intentionally mirror the manuscript's broad
    tier semantics but are explicit and sensitivity-checkable.
    """
    x = recall_by_seed.astype(float)
    mean = x.mean(axis=1)
    sd = x.std(axis=1, ddof=1)
    band = 2.0 * sd
    tier = []
    for m, b in zip(mean, band):
        if m <= floored_max:
            tier.append("floored")
        elif b >= unstable_2sd_min:
            tier.append("unstable_confusable")
        elif m >= robust_min:
            tier.append("robust")
        else:
            tier.append("measurable")
    return pd.DataFrame({"validation_recall_mean": mean, "validation_recall_sd": sd,
                         "validation_2sd_band": band, "validation_tier": tier})


# ---------------------------------------------------------------------------
# Strict probes
# ---------------------------------------------------------------------------

def _fit_logistic_scores(
    X_fit, y_fit, X_test, *, C: float = 1.0, seed: int = 0
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1500, C=C, class_weight="balanced", random_state=seed,
            solver="lbfgs",
        )),
    ])
    pipe.fit(X_fit, y_fit)
    return pipe.predict_proba(X_test)[:, 1]


def _auc_and_ci(
    y_true: np.ndarray, score_m0: np.ndarray, score_comp: np.ndarray,
    *, bootstrap_B: int, seed: int
) -> tuple[float, float, float, float, float, float]:
    from sklearn.metrics import roc_auc_score

    auc0 = float(roc_auc_score(y_true, score_m0))
    aucc = float(roc_auc_score(y_true, score_comp))
    drop = auc0 - aucc
    if bootstrap_B <= 0:
        return auc0, aucc, drop, float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y_true)
    for _ in range(bootstrap_B):
        idx = rng.integers(0, n, n)
        if np.unique(y_true[idx]).size < 2:
            continue
        vals.append(
            roc_auc_score(y_true[idx], score_m0[idx])
            - roc_auc_score(y_true[idx], score_comp[idx])
        )
    if not vals:
        return auc0, aucc, drop, float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return auc0, aucc, drop, float(lo), float(hi), float(np.std(vals, ddof=1))


def strict_ovr_probe_table(
    feats_fit_m0: np.ndarray,
    feats_test_m0: np.ndarray,
    feats_fit_comp: np.ndarray,
    feats_test_comp: np.ndarray,
    y_fit: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
    *,
    C: float = 1.0,
    seed: int = 0,
    bootstrap_B: int = 0,
    class_filter: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Fit probes on train/validation representations and evaluate on test.

    Both representations use identical fitting and test rows.  Optional paired
    test-set bootstrap intervals quantify the AUC-drop uncertainty.
    """
    keep = set(class_filter) if class_filter is not None else None
    rows = []
    for c, name in enumerate(class_names):
        if keep is not None and str(name) not in keep:
            continue
        yf = (np.asarray(y_fit) == c).astype(int)
        yt = (np.asarray(y_test) == c).astype(int)
        if yf.sum() < 5 or yt.sum() < 5 or (len(yf) - yf.sum()) < 5 or (len(yt) - yt.sum()) < 5:
            continue
        score0 = _fit_logistic_scores(feats_fit_m0, yf, feats_test_m0, C=C, seed=seed)
        scorec = _fit_logistic_scores(feats_fit_comp, yf, feats_test_comp, C=C, seed=seed)
        a0, ac, drop, lo, hi, boot_sd = _auc_and_ci(
            yt, score0, scorec, bootstrap_B=bootstrap_B, seed=seed + c
        )
        rows.append({
            "label": str(name), "auc_M0": a0, "auc_compressed": ac,
            "auc_drop": drop, "auc_drop_ci_low": lo, "auc_drop_ci_high": hi,
            "auc_drop_bootstrap_sd": boot_sd,
            "n_fit_positive": int(yf.sum()), "n_fit_total": int(len(yf)),
            "n_test_positive": int(yt.sum()), "n_test_total": int(len(yt)),
            "C": float(C), "seed": int(seed),
        })
    return pd.DataFrame(rows)


def strict_pairwise_probe_table(
    feats_fit_m0: np.ndarray,
    feats_test_m0: np.ndarray,
    feats_fit_comp: np.ndarray,
    feats_test_comp: np.ndarray,
    y_fit: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    *,
    C: float = 1.0,
    seed: int = 0,
    bootstrap_B: int = 0,
) -> pd.DataFrame:
    name_to_i = {str(n): i for i, n in enumerate(class_names)}
    rows = []
    y_fit = np.asarray(y_fit)
    y_test = np.asarray(y_test)
    for pair_i, (collapsed, absorber) in enumerate(pairs):
        if collapsed not in name_to_i or absorber not in name_to_i:
            continue
        c, a = name_to_i[collapsed], name_to_i[absorber]
        mf = (y_fit == c) | (y_fit == a)
        mt = (y_test == c) | (y_test == a)
        yf = (y_fit[mf] == c).astype(int)
        yt = (y_test[mt] == c).astype(int)
        if min(yf.sum(), len(yf)-yf.sum(), yt.sum(), len(yt)-yt.sum()) < 5:
            continue
        score0 = _fit_logistic_scores(feats_fit_m0[mf], yf, feats_test_m0[mt], C=C, seed=seed)
        scorec = _fit_logistic_scores(feats_fit_comp[mf], yf, feats_test_comp[mt], C=C, seed=seed)
        a0, ac, drop, lo, hi, boot_sd = _auc_and_ci(
            yt, score0, scorec, bootstrap_B=bootstrap_B, seed=seed + pair_i
        )
        rows.append({
            "collapsed": collapsed, "absorber": absorber,
            "pair_auc_M0": a0, "pair_auc_compressed": ac, "pair_auc_drop": drop,
            "pair_auc_drop_ci_low": lo, "pair_auc_drop_ci_high": hi,
            "pair_auc_drop_bootstrap_sd": boot_sd,
            "n_fit": int(mf.sum()), "n_test": int(mt.sum()),
            "n_fit_collapsed": int(yf.sum()), "n_test_collapsed": int(yt.sum()),
            "C": float(C), "seed": int(seed),
        })
    return pd.DataFrame(rows)


def bootstrap_auc_difference(
    score_m0: np.ndarray,
    score_comp: np.ndarray,
    y_true: np.ndarray,
    *,
    B: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for AUC(M0)-AUC(compressed) on a fixed test set."""
    from sklearn.metrics import roc_auc_score

    score_m0 = np.asarray(score_m0, dtype=float)
    score_comp = np.asarray(score_comp, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    point = float(roc_auc_score(y_true, score_m0) - roc_auc_score(y_true, score_comp))
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y_true)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if np.unique(y_true[idx]).size < 2:
            continue
        vals.append(roc_auc_score(y_true[idx], score_m0[idx]) - roc_auc_score(y_true[idx], score_comp[idx]))
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


# ---------------------------------------------------------------------------
# Compression masks and BatchNorm recalibration
# ---------------------------------------------------------------------------

def nonzero_mask(model) -> dict[str, np.ndarray]:
    """Return Boolean masks for Conv1d/Linear weight tensors."""
    import torch.nn as nn

    out: dict[str, np.ndarray] = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            out[f"{name}.weight"] = module.weight.detach().cpu().numpy() != 0
    return out


def mask_jaccard(model_a, model_b) -> pd.DataFrame:
    ma, mb = nonzero_mask(model_a), nonzero_mask(model_b)
    rows = []
    for name in sorted(set(ma) & set(mb)):
        a, b = ma[name], mb[name]
        if a.shape != b.shape:
            continue
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        rows.append({"tensor": name, "jaccard_nonzero": _safe_div(inter, union),
                     "nonzero_a": int(a.sum()), "nonzero_b": int(b.sum()), "total": int(a.size)})
    if rows:
        total_inter = sum(np.logical_and(ma[r["tensor"]], mb[r["tensor"]]).sum() for r in rows)
        total_union = sum(np.logical_or(ma[r["tensor"]], mb[r["tensor"]]).sum() for r in rows)
        rows.append({"tensor": "GLOBAL", "jaccard_nonzero": _safe_div(total_inter, total_union),
                     "nonzero_a": sum(r["nonzero_a"] for r in rows),
                     "nonzero_b": sum(r["nonzero_b"] for r in rows),
                     "total": sum(r["total"] for r in rows)})
    return pd.DataFrame(rows)


def recalibrate_batchnorm(model, X_train, *, batch_size: int = 8192, reset: bool = True, device: str | None = None):
    """Update BatchNorm running statistics without gradient updates.

    Returns a deep-copied model so the input object is never mutated.
    """
    import torch
    import torch.nn as nn

    m = copy.deepcopy(model)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device)
    for module in m.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if reset:
                module.reset_running_stats()
            module.momentum = None  # cumulative average over the calibration pass
    m.train()
    with torch.no_grad():
        for i in range(0, len(X_train), batch_size):
            xb = X_train[i:i + batch_size]
            if not torch.is_tensor(xb):
                xb = torch.tensor(xb, dtype=torch.float32)
            m(xb.to(device))
    m.eval()
    return m


def recalibrate_batchnorm_from_dataframe(
    model, df, indices: Sequence[int], scaler, feat_cols: Sequence[str],
    *, batch_size: int = 8192, reset: bool = True, device: str | None = None
):
    """Stream a DataFrame calibration subset through BatchNorm layers.

    Unlike materialising the entire standardised training matrix, this helper
    transforms one batch at a time.  The supplied indices should be selected
    without reference to test outcomes.
    """
    import torch
    import torch.nn as nn

    m = copy.deepcopy(model)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device)
    for module in m.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if reset:
                module.reset_running_stats()
            module.momentum = None
    m.train()
    idx = np.asarray(indices)
    with torch.no_grad():
        for i in range(0, len(idx), batch_size):
            sub = df.loc[idx[i:i + batch_size], list(feat_cols)].to_numpy(np.float32)
            xb = torch.tensor(scaler.transform(sub), dtype=torch.float32, device=device)
            m(xb)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Deployment measurement
# ---------------------------------------------------------------------------

def serialized_state_dict_bytes(model) -> int:
    """Actual bytes produced by torch.save(model.state_dict())."""
    import torch

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return int(buf.tell())


def index_value_sparse_payload_bytes(model, *, index_bytes: int = 4) -> int:
    """Estimate an index-value payload, not an executable sparse backend artifact."""
    import torch

    total = 0
    for p in model.parameters():
        arr = p.detach().cpu()
        nz = int((arr != 0).sum())
        value_bytes = arr.element_size()
        if nz == arr.numel():
            total += int(arr.numel()) * value_bytes
        else:
            total += nz * (value_bytes + index_bytes)
    return int(total)


def benchmark_cpu_model(
    model,
    input_dim: int,
    *,
    batch_sizes: Sequence[int] = (1, 32, 256, 1024),
    warmup: int = 20,
    repeats: int = 100,
    dtype: str = "float32",
) -> pd.DataFrame:
    """Dense PyTorch CPU latency/throughput benchmark.

    This intentionally measures the actual backend used rather than assuming
    unstructured zeros produce acceleration.
    """
    import torch

    m = copy.deepcopy(model).cpu().eval()
    dt = torch.float16 if dtype == "float16" else torch.float32
    rows = []
    try:
        import psutil
        process = psutil.Process(os.getpid())
    except Exception:
        process = None
    torch.set_grad_enabled(False)
    for b in batch_sizes:
        x = torch.randn(int(b), int(input_dim), dtype=dt)
        # Some CPU operators do not support fp16; report failure explicitly.
        try:
            rss_before = int(process.memory_info().rss) if process is not None else np.nan
            for _ in range(warmup):
                _ = m(x)
            rss_after_warmup = int(process.memory_info().rss) if process is not None else np.nan
            times_ms = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                _ = m(x)
                times_ms.append((time.perf_counter() - t0) * 1000.0)
            med = float(np.median(times_ms))
            p95 = float(np.percentile(times_ms, 95))
            rss_after = int(process.memory_info().rss) if process is not None else np.nan
            rows.append({"batch_size": int(b), "median_latency_ms": med,
                         "p95_latency_ms": p95,
                         "throughput_flows_per_s": float(1000.0 * b / med),
                         "warmup_runs": int(warmup), "timed_repeats": int(repeats),
                         "input_dtype": dtype,
                         "rss_before_bytes": rss_before,
                         "rss_after_warmup_bytes": rss_after_warmup,
                         "rss_after_benchmark_bytes": rss_after,
                         "status": "ok"})
        except Exception as exc:  # keep notebook running and make unsupported paths visible
            rows.append({"batch_size": int(b), "median_latency_ms": np.nan,
                         "p95_latency_ms": np.nan, "throughput_flows_per_s": np.nan,
                         "warmup_runs": int(warmup), "timed_repeats": int(repeats),
                         "input_dtype": dtype,
                         "rss_before_bytes": np.nan,
                         "rss_after_warmup_bytes": np.nan,
                         "rss_after_benchmark_bytes": np.nan,
                         "status": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def environment_record() -> dict:
    rec = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        rec.update({"torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()),
                    "torch_threads": int(torch.get_num_threads())})
    except Exception:
        pass
    try:
        import psutil
        rec["ram_bytes"] = int(psutil.virtual_memory().total)
    except Exception:
        pass
    return rec


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
