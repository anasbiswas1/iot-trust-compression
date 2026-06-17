
"""src/explanation_trust.py — explanation-trust under compression (built fresh, no SHAP dep).
Two SEPARATE axes: STABILITY/DRIFT (per-instance Spearman of KernelSHAP attributions vs M0,
judged against explainer-noise floor + retraining null band) and FAITHFULNESS (top-k deletion
AOPC vs random). They dissociate: drift can be high while faithfulness is preserved."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .config import CFG, PATHS
from . import train as _train


def _predict_proba_np(model, X, is_half=False):
    """Softmax probs for a numpy batch, run on the MODEL's own device (int8 models are
    CPU-only; M0/pruned live on GPU). Batched to avoid OOM."""
    import torch
    # detect the model's device from its first parameter (default cpu if none/quantized)
    try:
        dev = next(model.parameters()).device
    except (StopIteration, AttributeError):
        dev = torch.device("cpu")
    model.eval()
    outs = []
    bs = 8192
    with torch.no_grad():
        for i in range(0, len(X), bs):
            t = torch.tensor(X[i:i+bs], dtype=torch.float32).to(dev)
            outs.append(model(t).detach().cpu().numpy())
    logits = np.concatenate(outs, 0)
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def kernel_shap_instance(model, x, background, target_class, n_coalitions=200, is_half=False, seed=0):
    rng = np.random.default_rng(seed); d = len(x); bg_mean = background.mean(0)
    sizes = rng.integers(1, d, size=n_coalitions); Z = np.zeros((n_coalitions, d))
    for i, s in enumerate(sizes):
        on = rng.choice(d, s, replace=False); Z[i, on] = 1
    Xz = Z * x + (1 - Z) * bg_mean
    fz = _predict_proba_np(model, Xz, is_half)[:, target_class]
    f1 = _predict_proba_np(model, x[None], is_half)[0, target_class]
    f0 = _predict_proba_np(model, bg_mean[None], is_half)[0, target_class]
    k = Z.sum(1).astype(int)
    from scipy.special import comb
    w = (d - 1) / (comb(d, k) * k * (d - k) + 1e-12)
    A = Z; y = fz - f0; W = np.diag(w)
    AtWA = A.T @ W @ A + 1e-6 * np.eye(d); AtWy = A.T @ W @ y; ones = np.ones(d)
    inv = np.linalg.solve(AtWA, np.column_stack([AtWy, ones]))
    phi_u, lam_dir = inv[:, 0], inv[:, 1]
    mu = (ones @ phi_u - (f1 - f0)) / (ones @ lam_dir + 1e-12)
    return phi_u - mu * lam_dir


def attributions_for_class(model, X_class, background, target_class, n_coalitions=200, is_half=False, seed=0):
    return np.array([kernel_shap_instance(model, X_class[i], background, target_class,
                     n_coalitions, is_half, seed + i) for i in range(len(X_class))])


def attribution_stability(phi_a, phi_b):
    out = []
    for i in range(len(phi_a)):
        r, _ = spearmanr(phi_a[i], phi_b[i]); out.append(r if r == r else 0.0)
    return np.array(out)


def aopc_faithfulness(model, X_class, phi, target_class, background, ks=(1,2,3,5,8), is_half=False, seed=0):
    rng = np.random.default_rng(seed); bg = background.mean(0)
    base = _predict_proba_np(model, X_class, is_half)[:, target_class]; d = X_class.shape[1]
    res = {"k": list(ks), "topk_drop": [], "random_drop": []}
    for k in ks:
        topk_idx = np.argsort(-phi, axis=1)[:, :k]; Xt = X_class.copy()
        for i in range(len(Xt)): Xt[i, topk_idx[i]] = bg[topk_idx[i]]
        pt = _predict_proba_np(model, Xt, is_half)[:, target_class]
        Xr = X_class.copy()
        for i in range(len(Xr)):
            ri = rng.choice(d, k, replace=False); Xr[i, ri] = bg[ri]
        pr = _predict_proba_np(model, Xr, is_half)[:, target_class]
        res["topk_drop"].append(float((base - pt).mean())); res["random_drop"].append(float((base - pr).mean()))
    res["aopc_topk"] = float(np.mean(res["topk_drop"])); res["aopc_random"] = float(np.mean(res["random_drop"]))
    res["faithful"] = bool(res["aopc_topk"] > res["aopc_random"])
    res["faithfulness_margin"] = round(res["aopc_topk"] - res["aopc_random"], 4)
    return res


def classify_four_case(drift_median, faithful, retrain_null, margin=0.15):
    """Drift is 'high' if attribution stability falls meaningfully BELOW the retraining
    null band (two M0 seeds) -- i.e. compression perturbs explanations more than mere
    retraining does. Compared against the retraining null, NOT the explainer-noise floor."""
    high_drift = drift_median < (retrain_null - margin)
    if not high_drift and faithful:   return "stable_and_faithful"
    if high_drift and faithful:       return "drifted_but_faithful"
    if not high_drift and not faithful: return "stable_but_unfaithful"
    return "drifted_and_unfaithful"
