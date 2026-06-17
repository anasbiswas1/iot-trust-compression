
"""
src/predict.py — Stage 3: PREDICT (the headline).
Forecast which classes lose recall under compression, from the M0 model + data ALONE.
Option 3: assemble neural-collapse + confusability + frequency/capacity features, let
the data adjudicate. Tiny-n discipline: low-capacity rules, LOCO-CV, bootstrap CIs.
Must beat frequency-only AND margin-only (Tran & Fioretto) baselines.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, kendalltau

from .config import CFG, PATHS
from . import explain as _explain


def assemble_features(model_M0, df, splits, scaler, feat_cols, le, which="test",
                      max_n=60000, seed=0):
    """All candidate per-class predictor features from the M0 model only."""
    feats, logits, y = _explain.extract_features(model_M0, df, splits, scaler, feat_cols, le, which=which)
    rng = np.random.default_rng(seed)
    if len(y) > max_n:
        idx = rng.choice(len(y), max_n, replace=False)
        feats, logits, y = feats[idx], logits[idx], y[idx]
    classes = np.unique(y)
    names = [le.classes_[int(c)] for c in classes]
    pred = logits.argmax(1)
    probs = np.exp(logits - logits.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
    means = {int(c): feats[y == c].mean(0) for c in classes}
    support = {int(c): int((y == c).sum()) for c in classes}
    K = len(le.classes_)
    cm = np.zeros((K, K))
    for t, p in zip(y, pred):
        cm[t, p] += 1
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    etf = _explain.geom.etf_deviation(feats, y)
    marg = _explain.geom.per_class_margin(logits, y)
    rows = {}
    for c in classes:
        ci = int(c); m_c = means[ci]
        higher = [int(o) for o in classes if support[int(o)] > support[ci]]
        if higher:
            sims = [np.dot(m_c, means[o]) / (np.linalg.norm(m_c)*np.linalg.norm(means[o]) + 1e-9)
                    for o in higher]
            sim_nearest_higher = float(np.max(sims))
        else:
            sim_nearest_higher = 0.0   # most-frequent class: upward-confusability is zero, not missing
        offdiag = float(cm_norm[ci].sum() - cm_norm[ci, ci])
        pc = probs[y == c].mean(0); pc = pc[pc > 0]
        pred_entropy = float(-(pc * np.log(pc)).sum())
        rows[names[list(classes).index(c)]] = {
            "support": support[ci],
            "baseline_recall": float((pred[y == c] == c).mean()),
            "etf_cos_to_others": etf[c]["mean_cos_to_others"],
            "margin": marg[c],
            "sim_nearest_higher_freq": sim_nearest_higher,
            "confusion_offdiag": offdiag,
            "pred_entropy": pred_entropy,
        }
    return pd.DataFrame(rows).T


def build_prediction_table(feature_df, delta_recall_df, cells=("prune50","prune80","distillation"),
                           tiers=None, measurable_only=False):
    idx = feature_df.index
    if measurable_only and tiers is not None:
        idx = [c for c in idx if tiers.get(c) == "measurable"]
    recs = []
    for cls in idx:
        if cls not in delta_recall_df.index:
            continue
        for cell in cells:
            if cell not in delta_recall_df.columns:
                continue
            row = dict(feature_df.loc[cls])
            row["class"] = cls; row["cell"] = cell
            row["delta_recall"] = float(delta_recall_df.loc[cls, cell])
            if tiers is not None:
                row["tier"] = tiers.get(cls, "NA")
            recs.append(row)
    return pd.DataFrame(recs)


def screen_features(table, feature_cols, target="delta_recall"):
    rows = []
    d = table[target].astype(float)
    for f in feature_cols:
        x = table[f].astype(float)
        ok = x.notna() & d.notna()
        if ok.sum() < 5:
            rows.append({"feature": f, "spearman": np.nan, "n": int(ok.sum())}); continue
        sr, sp = spearmanr(x[ok], d[ok]); kt, kp = kendalltau(x[ok], d[ok])
        rows.append({"feature": f, "spearman": round(sr,4), "spearman_p": round(sp,4),
                     "kendall": round(kt,4), "n": int(ok.sum())})
    return pd.DataFrame(rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False)


def _loco_predict(table, feature_cols, target="delta_recall"):
    from sklearn.linear_model import LinearRegression
    classes = table["class"].unique()
    X = table[feature_cols].astype(float)
    valid = X.notna().all(1)
    t = table[valid].reset_index(drop=True)
    X = t[feature_cols].astype(float).values; y = t[target].astype(float).values
    cls_arr = t["class"].values
    preds, actuals, keys = [], [], []
    for held in classes:
        tr = cls_arr != held; te = cls_arr == held
        if tr.sum() < 5 or te.sum() == 0:
            continue
        m = LinearRegression().fit(X[tr], y[tr]); p = m.predict(X[te])
        preds.extend(p); actuals.extend(y[te]); keys.extend([held]*te.sum())
    return np.array(preds), np.array(actuals), keys


def evaluate_predictor(table, feature_sets, target="delta_recall", n_boot=1000, seed=0):
    from sklearn.metrics import r2_score
    rng = np.random.default_rng(seed); results = {}
    for name, cols in feature_sets.items():
        preds, actuals, keys = _loco_predict(table, cols, target)
        if len(preds) < 5:
            results[name] = {"error": "insufficient data"}; continue
        sr, sp = spearmanr(preds, actuals); r2 = r2_score(actuals, preds)
        boot = []
        for _ in range(n_boot):
            bi = rng.choice(len(preds), len(preds), replace=True)
            if len(np.unique(actuals[bi])) > 2:
                boot.append(spearmanr(preds[bi], actuals[bi])[0])
        lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan))
        results[name] = {
            "spearman_pred_vs_actual": round(float(sr),4), "p": round(float(sp),4),
            "spearman_CI": (round(float(lo),4), round(float(hi),4)),
            "r2": round(float(r2),4), "n_pred": len(preds), "n_features": len(cols),
        }
    return results
