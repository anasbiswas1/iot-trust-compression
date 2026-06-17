
"""
src/mitigate.py — Stage 4: predictor-FREE, measurement-guided decision-layer recovery.
The crux showed information SURVIVES under compression, so protection is DECISION-LAYER.
Recover prune80-collapsed classes by adjusting the decision rule on the FROZEN compressed
model. Temperature = negative control (can't move argmax). Per-class bias = the cheap fix.
ALL params fit on VAL, evaluated on TEST. Report the trade-off frontier, not a point.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .config import CFG, PATHS
from . import train as _train


def _logits_for_split(model, df, splits, scaler, feat_cols, le, which):
    import torch
    sub = df.loc[splits[which]]
    X = scaler.transform(sub[feat_cols].to_numpy(np.float32))
    y = le.transform(sub["label"].to_numpy())
    logits = _train._forward_batched(model, torch.tensor(X, dtype=torch.float32)).numpy()
    return logits, y


def _per_class_recall(logits, y, bias=None):
    pred = (logits + bias).argmax(1) if bias is not None else logits.argmax(1)
    classes = np.unique(y)
    return {int(c): float((pred[y == c] == c).mean()) for c in classes}, pred


def temperature_control(logits_te, y_te, T_grid=(0.5, 1.0, 2.0, 5.0)):
    rows = []
    for T in T_grid:
        pred = (logits_te / T).argmax(1)
        rows.append({"T": T, "macroF1": round(f1_score(y_te, pred, average="macro"), 4),
                     "accuracy": round(float((pred == y_te).mean()), 4)})
    return pd.DataFrame(rows)


def fit_per_class_bias(logits_val, y_val, target_classes, strength,
                       n_classes, max_bias=20.0, step=0.25):
    bias = np.zeros(n_classes)
    for c in target_classes:
        best_b, best_r = 0.0, _per_class_recall(logits_val, y_val, bias)[0].get(c, 0.0)
        b = 0.0
        while b < max_bias * strength:
            b += step
            trial = bias.copy(); trial[c] = b
            r = _per_class_recall(logits_val, y_val, trial)[0].get(c, 0.0)
            if r > best_r + 1e-4:
                best_r, best_b = r, b
            elif r < best_r - 0.02:
                break
        bias[c] = best_b
    return bias


def recovery_metrics(logits, y, bias, target_classes, le, benign_idx=None):
    base_rec, base_pred = _per_class_recall(logits, y, None)
    new_rec, new_pred = _per_class_recall(logits, y, bias)
    base_f1 = f1_score(y, base_pred, average="macro")
    new_f1 = f1_score(y, new_pred, average="macro")
    fpr_change = np.nan
    if benign_idx is not None:
        bm = (y == benign_idx)
        if bm.any():
            base_benign_recall = float((base_pred[bm] == benign_idx).mean())
            new_benign_recall = float((new_pred[bm] == benign_idx).mean())
            fpr_change = round(base_benign_recall - new_benign_recall, 4)
    target_gain = {le.classes_[c]: round(new_rec[c] - base_rec[c], 4) for c in target_classes}
    others = [c for c in base_rec if c not in target_classes]
    other_cost = {le.classes_[c]: round(new_rec[c] - base_rec[c], 4)
                  for c in others if abs(new_rec[c] - base_rec[c]) > 0.01}
    return {
        "macroF1_base": round(base_f1, 4), "macroF1_new": round(new_f1, 4),
        "macroF1_delta": round(new_f1 - base_f1, 4),
        "mean_target_recovery": round(np.mean(list(target_gain.values())), 4) if target_gain else 0.0,
        "benign_recall_drop_as_FPR": fpr_change,
        "target_gain": target_gain, "other_class_cost": other_cost,
    }


def frontier(model_comp, df, splits, scaler, feat_cols, le, target_classes,
             strengths=(0.0, 0.25, 0.5, 0.75, 1.0), benign_name="BenignTraffic"):
    Lval, yval = _logits_for_split(model_comp, df, splits, scaler, feat_cols, le, "val")
    Lte, yte = _logits_for_split(model_comp, df, splits, scaler, feat_cols, le, "test")
    n_classes = Lte.shape[1]
    tgt_idx = [int(np.where(le.classes_ == c)[0][0]) for c in target_classes if c in le.classes_]
    benign_idx = int(np.where(le.classes_ == benign_name)[0][0]) if benign_name in le.classes_ else None
    rows = []
    for s in strengths:
        bias = fit_per_class_bias(Lval, yval, tgt_idx, s, n_classes) if s > 0 else np.zeros(n_classes)
        m = recovery_metrics(Lte, yte, bias, tgt_idx, le, benign_idx)
        rows.append({"strength": s, "mean_target_recovery": m["mean_target_recovery"],
                     "macroF1_test": m["macroF1_new"], "macroF1_delta": m["macroF1_delta"],
                     "benign_FPR_proxy": m["benign_recall_drop_as_FPR"],
                     "n_other_classes_hurt": len(m["other_class_cost"])})
    return pd.DataFrame(rows), (Lval, yval, Lte, yte, tgt_idx, benign_idx)


def fit_joint_bias_macroF1(logits_val, y_val, n_classes, *, n_restarts=3, iters=400, seed=0):
    """Coordinate-ascent per-class bias maximizing VAL macro-F1 (not greedy per-target)."""
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    classes = np.unique(y_val)
    def macro(bias):
        return f1_score(y_val, (logits_val + bias).argmax(1), average="macro")
    best_bias, best_f1 = np.zeros(n_classes), macro(np.zeros(n_classes))
    for r in range(n_restarts):
        bias = (rng.normal(0, 0.5, n_classes) if r > 0 else np.zeros(n_classes))
        cur = macro(bias)
        for it in range(iters):
            c = int(rng.choice(classes)); improved = False
            for delta in (0.5, -0.5, 0.25, -0.25, 1.0, -1.0):
                trial = bias.copy(); trial[c] += delta
                f = macro(trial)
                if f > cur + 1e-5:
                    bias, cur = trial, f; improved = True; break
        if cur > best_f1:
            best_bias, best_f1 = bias.copy(), cur
    return best_bias, best_f1


def refit_head(model_comp, df, splits, scaler, feat_cols, le, *, epochs=15, lr=1e-2,
               batch_size=4096, seed=0):
    """Re-fit ONLY a fresh linear head on the FROZEN compressed representation."""
    import torch, torch.nn as nn
    from . import explain as _explain
    from torch.utils.data import DataLoader, TensorDataset
    _train.set_all_seeds(seed)
    def feats(which):
        return _explain.extract_features(model_comp, df, splits, scaler, feat_cols, le, which=which)
    ftr, _, ytr = feats("train"); fva, _, yva = feats("val"); fte, _, yte = feats("test")
    d = ftr.shape[1]; K = len(le.classes_)
    head = nn.Linear(d, K).to(_train.DEVICE)
    w = _train.tempered_class_weights(ytr, K)
    crit = nn.CrossEntropyLoss(weight=w); opt = torch.optim.Adam(head.parameters(), lr=lr)
    Xtr = torch.tensor(ftr, dtype=torch.float32); Ytr = torch.tensor(ytr, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)
    for ep in range(epochs):
        head.train()
        for xb, yb in loader:
            xb, yb = xb.to(_train.DEVICE), yb.to(_train.DEVICE)
            opt.zero_grad(); crit(head(xb), yb).backward(); opt.step()
    head.eval()
    with torch.no_grad():
        Lva = head(torch.tensor(fva, dtype=torch.float32).to(_train.DEVICE)).cpu().numpy()
        Lte = head(torch.tensor(fte, dtype=torch.float32).to(_train.DEVICE)).cpu().numpy()
    return Lva, yva, Lte, yte


def pairwise_vs_ovr_separability(model_comp, model_M0, df, splits, scaler, feat_cols, le,
                                 class_pairs, which="test", max_n=40000, seed=0):
    """OvR probe AUC vs PAIRWISE (collapsed-vs-absorber) probe AUC, M0 vs compressed.
    OvR survives + pairwise collapses => explains zero-sum mitigation."""
    import numpy as np, pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from . import explain as _explain
    rng = np.random.default_rng(seed)
    fM0, _, y = _explain.extract_features(model_M0, df, splits, scaler, feat_cols, le, which=which)
    fC, _, _ = _explain.extract_features(model_comp, df, splits, scaler, feat_cols, le, which=which)
    if len(y) > max_n:
        idx = rng.choice(len(y), max_n, replace=False); fM0, fC, y = fM0[idx], fC[idx], y[idx]
    name2i = {n: i for i, n in enumerate(le.classes_)}
    def ovr_auc(F, c):
        yb = (y == c).astype(int)
        if yb.sum() < 5: return np.nan
        tr, te = train_test_split(np.arange(len(y)), test_size=0.4, random_state=seed, stratify=yb)
        clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(F[tr], yb[tr])
        return roc_auc_score(yb[te], clf.predict_proba(F[te])[:, 1])
    def pair_auc(F, c, a):
        m = (y == c) | (y == a)
        if (y == c).sum() < 5 or (y == a).sum() < 5: return np.nan
        Fm, ym = F[m], (y[m] == c).astype(int)
        tr, te = train_test_split(np.arange(len(ym)), test_size=0.4, random_state=seed, stratify=ym)
        clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(Fm[tr], ym[tr])
        return roc_auc_score(ym[te], clf.predict_proba(Fm[te])[:, 1])
    rows = []
    for collapsed, absorber in class_pairs:
        c, a = name2i[collapsed], name2i[absorber]
        rows.append({"collapsed": collapsed, "absorber": absorber,
            "ovr_auc_M0": round(ovr_auc(fM0, c), 4), "ovr_auc_comp": round(ovr_auc(fC, c), 4),
            "pair_auc_M0": round(pair_auc(fM0, c, a), 4), "pair_auc_comp": round(pair_auc(fC, c, a), 4)})
    out = pd.DataFrame(rows)
    out["ovr_drop"] = (out["ovr_auc_M0"] - out["ovr_auc_comp"]).round(4)
    out["pair_drop"] = (out["pair_auc_M0"] - out["pair_auc_comp"]).round(4)
    return out
