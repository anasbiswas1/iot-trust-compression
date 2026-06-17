
"""
src/explain.py — Stage 2 (EXPLAIN) mechanism analysis.

Core hypothesis: the classes that collapse under pruning are the ones whose
BASELINE (M0) representation geometry was already most fragile (closest to
Minority Collapse). Pruning, capacity-starved, sacrifices them first.

Two questions, both answered here:
  (a) PREDICTIVE: does M0 per-class geometry predict prune80 Δrecall?
  (b) MECHANISTIC: does pruning push the collapsed classes' geometry further?

Honesty controls (prereg-mandated):
  * geometry is noisy at low class-n -> bootstrap CIs on every per-class measure
  * each rare class paired with a SAME-N frequent-class control, so 'fragile
    geometry' can't be a small-sample artifact
  * features taken from .features() (penultimate), the neural-collapse layer
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from .config import CFG, PATHS
from . import geometry as geom
from . import train as _train

DEVICE = _train.DEVICE


@torch.no_grad()
def extract_features(model, df, splits, scaler, feat_cols, le, which="test",
                     batch_size=8192, is_half=False):
    """Penultimate representations + logits + labels for a split. Batched (no OOM)."""
    model = model.to(DEVICE).eval()
    sub = df.loc[splits[which]]
    X = scaler.transform(sub[feat_cols].to_numpy(np.float32))
    y = le.transform(sub["label"].to_numpy())
    feats, logits = [], []
    dt = torch.float16 if is_half else torch.float32
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i+batch_size], dtype=dt).to(DEVICE)
        feats.append(model.features(xb).float().cpu().numpy())
        logits.append(model(xb).float().cpu().numpy())
    return np.concatenate(feats), np.concatenate(logits), y


def per_class_geometry(feats, logits, y, le, n_boot=200, seed=0, max_per_class=4000):
    """Per-class geometry with bootstrap CIs. Returns a DataFrame indexed by class.
    Columns: mean_cos_to_others (collapse: higher=worse), min_angular_sep_deg,
    etf_gap, margin, plus *_lo/_hi bootstrap bounds. Caps per-class sample for speed."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    rows = {}
    etf = geom.etf_deviation(feats, y)
    marg = geom.per_class_margin(logits, y)
    for c in classes:
        rows[int(c)] = {
            "mean_cos_to_others": etf[c]["mean_cos_to_others"],
            "min_angular_sep_deg": etf[c]["min_angular_sep_deg"],
            "etf_gap": etf[c]["etf_gap"],
            "margin": marg[c],
            "support": int((y == c).sum()),
        }
    boot = {int(c): {"mean_cos_to_others": [], "margin": []} for c in classes}
    for b in range(n_boot):
        idx_parts = []
        for c in classes:
            ci = np.where(y == c)[0]
            take = min(len(ci), max_per_class)
            idx_parts.append(rng.choice(ci, take, replace=True))
        bidx = np.concatenate(idx_parts)
        bf, bl, by = feats[bidx], logits[bidx], y[bidx]
        be = geom.etf_deviation(bf, by)
        bm = geom.per_class_margin(bl, by)
        for c in classes:
            boot[int(c)]["mean_cos_to_others"].append(be[c]["mean_cos_to_others"])
            boot[int(c)]["margin"].append(bm[c])
    for c in classes:
        for key in ("mean_cos_to_others", "margin"):
            arr = np.array(boot[int(c)][key])
            rows[int(c)][f"{key}_lo"] = float(np.percentile(arr, 2.5))
            rows[int(c)][f"{key}_hi"] = float(np.percentile(arr, 97.5))
    out = pd.DataFrame(rows).T
    out.index = [le.classes_[int(c)] for c in out.index]
    return out


def same_n_control(feats, y, le, rare_class_name, frequent_class_name, n_boot=200, seed=0):
    """Subsample the FREQUENT class to the rare class's n, recompute its collapse
    geometry. If the rare class's fragility persists vs the same-n frequent control,
    it's real, not a small-sample artifact. Returns (rare_cos, freq_cos_same_n_CI)."""
    rng = np.random.default_rng(seed)
    name_to_idx = {n: i for i, n in enumerate(le.classes_)}
    rc, fc = name_to_idx[rare_class_name], name_to_idx[frequent_class_name]
    n_rare = int((y == rc).sum())
    rare_cos = geom.etf_deviation(feats, y)[rc]["mean_cos_to_others"]
    fi = np.where(y == fc)[0]
    cos_samples = []
    others = feats[y != fc]; others_y = y[y != fc]
    for b in range(n_boot):
        sub = rng.choice(fi, min(n_rare, len(fi)), replace=True)
        ff = np.concatenate([feats[sub], others]); fy = np.concatenate([np.full(len(sub), fc), others_y])
        cos_samples.append(geom.etf_deviation(ff, fy)[fc]["mean_cos_to_others"])
    return rare_cos, (float(np.percentile(cos_samples, 2.5)), float(np.percentile(cos_samples, 97.5)))


def correlate_geometry_vs_collapse(geom_df, delta_recall_series, geom_col="mean_cos_to_others"):
    """The headline test: does baseline geometry predict prune80 collapse?
    Spearman + Kendall (rank, robust) between a geometry measure and Δrecall.
    Positive cos<->negative Δrecall => fragile geometry predicts collapse."""
    from scipy.stats import spearmanr, kendalltau
    common = geom_df.index.intersection(delta_recall_series.index)
    g = geom_df.loc[common, geom_col].astype(float)
    d = delta_recall_series.loc[common].astype(float)
    sr, sp = spearmanr(g, d)
    kt, kp = kendalltau(g, d)
    return {"n": len(common), "spearman_r": float(sr), "spearman_p": float(sp),
            "kendall_tau": float(kt), "kendall_p": float(kp),
            "interpretation": "fragile baseline geometry predicts collapse" if sr < 0 else "no/positive relationship"}


def confusability_reallocation(model_M0, model_comp, df, splits, scaler, feat_cols, le,
                               collapsed_classes, is_half_comp=False, which="test"):
    """Test the CONFUSABILITY mechanism: when a class collapses under compression,
    where do its samples go? For each collapsed class, find which OTHER class
    absorbs the most of its (now-misclassified) test samples, and whether that
    absorber's recall RISES. If collapse pairs with a sibling's gain, the
    mechanism is winner-take-all reallocation among confusable classes."""
    import numpy as np, pandas as pd
    fM0, lM0, y = extract_features(model_M0, df, splits, scaler, feat_cols, le, which=which)
    fC, lC, _ = extract_features(model_comp, df, splits, scaler, feat_cols, le,
                                 which=which, is_half=is_half_comp)
    predM0 = lM0.argmax(1); predC = lC.argmax(1)
    name = lambda i: le.classes_[int(i)]
    idx = {n: i for i, n in enumerate(le.classes_)}
    def recall(pred):
        return {int(c): float((pred[y == c] == c).mean()) if (y == c).any() else np.nan
                for c in np.unique(y)}
    rM0, rC = recall(predM0), recall(predC)
    rows = []
    for cls in collapsed_classes:
        c = idx[cls]
        mask = (y == c)
        comp_pred = predC[mask]
        wrong = comp_pred[comp_pred != c]
        if len(wrong) == 0:
            rows.append({"collapsed_class": cls, "top_absorber": None}); continue
        vals, counts = np.unique(wrong, return_counts=True)
        top = vals[counts.argmax()]
        frac = counts.max() / mask.sum()
        rows.append({
            "collapsed_class": cls,
            "M0_recall": round(rM0[c], 3),
            "comp_recall": round(rC[c], 3),
            "top_absorber": name(top),
            "absorber_took_frac": round(float(frac), 3),
            "absorber_recall_M0": round(rM0[int(top)], 3),
            "absorber_recall_comp": round(rC[int(top)], 3),
            "absorber_recall_gain": round(rC[int(top)] - rM0[int(top)], 3),
        })
    return pd.DataFrame(rows).sort_values("absorber_took_frac", ascending=False)


def rank_collapse_analysis(model_M0, model_comp, df, splits, scaler, feat_cols, le,
                           is_half_comp=False, which="test", max_n=60000, seed=0):
    """Test whether compression collapses the REPRESENTATION rank (the entanglement
    story). Computes effective rank of the penultimate representation for M0 and the
    compressed model, GLOBALLY and per-class. If sink-consolidation = rank collapse,
    the compressed model's global + per-class effective rank drops sharply."""
    import numpy as np, pandas as pd
    rng = np.random.default_rng(seed)
    fM0, lM0, y = extract_features(model_M0, df, splits, scaler, feat_cols, le, which=which)
    fC, lC, _ = extract_features(model_comp, df, splits, scaler, feat_cols, le,
                                 which=which, is_half=is_half_comp)
    if len(y) > max_n:
        idx = rng.choice(len(y), max_n, replace=False)
        fM0s, fCs, ys = fM0[idx], fC[idx], y[idx]
    else:
        fM0s, fCs, ys = fM0, fC, y
    glob = {
        "M0_effrank": geom.effective_rank(fM0s),
        "comp_effrank": geom.effective_rank(fCs),
        "M0_dim": fM0s.shape[1],
    }
    glob["rank_retained_frac"] = round(glob["comp_effrank"] / glob["M0_effrank"], 3)
    rows = []
    for c in np.unique(ys):
        m = ys == c
        if m.sum() < 20:
            continue
        rows.append({
            "label": le.classes_[int(c)],
            "support": int(m.sum()),
            "M0_class_effrank": round(geom.effective_rank(fM0s[m]), 3),
            "comp_class_effrank": round(geom.effective_rank(fCs[m]), 3),
        })
    pc = pd.DataFrame(rows)
    pc["rank_change"] = (pc["comp_class_effrank"] - pc["M0_class_effrank"]).round(3)
    return glob, pc.sort_values("support")
