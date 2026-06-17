
"""
src/shuffle_experiment.py — Test 3: is the CNN's pruning fragility FEATURE-ORDER
dependent (=> convolution-over-non-local-tabular-features is the root cause),
while the MLP is order-invariant (control)?
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from .config import CFG, PATHS
from . import train as _train
from . import compression as _comp
from . import explain as _explain
from . import geometry as _geom


def permute_features(df, feat_cols, seed):
    """Return a copy of df with feature columns reordered by a fixed permutation.
    Renames to feat_0..feat_N so column NAMES are identical, DATA order differs."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(feat_cols))
    new_order = [feat_cols[i] for i in perm]
    meta = [c for c in df.columns if c not in feat_cols]
    out = df[new_order + meta].copy()
    rename = {new_order[i]: f"feat_{i}" for i in range(len(new_order))}
    out = out.rename(columns=rename)
    return out, [f"feat_{i}" for i in range(len(new_order))], perm.tolist()


def run_one_ordering(arch, df, splits, seed, order_seed, arch_kwargs, *,
                     collapse_thresh=0.15, ft_epochs=8, epochs=40, patience=6):
    """Train arch on ONE feature ordering, prune80, return collapse summary."""
    feat_cols = _train.feature_columns(df)
    dfp, pcols, perm = permute_features(df, feat_cols, order_seed)

    model, info = _train.train_model(arch, dfp, "ciciot2023_shuf", splits, seed,
                                     epochs=epochs, patience=patience,
                                     arch_kwargs=arch_kwargs, save=False, verbose=False)
    le, scaler = info["label_encoder"], info["scaler"]

    yt, yp, _ = _train.predict(model, dfp, splits, le, scaler, pcols)
    r0 = _train.per_class_recall_table(yt, yp, le).set_index("label")["recall"]

    p80, _le, _sc = _comp.prune_and_finetune(model, dfp, "ciciot2023_shuf", splits, seed,
                                             0.80, ft_epochs=ft_epochs, arch=arch, verbose=False)
    ytp, ypp, _ = _train.predict(p80, dfp, splits, le, scaler, pcols)
    r8 = _train.per_class_recall_table(ytp, ypp, le).set_index("label")["recall"]
    macro8 = f1_score(ytp, ypp, average="macro")
    macro0 = info["macroF1_test"]

    delta = (r8 - r0)
    collapsed = set(delta.index[delta < -collapse_thresh])

    gC, _ = _explain.rank_collapse_analysis(model, p80, dfp, splits, scaler, pcols, le, max_n=40000)

    return {
        "order_seed": order_seed,
        "macroF1_M0": round(macro0, 4),
        "macroF1_prune80": round(macro8, 4),
        "n_collapsed": len(collapsed),
        "collapsed_set": collapsed,
        "rank_retained": gC["rank_retained_frac"],
        "delta_recall": delta,
    }


def feature_order_sensitivity(arch, df, splits, seed, order_seeds, arch_kwargs, **kw):
    """Run multiple orderings; quantify how much the collapse pattern VARIES.
    Returns (summary_df, jaccard_stability, collapsed_union, runs)."""
    runs = [run_one_ordering(arch, df, splits, seed, os_, arch_kwargs, **kw) for os_ in order_seeds]
    rows = [{k: v for k, v in r.items() if k not in ("collapsed_set", "delta_recall")} for r in runs]
    summ = pd.DataFrame(rows)
    sets = [r["collapsed_set"] for r in runs]
    jac = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]; inter = sets[i] & sets[j]
            jac.append(len(inter) / len(u) if u else 1.0)
    jaccard_stability = float(np.mean(jac)) if jac else 1.0
    union = set().union(*sets) if sets else set()
    return summ, jaccard_stability, union, runs
