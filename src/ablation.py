
"""
src/ablation.py — D4: rarity-vs-separability ablation.
Subsample a class's TRAIN rows to a target (rare) count, retrain, prune80, measure if
it collapses. Run for a SEPARABLE class and a CONFUSABLE class at the SAME count:
  both collapse -> rarity sufficient; separable resists -> separability matters beyond rarity.
Test set never subsampled.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .config import CFG, PATHS
from . import train as _train
from . import compression as _comp


def subsample_class_in_train(df, splits, class_name, target_count, seed=0):
    rng = np.random.default_rng(seed)
    train_idx = np.array(splits["train"])
    labels = df.loc[train_idx, "label"].to_numpy()
    keep_mask = labels != class_name
    cls_pos = np.where(labels == class_name)[0]
    if len(cls_pos) > target_count:
        chosen = rng.choice(cls_pos, target_count, replace=False)
        keep_mask[chosen] = True
    else:
        keep_mask[cls_pos] = True
    new_train = train_idx[keep_mask]
    return {**splits, "train": list(new_train)}


def run_ablation_for_class(df, splits, dataset, class_name, target_count, *,
                           arch="cnn1d", arch_kwargs=None, seed=0,
                           epochs=40, patience=6, ft_epochs=8):
    sp = subsample_class_in_train(df, splits, class_name, target_count, seed=seed)
    n_kept = int((df.loc[sp["train"], "label"] == class_name).sum())
    model, info = _train.train_model(arch, df, dataset, sp, seed, epochs=epochs,
                                     patience=patience, arch_kwargs=arch_kwargs,
                                     save=False, verbose=False)
    le, scaler, feat = info["label_encoder"], info["scaler"], info["feat_cols"]
    yt, yp, _ = _train.predict(model, df, sp, le, scaler, feat)
    r0 = _train.per_class_recall_table(yt, yp, le).set_index("label")["recall"]
    p80, _, _ = _comp.prune_and_finetune(model, df, dataset, sp, seed, 0.80,
                                         ft_epochs=ft_epochs, arch=arch, verbose=False)
    ytp, ypp, _ = _train.predict(p80, df, sp, le, scaler, feat)
    r8 = _train.per_class_recall_table(ytp, ypp, le).set_index("label")["recall"]
    return {
        "class": class_name, "kept_train_count": n_kept, "target_count": target_count,
        "macroF1_M0": round(info["macroF1_test"], 4),
        "macroF1_prune80": round(f1_score(ytp, ypp, average="macro"), 4),
        "class_recall_M0": round(float(r0.get(class_name, np.nan)), 4),
        "class_recall_prune80": round(float(r8.get(class_name, np.nan)), 4),
        "class_delta": round(float(r8.get(class_name, np.nan) - r0.get(class_name, np.nan)), 4),
    }
