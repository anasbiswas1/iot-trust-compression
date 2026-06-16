
"""
src/train.py — training + evaluation core (thin notebooks call these).

Tempered class weights (sqrt-inverse-frequency) by design: a GENTLE rare-class
boost. Full inverse-frequency or focal loss would artificially prop up the rare
classes and MASK the per-class collapse this paper exists to measure. We want
rare classes honestly fragile at baseline so the collapse-under-compression
effect is real and the diagnostic has something to predict.

Leakage discipline: scaler is fit on TRAIN ONLY. Device auto-detected.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score

from .config import CFG, PATHS, set_all_seeds
from . import models as _models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
META = ("label", "group", "_part", "_row")


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in META]


def tempered_class_weights(y_enc: np.ndarray, n_classes: int, temper: float = 0.5) -> torch.Tensor:
    counts = np.bincount(y_enc, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = (1.0 / counts) ** temper          # sqrt-inverse-frequency (temper=0.5)
    w = w / w.mean()                       # normalise around 1
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def make_tensors(df, splits, feat_cols, le: LabelEncoder, scaler: StandardScaler):
    out = {}
    for name, idx in splits.items():
        if name not in ("train", "val", "test"):
            continue
        sub = df.loc[idx]
        X = scaler.transform(sub[feat_cols].to_numpy(np.float32))
        y = le.transform(sub["label"].to_numpy())
        out[name] = (torch.tensor(X, dtype=torch.float32),
                     torch.tensor(y, dtype=torch.long))
    return out


@torch.no_grad()
def _forward_batched(model, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Run a forward pass in batches so a 500k-row split can't OOM the GPU.
    Returns logits on CPU."""
    model.eval()
    outs = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(DEVICE)
        outs.append(model(xb).cpu())
    return torch.cat(outs, 0)


def train_model(arch, df, dataset, splits, seed, *, epochs=40, patience=6,
                batch_size=4096, lr=1e-3, compression="M0", arch_kwargs=None,
                save=True, verbose=True):
    """Train one anchor under the given (frozen) split. Saves M0 to Drive.
    Returns (model, info) where info has params, macroF1_val/test, label_encoder, scaler."""
    set_all_seeds(seed)
    feat_cols = feature_columns(df)
    le = LabelEncoder().fit(df["label"].to_numpy())
    n_classes = len(le.classes_)

    scaler = StandardScaler().fit(df.loc[splits["train"], feat_cols].to_numpy(np.float32))
    t = make_tensors(df, splits, feat_cols, le, scaler)
    Xtr, ytr = t["train"]; Xva, yva = t["val"]

    model = _models.build(arch, len(feat_cols), n_classes, **(arch_kwargs or {})).to(DEVICE)
    w = tempered_class_weights(ytr.numpy(), n_classes)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    best_f1, best_state, since = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        # val macro-F1 (BATCHED forward — never push the whole split at once)
        vp = _forward_batched(model, Xva).argmax(1).numpy()
        f1 = f1_score(yva.numpy(), vp, average="macro")
        if verbose:
            print(f"  epoch {ep:02d}  val_macroF1={f1:.4f}")
        if f1 > best_f1 + 1e-4:
            best_f1, best_state, since = f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            since += 1
            if since >= patience:
                if verbose: print(f"  early stop @ epoch {ep}")
                break
    model.load_state_dict(best_state)

    # test macro-F1 (batched)
    Xte, yte = t["test"]
    tp = _forward_batched(model, Xte).argmax(1).numpy()
    test_f1 = f1_score(yte.numpy(), tp, average="macro")

    if save:
        path = PATHS.model(dataset, arch, compression, seed)
        torch.save({"state_dict": model.state_dict(),
                    "classes": list(le.classes_),
                    "feat_cols": feat_cols,
                    "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_},
                   path)
        if verbose: print(f"  saved -> {path}")

    info = {"arch": arch, "params": model.num_params(), "n_classes": n_classes,
            "macroF1_val": float(best_f1), "macroF1_test": float(test_f1),
            "label_encoder": le, "scaler": scaler, "feat_cols": feat_cols, "seed": seed}
    return model, info


@torch.no_grad()
def predict(model, df, splits, le, scaler, feat_cols, which="test"):
    sub = df.loc[splits[which]]
    X = torch.tensor(scaler.transform(sub[feat_cols].to_numpy(np.float32)), dtype=torch.float32)
    logits = _forward_batched(model, X)          # BATCHED — safe on 500k rows
    probs = torch.softmax(logits, 1).numpy()
    y_true = le.transform(sub["label"].to_numpy())
    y_pred = probs.argmax(1)
    return y_true, y_pred, probs


def per_class_recall_table(y_true, y_pred, le) -> pd.DataFrame:
    from .metrics import per_class_recall
    rec = per_class_recall(y_true, y_pred)
    rows = [{"label": le.classes_[c], "recall": rec[c],
             "support": int((y_true == c).sum())} for c in sorted(rec)]
    return pd.DataFrame(rows).sort_values("support")


def seed_null_band(arch, df, dataset, splits, seeds, **kw) -> pd.DataFrame:
    """Train at multiple seeds; per-class recall mean/std across seeds = the NULL band.
    A 'collapse' under compression is only real if it exceeds this band."""
    recs = {}
    for s in seeds:
        m, info = train_model(arch, df, dataset, splits, s, save=(s == seeds[0]),
                              compression="M0", verbose=False, **kw)
        yt, yp, _ = predict(m, df, splits, info["label_encoder"], info["scaler"], info["feat_cols"])
        tab = per_class_recall_table(yt, yp, info["label_encoder"]).set_index("label")["recall"]
        recs[s] = tab
    R = pd.DataFrame(recs)
    R["mean"] = R.mean(1); R["std"] = R.std(1)
    return R
