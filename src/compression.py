
"""
src/compression.py — the compression matrix, PyTorch-native.

Matrix: M0, prune50, prune80, distillation, int8, float16.
Two FAMILIES, and the split is load-bearing for the EXPLAIN stage:
  * post-training, NO retrain: float16, int8  (pure information loss)
  * retrain/finetune-based:    prune50, prune80, distillation  (loss + re-optimisation)
Comparing families is how we dissociate "information lost" from "boundary re-fit".

Every returned object exposes .features() and .forward() so the probe / KernelSHAP /
geometry code reads compressed models identically to the M0 anchor.

int8 carries the PRE-REGISTERED fallback: dynamic quantisation targets Linear layers;
on architectures where it won't yield a clean attribution-accessible model we report
the cell as a documented limitation rather than contorting the pipeline.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset

from .config import CFG, PATHS
from . import models as _models
from . import train as _train

DEVICE = _train.DEVICE


# ----------------------------------------------------------------------
# post-training family (no retrain)
# ----------------------------------------------------------------------
def to_float16(model: nn.Module) -> nn.Module:
    """Half precision. Forward expects .half() inputs (the eval helper handles it)."""
    return copy.deepcopy(model).half()


def to_int8(model: nn.Module, arch: str):
    """Dynamic int8 quantisation of Linear layers (post-training, no retrain).
    Returns (quantised_model_or_None, note). Dynamic PTQ runs on CPU.
    Conv1d is not dynamically quantisable; for CNN/transformer this quantises the
    Linear head (+ any Linear blocks) and is reported honestly as partial."""
    m = copy.deepcopy(model).cpu().eval()
    try:
        q = torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
        n_lin = sum(isinstance(mod, nn.Linear) for mod in model.modules())
        note = f"int8 dynamic on {n_lin} Linear layer(s); conv/other layers fp32 (partial, reported)"
        return q, note
    except Exception as e:
        return None, f"int8 skipped for {arch}: {type(e).__name__} {e}"


# ----------------------------------------------------------------------
# retrain family (prune-then-finetune, distillation)
# ----------------------------------------------------------------------
def _magnitude_prune(model: nn.Module, amount: float) -> nn.Module:
    """L1 unstructured prune to `amount` sparsity, made permanent. Class-BLIND by
    design — it deletes low-magnitude weights regardless of which class they serve,
    which is itself a Stage-2 mechanism (does it preferentially hurt rare classes?)."""
    m = copy.deepcopy(model)
    for module in m.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return m


def _reapply_sparsity_mask(model: nn.Module):
    """After fine-tuning a pruned model, zero out the weights that were pruned so
    the sparsity LEVEL is preserved (fine-tune updates would otherwise refill zeros).
    Returns a hook list; call .remove() to detach."""
    masks, hooks = {}, []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            masks[module] = (module.weight != 0).float()
    def make_hook(mask):
        def hook(grad):
            return grad * mask
        return hook
    for module, mask in masks.items():
        hooks.append(module.weight.register_hook(make_hook(mask)))
    return hooks, masks


def prune_and_finetune(anchor, df, dataset, splits, seed, amount, *,
                       ft_epochs=8, batch_size=4096, lr=5e-4, arch="cnn1d",
                       verbose=False):
    """Prune the anchor to `amount` sparsity then fine-tune (gradient-masked so the
    sparsity level holds). Returns the fine-tuned pruned model on DEVICE."""
    _train.set_all_seeds(seed)
    feat_cols = _train.feature_columns(df)
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    le = LabelEncoder().fit(df["label"].to_numpy())
    scaler = StandardScaler().fit(df.loc[splits["train"], feat_cols].to_numpy(np.float32))
    t = _train.make_tensors(df, splits, feat_cols, le, scaler)
    Xtr, ytr = t["train"]

    model = _magnitude_prune(anchor, amount).to(DEVICE)
    hooks, _ = _reapply_sparsity_mask(model)        # keep pruned weights at zero during FT
    w = _train.tempered_class_weights(ytr.numpy(), len(le.classes_))
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    for ep in range(ft_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        if verbose:
            print(f"    prune{int(amount*100)} ft epoch {ep}")
    for h in hooks:
        h.remove()
    # final hard mask so saved sparsity is exact
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                module.weight.mul_((module.weight != 0).float())
    return model, le, scaler


def distill(df, dataset, splits, seed, teacher, *, student_kwargs=None,
            T=3.0, alpha=0.5, epochs=12, batch_size=4096, lr=1e-3, arch="cnn1d",
            verbose=False):
    """Train a SMALLER student against the teacher's softened logits (+ true labels).
    student_kwargs sizes the student down from the teacher. Retrain family."""
    _train.set_all_seeds(seed)
    feat_cols = _train.feature_columns(df)
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    le = LabelEncoder().fit(df["label"].to_numpy())
    scaler = StandardScaler().fit(df.loc[splits["train"], feat_cols].to_numpy(np.float32))
    t = _train.make_tensors(df, splits, feat_cols, le, scaler)
    Xtr, ytr = t["train"]

    student = _models.build(arch, len(feat_cols), len(le.classes_),
                            **(student_kwargs or {"channels": (32, 64)})).to(DEVICE)
    teacher = teacher.to(DEVICE).eval()
    w = _train.tempered_class_weights(ytr.numpy(), len(le.classes_))
    hard = nn.CrossEntropyLoss(weight=w)
    kl = nn.KLDivLoss(reduction="batchmean")
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    for ep in range(epochs):
        student.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.no_grad():
                tlog = teacher(xb)
            slog = student(xb)
            soft = kl(torch.log_softmax(slog / T, 1), torch.softmax(tlog / T, 1)) * (T * T)
            loss = alpha * soft + (1 - alpha) * hard(slog, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if verbose:
            print(f"    distill epoch {ep}")
    return student, le, scaler


# ----------------------------------------------------------------------
# unified entry point + per-class evaluation across the matrix
# ----------------------------------------------------------------------
def model_size_report(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(int((p != 0).sum()) for p in model.parameters())
    return {"params": total, "nonzero_params": nonzero,
            "sparsity": round(1 - nonzero / max(total, 1), 4)}


def build_matrix(anchor, df, dataset, splits, seed, *, arch="cnn1d",
                 anchor_kwargs=None, cells=None, verbose=True):
    """Produce every compression-matrix cell from the M0 anchor.
    Returns {cell_name: {"model":..., "note":..., "size":..., "is_half":bool, "is_int8":bool}}.
    distillation student defaults to ~1/3 the anchor width."""
    cells = cells or CFG["compression"]["matrix"]
    out = {}
    for cell in cells:
        if verbose: print(f"[{cell}]")
        if cell == "M0":
            m = anchor; note = "uncompressed anchor"; half = i8 = False
        elif cell == "float16":
            m = to_float16(anchor); note = "fp16 .half()"; half, i8 = True, False
        elif cell == "int8":
            m, note = to_int8(anchor, arch); half, i8 = False, (m is not None)
        elif cell.startswith("prune"):
            amt = int(cell.replace("prune", "")) / 100.0
            m, _le, _sc = prune_and_finetune(anchor, df, dataset, splits, seed, amt,
                                             arch=arch, verbose=verbose)
            note = f"L1 prune {int(amt*100)}% + finetune"; half = i8 = False
        elif cell == "distillation":
            sk = {"channels": (24, 48)}   # smaller student than the (64,128) anchor
            m, _le, _sc = distill(df, dataset, splits, seed, anchor,
                                  student_kwargs=sk, arch=arch, verbose=verbose)
            note = "KD student (24,48) <- anchor"; half = i8 = False
        else:
            raise ValueError(cell)
        size = model_size_report(m) if m is not None else {}
        out[cell] = {"model": m, "note": note, "size": size, "is_half": half, "is_int8": i8}
        if verbose and m is not None:
            print(f"   {note} | {size}")
    return out


@torch.no_grad()
def evaluate_cell(entry, df, splits, le, scaler, feat_cols, which="test"):
    """Per-class recall + overall macro-F1 for one compression cell, handling
    fp16/int8 device quirks. Returns (per_class_recall_dict, macro_f1, probs, y_true, y_pred)."""
    from sklearn.metrics import f1_score
    m = entry["model"]
    if m is None:
        return None, None, None, None, None
    sub = df.loc[splits[which]]
    X = scaler.transform(sub[feat_cols].to_numpy(np.float32))
    y_true = le.transform(sub["label"].to_numpy())

    if entry["is_int8"]:
        m = m.cpu().eval(); Xt = torch.tensor(X, dtype=torch.float32)
        logits = torch.cat([m(Xt[i:i+8192]) for i in range(0, len(Xt), 8192)], 0)
    elif entry["is_half"]:
        m = m.to(DEVICE).eval(); Xt = torch.tensor(X, dtype=torch.float16)
        logits = torch.cat([m(Xt[i:i+8192].to(DEVICE)).float().cpu()
                            for i in range(0, len(Xt), 8192)], 0)
    else:
        m = m.to(DEVICE).eval(); Xt = torch.tensor(X, dtype=torch.float32)
        logits = _train._forward_batched(m, Xt)

    probs = torch.softmax(logits, 1).numpy()
    y_pred = probs.argmax(1)
    from .metrics import per_class_recall
    rec = per_class_recall(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    return rec, float(macro), probs, y_true, y_pred
