"""
src/compression.py — the compression matrix, PyTorch-native.

float16 and magnitude pruning are implemented. int8 PTQ is scaffolded with the
PRE-REGISTERED transformer fallback: if int8 on the FT-Transformer won't yield a
clean, attribution-accessible model, run int8 on MLP/CNN only and report the
transformer-int8 cell as a documented limitation.

Every returned object must still expose .features() and .forward() so the probe /
KernelSHAP / geometry code reads compressed models identically to baselines.
"""
from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from .config import CFG


def to_float16(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model).half()


def magnitude_prune(model: nn.Module, amount: float) -> nn.Module:
    """L1 unstructured pruning to `amount` sparsity, then fine-tune (caller).
    NOTE: magnitude is class-blind — it will happily delete low-magnitude but
    rare-class-critical units. That is itself a Stage 2 mechanism to test."""
    m = copy.deepcopy(model)
    for module in m.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")   # make sparsity permanent
    return m


def distill(student: nn.Module, teacher: nn.Module, loader, **kw) -> nn.Module:
    """Knowledge distillation. For the trust-preserving variant, the loss can
    up-weight rare-class logit/boundary matching (Stage 4, if information was lost)."""
    raise NotImplementedError("Implement KD train loop (T, alpha; optional rare-class weighting).")


def to_int8(model: nn.Module, calib_loader, arch: str):
    """int8 PTQ via torchao / FX-graph PTQ. Returns (model_or_None, note).
    Honour the pre-registered fallback for the transformer."""
    if arch == "ft_transformer" and CFG["compression"]["int8_transformer_fallback"]:
        # Attempt; if the quantized transformer is not attribution-accessible,
        # return (None, "transformer int8 skipped — documented limitation").
        pass
    raise NotImplementedError("Implement torchao/FX int8 PTQ; keep .features() accessible.")


def apply(model, name: str, arch: str, *, loader=None, calib_loader=None, teacher=None):
    """Dispatch one compression-matrix cell by name."""
    if name == "M0":
        return model
    if name == "float16":
        return to_float16(model)
    if name == "prune50":
        return magnitude_prune(model, 0.50)
    if name == "prune80":
        return magnitude_prune(model, 0.80)
    if name == "distillation":
        return distill(model, teacher, loader)
    if name == "int8":
        m, _note = to_int8(model, calib_loader, arch)
        return m
    raise ValueError(f"unknown compression cell: {name}")
