"""
src/crux.py — the spine (D1). Information-loss vs decision-rule-loss.

Freeze the compressed model's penultimate representation; train a one-vs-rest
linear probe per class; compare its AUC to the same probe on the M0 representation.
Probe survives  => information present, collapse is a DECISION-LAYER artifact.
Probe collapses => information genuinely LOST.
Pre-commit: the result is GRADED and class-dependent, not a clean binary.
Run under the PRIMARY grouped split (a probe on leakage-trained features reads
memorised near-duplicates and fakes 'information survived').
"""
from __future__ import annotations
import numpy as np


def probe_recoverability(feats_anchor: np.ndarray, feats_comp: np.ndarray,
                         labels: np.ndarray, seed: int = 0) -> dict:
    """Per-class one-vs-rest linear-probe AUC on anchor vs compressed features.
    Returns {class: {auc_anchor, auc_comp, retention}}. retention<threshold flags
    'information lost' — but report the continuum regardless (see config.crux)."""
    raise NotImplementedError(
        "Fit logistic-regression probes (frozen features) per class; report AUC + retention."
    )
