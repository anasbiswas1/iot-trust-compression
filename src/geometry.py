"""
src/geometry.py — neural-collapse geometry (v2.1 theoretical layer) + margin + rank.

This is the capacity-allocation account (Minority Collapse, Fang et al. PNAS 2021):
rare-class classifier/mean vectors lose angular separation and collapse toward
each other, so they start with the least margin and fail first under compression.
These quantities are the THEORY-DERIVED predictor features for the diagnostic, and
the test that EARNS the causal claim is that they predict collapse across all three
architectures. All operate on penultimate representations (model.features(x)).

Caution: class-conditional rank/separation is noisy at rare-class sample counts —
always pair with bootstrap CIs and a same-n subsampled frequent-class control.
"""
from __future__ import annotations
import numpy as np


def class_means(feats: np.ndarray, labels: np.ndarray) -> dict:
    return {int(c): feats[labels == c].mean(0) for c in np.unique(labels)}


def etf_deviation(feats: np.ndarray, labels: np.ndarray) -> dict:
    """Per-class deviation of (centred) class-mean directions from the ideal
    simplex-ETF maximal-equiangular configuration. Lower angular separation =>
    closer to Minority Collapse => predicted to fail first.

    Centring is by the GLOBAL SAMPLE mean (the neural-collapse convention),
    which is correct under class imbalance — the mean-of-class-means would be
    biased toward whichever classes are frequent."""
    g = feats.mean(0, keepdims=True)                 # global sample mean (NC convention)
    mus = class_means(feats, labels)
    classes = sorted(mus)
    M = np.stack([mus[c] for c in classes]) - g      # centre class means by global mean
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    cos = M @ M.T                                    # pairwise cosines of centred class means
    k = len(classes)
    ideal = -1.0 / (k - 1)                           # ETF off-diagonal target
    out = {}
    for i, c in enumerate(classes):
        off = np.delete(cos[i], i)
        out[c] = {
            "mean_cos_to_others": float(off.mean()),         # higher => more collapsed
            "min_angular_sep_deg": float(np.degrees(np.arccos(np.clip(off.max(), -1, 1)))),
            "etf_gap": float(np.mean(np.abs(off - ideal))),  # deviation from ideal ETF
        }
    return out


def minority_collapse_index(feats: np.ndarray, labels: np.ndarray) -> float:
    """Global scalar: mean pairwise cosine among the rarest-half class means,
    CENTRED by the global sample mean (consistent with etf_deviation and with
    neural-collapse theory). -> 1 indicates the minority simplex has collapsed.

    Centring matters: without it, the index conflates 'these class means sit
    near the origin' with 'these classes have collapsed onto each other'."""
    g = feats.mean(0)                                # global sample mean
    mus = class_means(feats, labels)
    counts = {c: int((labels == c).sum()) for c in mus}
    rare = sorted(counts, key=counts.get)[: max(2, len(mus) // 2)]
    M = np.stack([mus[c] for c in rare]) - g         # centre by global mean
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    cos = M @ M.T
    iu = np.triu_indices(len(rare), k=1)
    return float(cos[iu].mean())


def effective_rank(feats: np.ndarray) -> float:
    """Effective rank via singular-value (Shannon) entropy of the representation."""
    s = np.linalg.svd(feats - feats.mean(0, keepdims=True), compute_uv=False)
    p = s / (s.sum() + 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def per_class_margin(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Signed margin = true-class logit minus top competitor, per class (mean).
    Compression compresses the logit range; rare low-margin classes cross first."""
    out = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            out[int(c)] = float("nan"); continue
        z = logits[idx]
        true = z[:, c]
        comp = z.copy(); comp[:, c] = -np.inf
        out[int(c)] = float((true - comp.max(1)).mean())
    return out
