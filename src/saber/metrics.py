
"""Operational, hierarchical, calibration, and boundary-risk metrics."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)

from .taxonomy import CostProfile, LabelTaxonomy


def _as_numpy(x: object) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def class_recall(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    n_classes: int,
) -> np.ndarray:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_pred, dtype=np.int64)
    out = np.full(n_classes, np.nan, dtype=np.float64)
    for c in range(n_classes):
        mask = y == c
        if mask.any():
            out[c] = float(np.mean(p[mask] == c))
    return out


def semantic_decomposition(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    taxonomy: LabelTaxonomy,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_pred, dtype=np.int64)
    if y.shape != p.shape:
        raise ValueError("y_true and y_pred must have equal shape")
    if y.ndim != 1:
        raise ValueError("Targets must be one-dimensional")

    types = np.asarray(
        [taxonomy.transition_type(int(t), int(q)) for t, q in zip(y, p)],
        dtype=object,
    )
    attack_mask = y != taxonomy.benign_index
    benign_mask = ~attack_mask

    y_binary = taxonomy.binary_targets(y)
    p_binary = taxonomy.binary_targets(p)
    y_family = taxonomy.family_targets(y)
    p_family = taxonomy.family_targets(p)

    attack_to_benign = (
        float(np.mean(types[attack_mask] == "attack_to_benign"))
        if attack_mask.any()
        else np.nan
    )
    benign_to_attack = (
        float(np.mean(types[benign_mask] == "benign_to_attack"))
        if benign_mask.any()
        else np.nan
    )
    attack_types = types[attack_mask]
    same_family = (
        float(np.mean(attack_types == "same_family_attack"))
        if attack_mask.any()
        else np.nan
    )
    cross_family = (
        float(np.mean(attack_types == "cross_family_attack"))
        if attack_mask.any()
        else np.nan
    )
    exact_attack = (
        float(np.mean(y[attack_mask] == p[attack_mask]))
        if attack_mask.any()
        else np.nan
    )

    return {
        "n": int(len(y)),
        "fine_accuracy": float(accuracy_score(y, p)),
        "fine_macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "binary_accuracy": float(accuracy_score(y_binary, p_binary)),
        "binary_balanced_accuracy": float(
            balanced_accuracy_score(y_binary, p_binary)
        ),
        "binary_attack_recall": float(
            recall_score(y_binary, p_binary, pos_label=1, zero_division=0)
        ),
        "binary_attack_f1": float(
            f1_score(y_binary, p_binary, pos_label=1, zero_division=0)
        ),
        "family_accuracy": float(accuracy_score(y_family, p_family)),
        "family_macro_f1": float(
            f1_score(y_family, p_family, average="macro", zero_division=0)
        ),
        "exact_attack_type_rate": exact_attack,
        "attack_to_benign_rate": attack_to_benign,
        "benign_to_attack_rate": benign_to_attack,
        "same_family_attack_substitution_rate": same_family,
        "cross_family_attack_substitution_rate": cross_family,
    }


def hierarchical_semantic_risk(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    taxonomy: LabelTaxonomy,
    profile: CostProfile,
    *,
    normalize: bool = True,
) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_pred, dtype=np.int64)
    matrix = taxonomy.cost_matrix(profile)
    risks = matrix[y, p]
    value = float(np.mean(risks)) if len(risks) else np.nan
    if not normalize or not np.isfinite(value):
        return value
    max_cost = float(matrix.max())
    return value / max_cost if max_cost > 0 else value


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: Sequence[int] | np.ndarray,
    *,
    n_bins: int = 15,
    adaptive: bool = True,
) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    if probs.ndim != 2 or len(probs) != len(y):
        raise ValueError("probabilities and targets are not aligned")
    confidence = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)

    if adaptive:
        order = np.argsort(confidence)
        bins = np.array_split(order, n_bins)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins = [
            np.where((confidence > edges[i]) & (confidence <= edges[i + 1]))[0]
            for i in range(n_bins)
        ]
    ece = 0.0
    n = max(len(y), 1)
    for idx in bins:
        if len(idx) == 0:
            continue
        ece += len(idx) / n * abs(float(correct[idx].mean() - confidence[idx].mean()))
    return float(ece)


def multiclass_brier(
    probabilities: np.ndarray,
    targets: Sequence[int] | np.ndarray,
) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def negative_log_likelihood(
    probabilities: np.ndarray,
    targets: Sequence[int] | np.ndarray,
    eps: float = 1e-12,
) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], eps, 1.0))))


def cvar(values: Sequence[float] | np.ndarray, q: float = 0.8) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    if not 0 < q <= 1:
        raise ValueError("q must be in (0,1]")
    cutoff = np.quantile(arr, q)
    tail = arr[arr >= cutoff]
    return float(tail.mean()) if len(tail) else float(arr.max())


def recall_regret(
    teacher_y_pred: Sequence[int] | np.ndarray,
    student_y_pred: Sequence[int] | np.ndarray,
    y_true: Sequence[int] | np.ndarray,
    n_classes: int,
) -> np.ndarray:
    teacher = class_recall(y_true, teacher_y_pred, n_classes)
    student = class_recall(y_true, student_y_pred, n_classes)
    return teacher - student


def worst_class_regret(
    teacher_y_pred: Sequence[int] | np.ndarray,
    student_y_pred: Sequence[int] | np.ndarray,
    y_true: Sequence[int] | np.ndarray,
    n_classes: int,
) -> float:
    regret = recall_regret(teacher_y_pred, student_y_pred, y_true, n_classes)
    return float(np.nanmax(regret))


def cvar_class_regret(
    teacher_y_pred: Sequence[int] | np.ndarray,
    student_y_pred: Sequence[int] | np.ndarray,
    y_true: Sequence[int] | np.ndarray,
    n_classes: int,
    q: float = 0.8,
) -> float:
    return cvar(
        recall_regret(teacher_y_pred, student_y_pred, y_true, n_classes), q=q
    )


def action_weighted_boundary_inversion_rate(
    teacher_logits: np.ndarray,
    student_logits: np.ndarray,
    targets: Sequence[int] | np.ndarray,
    edges: pd.DataFrame,
    *,
    weight_column: str = "robust_weight",
) -> tuple[float, pd.DataFrame]:
    """Compute AWBIR and an edge-level audit table.

    Only teacher-positive pairwise margins are included in the denominator. This
    avoids calling a student "inverted" when the teacher itself did not place
    the true source above the edge target.
    """
    t = np.asarray(teacher_logits, dtype=np.float64)
    s = np.asarray(student_logits, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    if t.shape != s.shape or t.ndim != 2 or len(y) != len(t):
        raise ValueError("teacher_logits, student_logits, and targets must align")
    if weight_column not in edges.columns:
        if "normalized_weight" in edges.columns:
            weight_column = "normalized_weight"
        else:
            raise ValueError(f"No {weight_column!r} or normalized_weight column")

    rows: list[dict[str, float | int | str]] = []
    numerator = 0.0
    denominator = 0.0

    for _, edge in edges.iterrows():
        c = int(edge["source_index"])
        a = int(edge["target_index"])
        mask = y == c
        if not mask.any():
            continue
        teacher_margin = t[mask, c] - t[mask, a]
        student_margin = s[mask, c] - s[mask, a]
        eligible = teacher_margin > 0
        n_eligible = int(eligible.sum())
        weight = float(edge[weight_column])
        if n_eligible == 0:
            inversion = np.nan
            weighted = 0.0
        else:
            inversion = float(np.mean(student_margin[eligible] <= 0))
            weighted = weight * inversion
            numerator += weighted
            denominator += weight
        rows.append(
            {
                "source_index": c,
                "target_index": a,
                "source_class": edge.get("source_class", str(c)),
                "target_class": edge.get("target_class", str(a)),
                "weight": weight,
                "n_source": int(mask.sum()),
                "n_teacher_positive": n_eligible,
                "inversion_rate": inversion,
                "weighted_inversion": weighted,
                "teacher_margin_mean": float(np.mean(teacher_margin)),
                "student_margin_mean": float(np.mean(student_margin)),
                "margin_change": float(np.mean(student_margin - teacher_margin)),
            }
        )

    value = float(numerator / denominator) if denominator > 0 else np.nan
    return value, pd.DataFrame(rows)


def full_model_audit(
    logits: np.ndarray,
    targets: Sequence[int] | np.ndarray,
    taxonomy: LabelTaxonomy,
    cost_profiles: Mapping[str, CostProfile],
) -> dict[str, float]:
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    probs = softmax_np(z)
    pred = z.argmax(axis=1)
    out = semantic_decomposition(y, pred, taxonomy)
    out.update(
        {
            "ece15": expected_calibration_error(probs, y, n_bins=15, adaptive=True),
            "nll": negative_log_likelihood(probs, y),
            "brier": multiclass_brier(probs, y),
        }
    )
    for name, profile in cost_profiles.items():
        out[f"hsr_{name}"] = hierarchical_semantic_risk(
            y, pred, taxonomy, profile, normalize=True
        )
    return out
