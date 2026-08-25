
"""Directed alert-semantic vulnerability graph construction."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence
import json

import numpy as np
import pandas as pd
from scipy.special import softmax

from .taxonomy import CostProfile, LabelTaxonomy


@dataclass(frozen=True)
class RiskEdge:
    source_index: int
    target_index: int
    source_class: str
    target_class: str
    source_family: str
    target_family: str
    transition_type: str
    support: int
    confusion_rate: float
    mean_margin: float
    margin_std: float
    vulnerability_probability: float
    operational_cost: float
    stability: float
    raw_weight: float
    normalized_weight: float
    cost_profile: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _check_inputs(
    logits: np.ndarray,
    targets: Sequence[int] | np.ndarray,
    taxonomy: LabelTaxonomy,
    environments: Sequence[object] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    if z.ndim != 2 or z.shape[1] != taxonomy.n_classes:
        raise ValueError(f"logits must have shape [N,{taxonomy.n_classes}], got {z.shape}")
    if y.ndim != 1 or len(y) != len(z):
        raise ValueError("targets must be a 1-D array aligned with logits")
    if y.size and (y.min() < 0 or y.max() >= taxonomy.n_classes):
        raise ValueError("targets contain an out-of-range class index")
    env = None if environments is None else np.asarray(environments)
    if env is not None and len(env) != len(y):
        raise ValueError("environments must align with targets")
    return z, y, env


def _environment_stability(
    margins: np.ndarray,
    confusion_indicator: np.ndarray,
    env: np.ndarray | None,
    eps: float = 1e-8,
) -> float:
    """Stability in [0,1]; 1 means similar vulnerability across environments."""
    if env is None:
        return 1.0
    values: list[float] = []
    for e in np.unique(env):
        mask = env == e
        if mask.sum() < 3:
            continue
        # Larger values mean more vulnerability.
        score = float(np.exp(-np.clip(margins[mask].mean(), -20, 20)) + confusion_indicator[mask].mean())
        values.append(score)
    if len(values) < 2:
        return 1.0
    arr = np.asarray(values, dtype=np.float64)
    cv = float(arr.std(ddof=1) / (abs(arr.mean()) + eps))
    return float(1.0 / (1.0 + cv))


def build_alert_semantic_vulnerability_graph(
    logits: np.ndarray,
    targets: Sequence[int] | np.ndarray,
    taxonomy: LabelTaxonomy,
    cost_profile: CostProfile,
    *,
    top_k: int = 3,
    temperature: float = 1.0,
    confusion_weight: float = 0.50,
    margin_weight: float = 0.50,
    min_class_support: int = 20,
    environments: Sequence[object] | np.ndarray | None = None,
    include_benign_source: bool = True,
) -> pd.DataFrame:
    """Build a sparse directed graph using validation-only teacher evidence.

    Candidate targets are ranked within each true source class using a convex
    combination of:
      * validation confusion into the target; and
      * softmax-normalized vulnerability from the teacher pairwise margin.

    The final edge weight is multiplied by the directed alert-semantic cost and
    an optional environment-stability factor.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if confusion_weight < 0 or margin_weight < 0:
        raise ValueError("weights must be non-negative")
    if confusion_weight + margin_weight <= 0:
        raise ValueError("At least one evidence weight must be positive")
    norm = confusion_weight + margin_weight
    confusion_weight /= norm
    margin_weight /= norm

    cost_profile.validate()
    z, y, env = _check_inputs(logits, targets, taxonomy, environments)
    pred = z.argmax(axis=1)
    rows: list[dict[str, object]] = []

    for c in range(taxonomy.n_classes):
        if not include_benign_source and c == taxonomy.benign_index:
            continue
        mask = y == c
        n = int(mask.sum())
        if n < min_class_support:
            continue

        zc = z[mask]
        pred_c = pred[mask]
        env_c = None if env is None else env[mask]
        candidates = [a for a in range(taxonomy.n_classes) if a != c]
        mean_margins = np.asarray(
            [float(np.mean(zc[:, c] - zc[:, a])) for a in candidates],
            dtype=np.float64,
        )
        # Low/negative margin is vulnerable. Softmax is stable and sums to one.
        vulnerability = softmax(-mean_margins / temperature)
        confusion = np.asarray(
            [float(np.mean(pred_c == a)) for a in candidates], dtype=np.float64
        )
        combined = margin_weight * vulnerability + confusion_weight * confusion

        candidate_records: list[dict[str, object]] = []
        for local_idx, a in enumerate(candidates):
            margin_values = zc[:, c] - zc[:, a]
            conf_indicator = (pred_c == a).astype(np.float64)
            stability = _environment_stability(
                margins=margin_values,
                confusion_indicator=conf_indicator,
                env=env_c,
            )
            op_cost = taxonomy.transition_cost(c, a, cost_profile)
            raw = float(op_cost * combined[local_idx] * stability)
            candidate_records.append(
                {
                    "source_index": c,
                    "target_index": a,
                    "source_class": taxonomy.class_names[c],
                    "target_class": taxonomy.class_names[a],
                    "source_family": taxonomy.family_of(c),
                    "target_family": taxonomy.family_of(a),
                    "transition_type": taxonomy.transition_type(c, a),
                    "support": n,
                    "confusion_rate": float(confusion[local_idx]),
                    "mean_margin": float(mean_margins[local_idx]),
                    "margin_std": float(np.std(margin_values, ddof=1)) if n > 1 else 0.0,
                    "vulnerability_probability": float(vulnerability[local_idx]),
                    "operational_cost": float(op_cost),
                    "stability": float(stability),
                    "raw_weight": raw,
                    "cost_profile": cost_profile.name,
                }
            )

        # Preserve the highest-risk directed competitors for this source.
        candidate_records.sort(key=lambda r: float(r["raw_weight"]), reverse=True)
        rows.extend(candidate_records[:top_k])

    graph = pd.DataFrame(rows)
    if graph.empty:
        return graph
    total = float(graph["raw_weight"].sum())
    if total <= 0:
        graph["normalized_weight"] = 1.0 / len(graph)
    else:
        graph["normalized_weight"] = graph["raw_weight"] / total

    columns = [
        "source_index",
        "target_index",
        "source_class",
        "target_class",
        "source_family",
        "target_family",
        "transition_type",
        "support",
        "confusion_rate",
        "mean_margin",
        "margin_std",
        "vulnerability_probability",
        "operational_cost",
        "stability",
        "raw_weight",
        "normalized_weight",
        "cost_profile",
    ]
    return graph[columns].sort_values(
        ["source_index", "raw_weight"], ascending=[True, False]
    ).reset_index(drop=True)


def merge_cost_profile_graphs(graphs: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [g.copy() for g in graphs if g is not None and not g.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def aggregate_robust_edge_weights(
    graphs: pd.DataFrame,
    *,
    method: str = "cvar",
    q: float = 0.75,
) -> pd.DataFrame:
    """Aggregate the same directed edge across cost profiles.

    Edges can enter the top-k set under one profile but not another. Missing
    profile-edge combinations are treated as zero rather than ignored; otherwise
    a profile-specific edge would be falsely interpreted as important in every
    operational regime.
    """
    required = {"source_index", "target_index", "normalized_weight", "cost_profile"}
    missing = required.difference(graphs.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if method not in {"mean", "max", "cvar"}:
        raise ValueError("method must be one of: mean, max, cvar")
    if not 0 < q <= 1:
        raise ValueError("q must be in (0,1]")

    profiles = sorted(graphs["cost_profile"].dropna().astype(str).unique())
    if not profiles:
        return pd.DataFrame()

    id_cols = [
        "source_index",
        "target_index",
        "source_class",
        "target_class",
        "source_family",
        "target_family",
        "transition_type",
    ]
    rows: list[dict[str, object]] = []
    for (source, target), group in graphs.groupby(
        ["source_index", "target_index"], sort=False
    ):
        by_profile = (
            group.groupby("cost_profile")["normalized_weight"].max().to_dict()
        )
        vals = np.asarray(
            [float(by_profile.get(profile, 0.0)) for profile in profiles],
            dtype=np.float64,
        )
        if method == "mean":
            robust = float(vals.mean())
        elif method == "max":
            robust = float(vals.max())
        else:
            cutoff = np.quantile(vals, q)
            tail = vals[vals >= cutoff]
            robust = float(tail.mean()) if len(tail) else float(vals.max())
        base = {col: group.iloc[0][col] for col in id_cols if col in group.columns}
        base.update(
            {
                "robust_weight_raw": robust,
                "n_cost_profiles_present": int(len(by_profile)),
                "n_cost_profiles_total": int(len(profiles)),
                "profile_weight_min": float(vals.min()),
                "profile_weight_mean": float(vals.mean()),
                "profile_weight_max": float(vals.max()),
                "aggregation": method,
                "cvar_q": q if method == "cvar" else np.nan,
            }
        )
        for profile, value in zip(profiles, vals):
            base[f"profile_weight__{profile}"] = float(value)
        rows.append(base)

    out = pd.DataFrame(rows)
    denom = float(out["robust_weight_raw"].sum())
    out["robust_weight"] = (
        out["robust_weight_raw"] / denom
        if denom > 0
        else np.full(len(out), 1.0 / max(len(out), 1))
    )
    return out.sort_values("robust_weight", ascending=False).reset_index(drop=True)


def save_graph_bundle(
    directory: str | Path,
    per_profile_graphs: pd.DataFrame,
    robust_graph: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    per_profile_graphs.to_csv(out / "asvg_edges_by_cost_profile.csv", index=False)
    robust_graph.to_csv(out / "asvg_edges_robust.csv", index=False)
    (out / "asvg_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
