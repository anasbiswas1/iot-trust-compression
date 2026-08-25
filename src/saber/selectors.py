
"""Budgeted group selection for physically realised structured pruning."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def select_groups_greedy(
    groups: pd.DataFrame,
    *,
    importance_column: str,
    cost_column: str = "flops_cost",
    target_reduction_fraction: float,
    minimum_remaining_per_layer: int = 4,
    protect_group_ids: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select the least valuable groups until a realised-cost target is reached.

    Importance is interpreted as "large means retain". Selection ranks by
    importance per unit cost, while enforcing a minimum surviving width in each
    layer.
    """
    required = {
        "group_id",
        "module_path",
        "channel_index",
        importance_column,
        cost_column,
    }
    missing = required.difference(groups.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not 0 < target_reduction_fraction < 1:
        raise ValueError("target_reduction_fraction must be in (0,1)")
    protect = set(protect_group_ids or [])

    table = groups.copy()
    table[importance_column] = table[importance_column].astype(float)
    table[cost_column] = table[cost_column].astype(float).clip(lower=0.0)
    if table[cost_column].sum() <= 0:
        raise ValueError("Total removable cost is zero")

    table["selection_priority"] = table[importance_column] / np.maximum(
        table[cost_column], 1e-12
    )
    table["protected"] = table["group_id"].isin(protect)

    total_cost = float(table[cost_column].sum())
    target = target_reduction_fraction * total_cost
    layer_total = table.groupby("module_path")["group_id"].count().to_dict()
    layer_removed = defaultdict(int)
    selected: list[int] = []
    removed_cost = 0.0

    candidates = table.sort_values(
        ["protected", "selection_priority", importance_column],
        ascending=[True, True, True],
    )
    for idx, row in candidates.iterrows():
        if bool(row["protected"]):
            continue
        layer = str(row["module_path"])
        remaining_after = layer_total[layer] - layer_removed[layer] - 1
        if remaining_after < minimum_remaining_per_layer:
            continue
        selected.append(idx)
        layer_removed[layer] += 1
        removed_cost += float(row[cost_column])
        if removed_cost >= target:
            break

    out = table.copy()
    out["selected_for_pruning"] = out.index.isin(selected)
    out["selection_order"] = np.nan
    for order, idx in enumerate(selected, start=1):
        out.loc[idx, "selection_order"] = order

    achieved = removed_cost / total_cost
    summary = {
        "importance_column": importance_column,
        "cost_column": cost_column,
        "target_reduction_fraction": float(target_reduction_fraction),
        "achieved_reduction_fraction": float(achieved),
        "selected_groups": int(len(selected)),
        "total_groups": int(len(table)),
        "minimum_remaining_per_layer": int(minimum_remaining_per_layer),
        "removed_cost": float(removed_cost),
        "total_removable_cost": float(total_cost),
        "target_reached": bool(removed_cost >= target),
        "per_layer_removed": dict(layer_removed),
    }
    return out.reset_index(drop=True), summary


def selection_to_prune_map(selection: pd.DataFrame) -> dict[str, list[int]]:
    required = {"module_path", "channel_index", "selected_for_pruning"}
    missing = required.difference(selection.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    chosen = selection[selection["selected_for_pruning"].astype(bool)]
    out: dict[str, list[int]] = {}
    for layer, group in chosen.groupby("module_path"):
        out[str(layer)] = sorted(group["channel_index"].astype(int).tolist())
    return out


def summarize_layer_widths(selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer, group in selection.groupby("module_path"):
        total = len(group)
        removed = int(group["selected_for_pruning"].astype(bool).sum())
        rows.append(
            {
                "module_path": layer,
                "original_channels": total,
                "removed_channels": removed,
                "remaining_channels": total - removed,
                "removed_fraction": removed / total if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def pareto_nondominated(
    table: pd.DataFrame,
    *,
    maximize: Sequence[str] = (),
    minimize: Sequence[str] = (),
) -> pd.Series:
    """Return a boolean mask for non-dominated rows."""
    if not maximize and not minimize:
        raise ValueError("Specify at least one objective")
    data = table.reset_index(drop=True)
    keep = np.ones(len(data), dtype=bool)
    for i in range(len(data)):
        if not keep[i]:
            continue
        for j in range(len(data)):
            if i == j:
                continue
            no_worse = True
            strictly_better = False
            for col in maximize:
                if data.loc[j, col] < data.loc[i, col]:
                    no_worse = False
                    break
                strictly_better |= data.loc[j, col] > data.loc[i, col]
            if not no_worse:
                continue
            for col in minimize:
                if data.loc[j, col] > data.loc[i, col]:
                    no_worse = False
                    break
                strictly_better |= data.loc[j, col] < data.loc[i, col]
            if no_worse and strictly_better:
                keep[i] = False
                break
    return pd.Series(keep, index=table.index)
