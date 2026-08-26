"""Post-G5 utilities for the SABER-IDS benchmark and recovery experiments.

These helpers deliberately separate four objects that were conflated in the
first prototype:

1. single-group causal harm;
2. cost-efficient static ranking;
3. physically realised set selection;
4. recovery-aware deployed risk.

The module contains no dataset-specific paths and does not touch a test split.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    path = Path(path)
    h = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def json_sha256(payload: Mapping[str, object] | Sequence[object]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PrefixCalibration:
    prefix_length: int
    realised_reduction: float
    absolute_error: float
    within_tolerance: bool
    trace: tuple[dict[str, float | int], ...]


def calibrate_prefix_count(
    order: Sequence[object],
    *,
    target_reduction: float,
    build_model: Callable[[int], object],
    realised_reduction: Callable[[object], float],
    tolerance: float = 0.015,
) -> PrefixCalibration:
    """Choose an integer pruning prefix by measured, not nominal, reduction.

    ``build_model(k)`` must return the physically pruned model obtained by
    removing the first ``k`` groups in ``order``. The routine evaluates a
    monotone prefix using binary search and then checks neighbouring integers,
    because structural dependencies can create small non-linear jumps.
    """
    if not 0 < target_reduction < 1:
        raise ValueError("target_reduction must be in (0, 1)")
    if len(order) == 0:
        raise ValueError("order is empty")

    cache: dict[int, float] = {}
    trace: list[dict[str, float | int]] = []

    def evaluate(k: int) -> float:
        k = int(max(0, min(len(order), k)))
        if k not in cache:
            value = float(realised_reduction(build_model(k)))
            cache[k] = value
            trace.append({"prefix_length": k, "realised_reduction": value})
        return cache[k]

    lo, hi = 0, len(order)
    evaluate(lo)
    if evaluate(hi) < target_reduction:
        best = hi
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if evaluate(mid) >= target_reduction:
                hi = mid
            else:
                lo = mid + 1
        candidates = {lo, max(0, lo - 1), min(len(order), lo + 1)}
        best = min(candidates, key=lambda k: abs(evaluate(k) - target_reduction))

    realised = evaluate(best)
    error = abs(realised - target_reduction)
    return PrefixCalibration(
        prefix_length=int(best),
        realised_reduction=float(realised),
        absolute_error=float(error),
        within_tolerance=bool(error <= tolerance),
        trace=tuple(sorted(trace, key=lambda x: int(x["prefix_length"]))),
    )


def lexicographic_checkpoint_index(
    history: pd.DataFrame,
    *,
    constraints: Mapping[str, tuple[str, float]],
    primary_minimize: Sequence[str] = ("val_awbir", "val_hsr_balanced_soc"),
    secondary_maximize: Sequence[str] = ("val_family_macro_f1", "val_macro_f1"),
) -> int:
    """Select a checkpoint using feasibility before aggregate performance.

    ``constraints`` maps a column to (``"max"`` or ``"min"``, threshold).
    Feasible epochs are ranked by semantic risk, then family/fine macro-F1.
    If no epoch is feasible, the epoch with the smallest normalized violation
    is chosen before the same semantic tie-breaks.
    """
    if history.empty:
        raise ValueError("history is empty")
    table = history.copy().reset_index(drop=True)
    violations = np.zeros(len(table), dtype=float)
    feasible = np.ones(len(table), dtype=bool)
    for col, (direction, threshold) in constraints.items():
        if col not in table:
            raise KeyError(f"Missing constraint column: {col}")
        values = table[col].to_numpy(dtype=float)
        scale = max(abs(float(threshold)), 1e-6)
        if direction == "max":
            raw = np.maximum(values - float(threshold), 0.0)
        elif direction == "min":
            raw = np.maximum(float(threshold) - values, 0.0)
        else:
            raise ValueError("constraint direction must be 'max' or 'min'")
        violations += raw / scale
        feasible &= raw <= 0
    table["_feasible"] = feasible
    table["_violation"] = violations

    pool = table[table["_feasible"]].copy()
    if pool.empty:
        best_violation = float(table["_violation"].min())
        pool = table[np.isclose(table["_violation"], best_violation)].copy()

    sort_cols: list[str] = []
    ascending: list[bool] = []
    for col in primary_minimize:
        if col in pool:
            sort_cols.append(col)
            ascending.append(True)
    for col in secondary_maximize:
        if col in pool:
            sort_cols.append(col)
            ascending.append(False)
    if not sort_cols:
        raise ValueError("No requested checkpoint-selection columns are present")
    return int(pool.sort_values(sort_cols, ascending=ascending).index[0])


def stratified_spearman_bootstrap(
    table: pd.DataFrame,
    *,
    score_a: str,
    score_b: str,
    outcomes: Iterable[str],
    strata: str = "module_path",
    n_bootstrap: int = 5000,
    seed: int = 20260826,
) -> pd.DataFrame:
    """Bootstrap the difference in rank correlation within structural strata."""
    rng = np.random.default_rng(seed)
    groups = {key: frame.reset_index(drop=True) for key, frame in table.groupby(strata)}
    rows: list[dict[str, object]] = []
    for outcome in outcomes:
        rho_a = float(spearmanr(table[score_a], table[outcome]).correlation)
        rho_b = float(spearmanr(table[score_b], table[outcome]).correlation)
        draws: list[float] = []
        for _ in range(n_bootstrap):
            parts = []
            for frame in groups.values():
                idx = rng.integers(0, len(frame), size=len(frame))
                parts.append(frame.iloc[idx])
            boot = pd.concat(parts, ignore_index=True)
            a = spearmanr(boot[score_a], boot[outcome]).correlation
            b = spearmanr(boot[score_b], boot[outcome]).correlation
            draws.append(float(a - b))
        arr = np.asarray(draws, dtype=float)
        ci_low, ci_high = np.nanquantile(arr, [0.025, 0.975])
        p_two = min(1.0, 2.0 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0))))
        rows.append(
            {
                "outcome": outcome,
                "n_groups": int(len(table)),
                "rho_score_a": rho_a,
                "rho_score_b": rho_b,
                "difference": rho_a - rho_b,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "bootstrap_two_sided_p": p_two,
                "n_bootstrap": int(n_bootstrap),
                "strata": strata,
            }
        )
    return pd.DataFrame(rows)


def additive_set_summary(
    selected_group_ids: Sequence[str],
    single_group_table: pd.DataFrame,
    *,
    group_id_column: str = "group_id",
    harm_columns: Sequence[str] = (
        "harm_awbir",
        "harm_hsr_balanced_soc",
        "harm_fine_macro_f1",
        "harm_family_macro_f1",
    ),
) -> dict[str, float | int]:
    lookup = single_group_table.set_index(group_id_column)
    missing = sorted(set(selected_group_ids).difference(lookup.index))
    if missing:
        raise KeyError(f"Unknown group ids (first 10): {missing[:10]}")
    subset = lookup.loc[list(selected_group_ids)]
    out: dict[str, float | int] = {"n_removed": int(len(subset))}
    for column in harm_columns:
        if column in subset:
            out[f"sum_{column}"] = float(subset[column].sum())
            out[f"mean_{column}"] = float(subset[column].mean())
    return out
