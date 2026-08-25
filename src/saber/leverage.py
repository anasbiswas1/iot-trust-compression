
"""Baseline group saliency and Robust Semantic Boundary Leverage (R-SBL)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import kendalltau, spearmanr

from .surgery import get_module


def _unpack_batch(batch):
    """Return (features, targets, optional_environment) for tuple/dict batches."""
    if isinstance(batch, Mapping):
        x = batch.get("x", batch.get("features", batch.get("inputs")))
        y = batch.get("y", batch.get("label", batch.get("labels", batch.get("target"))))
        env = batch.get("environment", batch.get("env", batch.get("group")))
        if x is None or y is None:
            raise KeyError("Dictionary batch must expose features/inputs and labels/target.")
        return x, y, env
    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError("Tuple batch must contain at least (features, targets).")
        env = batch[2] if len(batch) > 2 else None
        return batch[0], batch[1], env
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def _group_rows(groups: pd.DataFrame) -> list[dict[str, object]]:
    required = {"group_id", "module_path", "channel_index"}
    missing = required.difference(groups.columns)
    if missing:
        raise ValueError(f"Group table missing columns: {sorted(missing)}")
    return groups.to_dict(orient="records")


def magnitude_scores(model: nn.Module, groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in _group_rows(groups):
        module = get_module(model, str(group["module_path"]))
        if not isinstance(module, nn.Conv1d):
            continue
        idx = int(group["channel_index"])
        score = float(module.weight[idx].detach().abs().sum().cpu())
        if module.bias is not None:
            score += float(module.bias[idx].detach().abs().cpu())
        rows.append({"group_id": group["group_id"], "magnitude": score})
    return pd.DataFrame(rows)


def gradient_saliency_scores(
    model: nn.Module,
    loader,
    groups: pd.DataFrame,
    *,
    device: torch.device | str,
    max_batches: int = 20,
    criterion: nn.Module | None = None,
    input_transform=None,
) -> pd.DataFrame:
    """Compute first-order Taylor and Fisher-style channel scores."""
    criterion = criterion or nn.CrossEntropyLoss()
    device = torch.device(device)
    model = model.to(device)
    training = model.training
    model.eval()

    group_records = _group_rows(groups)
    accum_taylor = {str(g["group_id"]): 0.0 for g in group_records}
    accum_fisher = {str(g["group_id"]): 0.0 for g in group_records}
    seen = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x, y, _ = _unpack_batch(batch)
        x = x.to(device)
        if input_transform is not None:
            x = input_transform(x)
        y = y.to(device).long()
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        batch_n = int(len(y))
        seen += batch_n

        for group in group_records:
            module = get_module(model, str(group["module_path"]))
            if not isinstance(module, nn.Conv1d) or module.weight.grad is None:
                continue
            idx = int(group["channel_index"])
            product = module.weight.grad[idx] * module.weight[idx]
            taylor = float(product.detach().abs().sum().cpu())
            fisher = float(product.detach().pow(2).sum().cpu())
            if module.bias is not None and module.bias.grad is not None:
                bprod = module.bias.grad[idx] * module.bias[idx]
                taylor += float(bprod.detach().abs().cpu())
                fisher += float(bprod.detach().pow(2).cpu())
            gid = str(group["group_id"])
            accum_taylor[gid] += taylor * batch_n
            accum_fisher[gid] += fisher * batch_n

    model.zero_grad(set_to_none=True)
    model.train(training)
    denom = max(seen, 1)
    rows = [
        {
            "group_id": gid,
            "taylor": accum_taylor[gid] / denom,
            "fisher": accum_fisher[gid] / denom,
        }
        for gid in accum_taylor
    ]
    return pd.DataFrame(rows)


def _collect_source_examples(
    loader,
    source_indices: set[int],
    *,
    device: torch.device,
    max_samples_per_class: int,
    input_transform=None,
) -> dict[int, torch.Tensor]:
    buckets: dict[int, list[torch.Tensor]] = {c: [] for c in source_indices}
    counts = {c: 0 for c in source_indices}
    for batch in loader:
        x, y, _ = _unpack_batch(batch)
        y_np = y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)
        for c in source_indices:
            need = max_samples_per_class - counts[c]
            if need <= 0:
                continue
            idx = np.flatnonzero(y_np == c)[:need]
            if len(idx):
                selected = x[idx].detach().cpu()
                if input_transform is not None:
                    selected = input_transform(selected)
                buckets[c].append(selected)
                counts[c] += len(idx)
        if all(v >= max_samples_per_class for v in counts.values()):
            break
    out = {}
    for c, chunks in buckets.items():
        if chunks:
            out[c] = torch.cat(chunks, dim=0)[:max_samples_per_class].to(device)
    return out


def semantic_boundary_leverage(
    model: nn.Module,
    loader,
    groups: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    device: torch.device | str,
    max_samples_per_class: int = 512,
    edge_weight_column: str = "robust_weight",
    normalize_by_group_size: float = 0.5,
    normalize_by_flops: float = 0.5,
    input_transform=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute directed pairwise margin leverage for every channel group.

    One backward pass is performed per directed edge. The method intentionally
    works on a capped validation subset and records the cap in the output.
    """
    if edge_weight_column not in edges.columns:
        if "normalized_weight" in edges.columns:
            edge_weight_column = "normalized_weight"
        else:
            raise ValueError(f"No usable edge weight column in {list(edges.columns)}")

    device = torch.device(device)
    model = model.to(device)
    training = model.training
    model.eval()
    group_records = _group_rows(groups)
    source_indices = {int(v) for v in edges["source_index"].unique()}
    source_examples = _collect_source_examples(
        loader,
        source_indices,
        device=device,
        max_samples_per_class=max_samples_per_class,
        input_transform=input_transform,
    )

    aggregate = {str(g["group_id"]): 0.0 for g in group_records}
    edge_rows: list[dict[str, object]] = []

    for edge_id, edge in edges.reset_index(drop=True).iterrows():
        c = int(edge["source_index"])
        a = int(edge["target_index"])
        x = source_examples.get(c)
        if x is None or len(x) == 0:
            continue
        weight = float(edge[edge_weight_column])
        model.zero_grad(set_to_none=True)
        logits = model(x)
        margin = logits[:, c] - logits[:, a]
        objective = margin.mean()
        objective.backward()

        for group in group_records:
            module = get_module(model, str(group["module_path"]))
            if not isinstance(module, nn.Conv1d) or module.weight.grad is None:
                continue
            idx = int(group["channel_index"])
            product = module.weight.grad[idx] * module.weight[idx]
            raw = float(product.detach().abs().sum().cpu())
            if module.bias is not None and module.bias.grad is not None:
                raw += float(
                    (module.bias.grad[idx] * module.bias[idx]).detach().abs().cpu()
                )
            weighted = raw * weight
            aggregate[str(group["group_id"])] += weighted
            edge_rows.append(
                {
                    "edge_id": int(edge_id),
                    "source_index": c,
                    "target_index": a,
                    "source_class": edge.get("source_class", str(c)),
                    "target_class": edge.get("target_class", str(a)),
                    "group_id": group["group_id"],
                    "module_path": group["module_path"],
                    "channel_index": int(group["channel_index"]),
                    "edge_weight": weight,
                    "n_examples": int(len(x)),
                    "teacher_margin_mean": float(margin.detach().mean().cpu()),
                    "raw_boundary_leverage": raw,
                    "weighted_boundary_leverage": weighted,
                }
            )

    model.zero_grad(set_to_none=True)
    model.train(training)

    scores = groups.copy()
    scores["sbl_raw"] = scores["group_id"].map(aggregate).fillna(0.0)
    param = np.maximum(scores.get("parameter_cost", 1).astype(float).to_numpy(), 1.0)
    flops = np.maximum(scores.get("flops_cost", 1.0).astype(float).to_numpy(), 1.0)
    denom = np.power(param, normalize_by_group_size) * np.power(
        flops, normalize_by_flops
    )
    scores["sbl"] = scores["sbl_raw"].to_numpy() / denom
    scores["sbl_param_alpha"] = normalize_by_group_size
    scores["sbl_flops_beta"] = normalize_by_flops
    scores["sbl_max_samples_per_class"] = max_samples_per_class
    return scores, pd.DataFrame(edge_rows)


def merge_score_tables(
    groups: pd.DataFrame, *score_tables: pd.DataFrame
) -> pd.DataFrame:
    out = groups.copy()
    for table in score_tables:
        if table is None or table.empty:
            continue
        cols = [c for c in table.columns if c == "group_id" or c not in out.columns]
        out = out.merge(table[cols], on="group_id", how="left")
    return out


def robust_profile_aggregation(
    score_tables: Mapping[str, pd.DataFrame],
    score_column: str = "sbl",
    *,
    method: str = "cvar",
    q: float = 0.75,
) -> pd.DataFrame:
    """Aggregate group scores across operational cost profiles."""
    frames = []
    for profile, table in score_tables.items():
        temp = table[["group_id", score_column]].copy()
        temp["cost_profile"] = profile
        frames.append(temp)
    long = pd.concat(frames, ignore_index=True)
    rows = []
    for gid, group in long.groupby("group_id"):
        values = group[score_column].to_numpy(dtype=float)
        if method == "mean":
            value = float(values.mean())
        elif method == "max":
            value = float(values.max())
        elif method == "cvar":
            cutoff = np.quantile(values, q)
            tail = values[values >= cutoff]
            value = float(tail.mean()) if len(tail) else float(values.max())
        else:
            raise ValueError("method must be mean, max, or cvar")
        rows.append(
            {
                "group_id": gid,
                "r_sbl": value,
                "profile_min": float(values.min()),
                "profile_mean": float(values.mean()),
                "profile_max": float(values.max()),
                "profile_aggregation": method,
                "profile_cvar_q": q if method == "cvar" else np.nan,
            }
        )
    return pd.DataFrame(rows)


def validate_score_against_harm(
    table: pd.DataFrame,
    score_columns: Sequence[str],
    harm_columns: Sequence[str],
    *,
    top_k_fraction: float = 0.20,
) -> pd.DataFrame:
    """Rank-correlation and top-k retrieval for the G1 score gate.

    Importance scores are expected to be *large for important groups*. Harm from
    removing a group must also be encoded as a larger-is-worse positive number.
    """
    rows = []
    n = len(table)
    k = max(1, int(math.ceil(n * top_k_fraction)))
    for score in score_columns:
        for harm in harm_columns:
            subset = table[[score, harm]].dropna()
            if len(subset) < 3:
                continue
            rho = spearmanr(subset[score], subset[harm]).statistic
            tau = kendalltau(subset[score], subset[harm]).statistic
            predicted_top = set(subset.nlargest(k, score).index)
            actual_top = set(subset.nlargest(k, harm).index)
            precision = len(predicted_top & actual_top) / k
            rows.append(
                {
                    "score": score,
                    "harm": harm,
                    "n_groups": int(len(subset)),
                    "spearman": float(rho),
                    "kendall": float(tau),
                    "top_k": int(k),
                    "top_k_precision": float(precision),
                }
            )
    return pd.DataFrame(rows)
