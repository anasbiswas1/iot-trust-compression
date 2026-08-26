"""Boundary-aligned differentiable risk constraints for SABER-IDS.

The original prototype used mean attack probability on benign examples as a
surrogate for the argmax benign-to-attack rate. That quantity can increase even
when no class boundary is crossed, and can remain poorly aligned with the
actual false-alert event. The v2 proxies operate on binary and pairwise logit
margins, which are directly tied to boundary crossing.
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from .losses import HierarchyTensors, aggregate_family_probabilities


def _attack_logit(logits: torch.Tensor, benign_index: int) -> torch.Tensor:
    mask = torch.ones(logits.shape[1], dtype=torch.bool, device=logits.device)
    mask[int(benign_index)] = False
    return torch.logsumexp(logits[:, mask], dim=1)


def boundary_aligned_risk_proxies(
    logits: torch.Tensor,
    targets: torch.Tensor,
    hierarchy: HierarchyTensors,
    *,
    temperature: float = 1.0,
    cvar_q: float = 0.80,
) -> dict[str, torch.Tensor]:
    """Return smooth surrogates aligned to binary argmax boundary events."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    hierarchy.to(logits.device)
    targets = targets.long()
    benign = logits[:, hierarchy.benign_index]
    attack = _attack_logit(logits, hierarchy.benign_index)
    attack_mask = targets != hierarchy.benign_index
    benign_mask = ~attack_mask

    # softplus(x) is small when the desired margin is positive and grows when
    # the wrong side approaches or crosses the boundary.
    attack_miss = (
        F.softplus((benign[attack_mask] - attack[attack_mask]) / temperature).mean()
        if torch.any(attack_mask)
        else logits.new_zeros(())
    )
    benign_false_alert = (
        F.softplus((attack[benign_mask] - benign[benign_mask]) / temperature).mean()
        if torch.any(benign_mask)
        else logits.new_zeros(())
    )

    probs = logits.softmax(dim=1)
    family_probs = aggregate_family_probabilities(probs, hierarchy)
    family_targets = hierarchy.class_to_family[targets]
    family_nll = F.nll_loss(torch.log(family_probs.clamp_min(1e-12)), family_targets)

    per_sample_ce = F.cross_entropy(logits, targets, reduction="none")
    class_means = []
    for c in torch.unique(targets):
        class_means.append(per_sample_ce[targets == c].mean())
    if class_means:
        vector = torch.stack(class_means)
        k = max(1, int(torch.ceil(vector.new_tensor((1.0 - cvar_q) * len(vector))).item()))
        class_cvar = torch.topk(vector, k=k, largest=True).values.mean()
    else:
        class_cvar = logits.new_zeros(())

    return {
        "attack_miss": attack_miss,
        "benign_false_alert": benign_false_alert,
        "family_nll": family_nll,
        "class_cvar": class_cvar,
    }


def directed_edge_violation_proxy(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    edge_source: torch.Tensor,
    edge_target: torch.Tensor,
    edge_weight: torch.Tensor,
    *,
    temperature: float = 1.0,
    teacher_positive_only: bool = True,
) -> torch.Tensor:
    """Weighted soft pairwise-boundary violation over ASVG edges.

    When ``teacher_positive_only`` is true, the proxy uses the same eligibility
    population as AWBIR: source examples for which the teacher places the source
    above the edge target.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    total = student_logits.new_zeros(())
    denom = student_logits.new_zeros(())
    for c, a, weight in zip(edge_source, edge_target, edge_weight):
        mask = targets == c
        if teacher_positive_only:
            t_margin = teacher_logits[:, c] - teacher_logits[:, a]
            mask = mask & (t_margin > 0)
        if not torch.any(mask):
            continue
        s_margin = student_logits[mask, c] - student_logits[mask, a]
        total = total + weight * F.softplus(-s_margin / temperature).mean()
        denom = denom + weight
    return total / denom.clamp_min(1e-12)


def detached(values: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in values.items()}
