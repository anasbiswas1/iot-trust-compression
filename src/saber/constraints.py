
"""Adaptive operational constraints for SABER-IDS."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
import torch.nn.functional as F

from .losses import HierarchyTensors, aggregate_family_probabilities


@dataclass
class ConstraintTargets:
    """Permitted validation degradation relative to a teacher/reference."""

    attack_miss_max: float
    benign_false_alert_max: float
    family_nll_max: float
    class_cvar_loss_max: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "attack_miss_max": float(self.attack_miss_max),
            "benign_false_alert_max": float(self.benign_false_alert_max),
            "family_nll_max": float(self.family_nll_max),
            "class_cvar_loss_max": (
                None
                if self.class_cvar_loss_max is None
                else float(self.class_cvar_loss_max)
            ),
        }


@dataclass
class DualRiskController:
    """Projected dual ascent for differentiable risk proxies."""

    targets: ConstraintTargets
    step_size: float = 0.05
    max_dual: float = 100.0
    lambdas: dict[str, float] = field(
        default_factory=lambda: {
            "attack_miss": 0.0,
            "benign_false_alert": 0.0,
            "family_nll": 0.0,
            "class_cvar": 0.0,
        }
    )

    def _threshold(self, name: str) -> float | None:
        mapping = {
            "attack_miss": self.targets.attack_miss_max,
            "benign_false_alert": self.targets.benign_false_alert_max,
            "family_nll": self.targets.family_nll_max,
            "class_cvar": self.targets.class_cvar_loss_max,
        }
        return mapping[name]

    def update(self, detached_metrics: Mapping[str, float]) -> dict[str, float]:
        """Update Lagrange multipliers using detached validation metrics."""
        for name, value in detached_metrics.items():
            if name not in self.lambdas:
                continue
            threshold = self._threshold(name)
            if threshold is None:
                continue
            new_value = self.lambdas[name] + self.step_size * (
                float(value) - float(threshold)
            )
            self.lambdas[name] = float(min(self.max_dual, max(0.0, new_value)))
        return dict(self.lambdas)

    def state_dict(self) -> dict[str, object]:
        return {
            "targets": self.targets.as_dict(),
            "step_size": self.step_size,
            "max_dual": self.max_dual,
            "lambdas": dict(self.lambdas),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.step_size = float(state.get("step_size", self.step_size))
        self.max_dual = float(state.get("max_dual", self.max_dual))
        values = state.get("lambdas", {})
        if isinstance(values, Mapping):
            for key in self.lambdas:
                if key in values:
                    self.lambdas[key] = float(values[key])


def differentiable_risk_proxies(
    logits: torch.Tensor,
    targets: torch.Tensor,
    hierarchy: HierarchyTensors,
    *,
    cvar_q: float = 0.80,
) -> dict[str, torch.Tensor]:
    """Differentiable proxies used in the training-time Lagrangian."""
    hierarchy.to(logits.device)
    targets = targets.long()
    probs = logits.softmax(dim=1)
    benign_prob = probs[:, hierarchy.benign_index]
    attack_prob = 1.0 - benign_prob
    attack_mask = targets != hierarchy.benign_index
    benign_mask = ~attack_mask

    attack_miss = (
        benign_prob[attack_mask].mean()
        if torch.any(attack_mask)
        else logits.new_zeros(())
    )
    benign_false_alert = (
        attack_prob[benign_mask].mean()
        if torch.any(benign_mask)
        else logits.new_zeros(())
    )

    family_probs = aggregate_family_probabilities(probs, hierarchy)
    family_targets = hierarchy.class_to_family[targets]
    family_nll = F.nll_loss(
        torch.log(family_probs.clamp_min(1e-12)),
        family_targets,
    )

    per_sample_ce = F.cross_entropy(logits, targets, reduction="none")
    # Approximate class-level tail risk by first averaging within each present
    # fine class, then taking the upper quantile of class means.
    class_means = []
    for c in torch.unique(targets):
        mask = targets == c
        class_means.append(per_sample_ce[mask].mean())
    if class_means:
        class_vector = torch.stack(class_means)
        k = max(1, int(torch.ceil(
            torch.tensor((1.0 - cvar_q) * len(class_means), device=logits.device)
        ).item()))
        class_cvar = torch.topk(class_vector, k=k, largest=True).values.mean()
    else:
        class_cvar = logits.new_zeros(())

    return {
        "attack_miss": attack_miss,
        "benign_false_alert": benign_false_alert,
        "family_nll": family_nll,
        "class_cvar": class_cvar,
    }


def lagrangian_penalty(
    metrics: Mapping[str, torch.Tensor],
    controller: DualRiskController,
) -> torch.Tensor:
    """Return sum lambda_j * relu(g_j - epsilon_j)."""
    first = next(iter(metrics.values()))
    penalty = first.new_zeros(())
    for name, metric in metrics.items():
        threshold = controller._threshold(name)
        if threshold is None:
            continue
        lam = float(controller.lambdas.get(name, 0.0))
        if lam <= 0:
            continue
        penalty = penalty + lam * torch.relu(metric - float(threshold))
    return penalty


def detached_metric_values(
    metrics: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in metrics.items()}
