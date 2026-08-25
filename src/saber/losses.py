
"""Hierarchical Semantic Boundary Distillation losses for SABER-IDS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class SaberLossConfig:
    fine_weight: float = 1.0
    family_weight: float = 0.5
    binary_weight: float = 0.5
    margin_weight: float = 1.0
    kd_weight: float = 0.5
    calibration_weight: float = 0.1
    kd_temperature: float = 3.0
    margin_delta: float = 1.0

    def validate(self) -> None:
        weights = (
            self.fine_weight,
            self.family_weight,
            self.binary_weight,
            self.margin_weight,
            self.kd_weight,
            self.calibration_weight,
        )
        if any(v < 0 for v in weights):
            raise ValueError("Loss weights must be non-negative")
        if self.kd_temperature <= 0:
            raise ValueError("kd_temperature must be positive")
        if self.margin_delta <= 0:
            raise ValueError("margin_delta must be positive")


class HierarchyTensors:
    """Device-aware class-to-family and benign-index tensors."""

    def __init__(self, class_to_family, benign_index: int, n_families: int):
        mapping = torch.as_tensor(class_to_family, dtype=torch.long)
        self.class_to_family = mapping
        self.benign_index = int(benign_index)
        self.n_families = int(n_families)

    def to(self, device):
        self.class_to_family = self.class_to_family.to(device)
        return self


def aggregate_family_probabilities(
    fine_probabilities: torch.Tensor,
    hierarchy: HierarchyTensors,
) -> torch.Tensor:
    if fine_probabilities.ndim != 2:
        raise ValueError("fine_probabilities must be [batch, classes]")
    out = fine_probabilities.new_zeros(
        (fine_probabilities.shape[0], hierarchy.n_families)
    )
    index = hierarchy.class_to_family.unsqueeze(0).expand(
        fine_probabilities.shape[0], -1
    )
    out.scatter_add_(1, index, fine_probabilities)
    return out


def _family_nll(
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    hierarchy: HierarchyTensors,
) -> torch.Tensor:
    probs = student_logits.softmax(dim=1)
    family_probs = aggregate_family_probabilities(probs, hierarchy)
    family_targets = hierarchy.class_to_family[targets]
    return F.nll_loss(torch.log(family_probs.clamp_min(1e-12)), family_targets)


def _binary_nll(
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    hierarchy: HierarchyTensors,
) -> torch.Tensor:
    probs = student_logits.softmax(dim=1)
    benign_prob = probs[:, hierarchy.benign_index]
    attack_prob = 1.0 - benign_prob
    binary_prob = torch.stack([benign_prob, attack_prob], dim=1)
    binary_target = (targets != hierarchy.benign_index).long()
    return F.nll_loss(torch.log(binary_prob.clamp_min(1e-12)), binary_target)


def _kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    t = float(temperature)
    return (
        F.kl_div(
            F.log_softmax(student_logits / t, dim=1),
            F.softmax(teacher_logits / t, dim=1),
            reduction="batchmean",
        )
        * t
        * t
    )


def _brier_loss(
    student_logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    probs = student_logits.softmax(dim=1)
    one_hot = F.one_hot(targets, num_classes=student_logits.shape[1]).to(
        dtype=probs.dtype
    )
    return torch.mean(torch.sum((probs - one_hot) ** 2, dim=1))


def _directed_margin_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    edge_source: torch.Tensor,
    edge_target: torch.Tensor,
    edge_weight: torch.Tensor,
    *,
    delta: float,
) -> torch.Tensor:
    if edge_source.numel() == 0:
        return student_logits.new_zeros(())
    total = student_logits.new_zeros(())
    denom = student_logits.new_zeros(())
    for c, a, weight in zip(edge_source, edge_target, edge_weight):
        mask = targets == c
        if not torch.any(mask):
            continue
        s_margin = student_logits[mask, c] - student_logits[mask, a]
        t_margin = teacher_logits[mask, c] - teacher_logits[mask, a]
        loss = F.huber_loss(
            s_margin,
            t_margin.detach(),
            reduction="mean",
            delta=delta,
        )
        total = total + weight * loss
        denom = denom + weight
    return total / denom.clamp_min(1e-12)


def saber_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    hierarchy: HierarchyTensors,
    edge_source: torch.Tensor,
    edge_target: torch.Tensor,
    edge_weight: torch.Tensor,
    config: SaberLossConfig,
    *,
    extra_penalty: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the SABER hierarchical boundary-distillation objective."""
    config.validate()
    hierarchy.to(student_logits.device)
    targets = targets.long()
    edge_source = edge_source.to(student_logits.device).long()
    edge_target = edge_target.to(student_logits.device).long()
    edge_weight = edge_weight.to(student_logits.device, dtype=student_logits.dtype)

    components = {
        "fine": F.cross_entropy(student_logits, targets),
        "family": _family_nll(student_logits, targets, hierarchy),
        "binary": _binary_nll(student_logits, targets, hierarchy),
        "margin": _directed_margin_loss(
            student_logits,
            teacher_logits,
            targets,
            edge_source,
            edge_target,
            edge_weight,
            delta=config.margin_delta,
        ),
        "kd": _kd_loss(
            student_logits, teacher_logits.detach(), config.kd_temperature
        ),
        "calibration": _brier_loss(student_logits, targets),
    }
    total = (
        config.fine_weight * components["fine"]
        + config.family_weight * components["family"]
        + config.binary_weight * components["binary"]
        + config.margin_weight * components["margin"]
        + config.kd_weight * components["kd"]
        + config.calibration_weight * components["calibration"]
    )
    if extra_penalty is not None:
        components["constraint_penalty"] = extra_penalty
        total = total + extra_penalty
    components["total"] = total
    return total, components


def edges_to_tensors(edges, weight_column: str = "robust_weight"):
    """Convert a graph DataFrame into source/target/weight tensors."""
    if weight_column not in edges.columns:
        if "normalized_weight" in edges.columns:
            weight_column = "normalized_weight"
        else:
            raise ValueError("No edge weight column found")
    return (
        torch.as_tensor(edges["source_index"].to_numpy(), dtype=torch.long),
        torch.as_tensor(edges["target_index"].to_numpy(), dtype=torch.long),
        torch.as_tensor(edges[weight_column].to_numpy(), dtype=torch.float32),
    )
