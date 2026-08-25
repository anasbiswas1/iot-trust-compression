
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.saber.surgery import enumerate_cnn1d_channel_groups, prune_cnn1d_channels
from src.saber.taxonomy import ciciot2023_taxonomy
from src.saber.losses import (
    HierarchyTensors, SaberLossConfig, saber_loss, edges_to_tensors
)


class TinyCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 10, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(10)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(10, n_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        return self.head(self.pool(x).squeeze(-1))


def test_physical_channel_surgery():
    model = TinyCNN()
    x = torch.randn(3, 1, 16)
    groups = enumerate_cnn1d_channel_groups(model, x, minimum_remaining_per_layer=4)
    pruned, audit = prune_cnn1d_channels(
        model, {"conv1": [0, 1], "conv2": [0, 1, 2]}, x,
        minimum_remaining_per_layer=4,
    )
    out = pruned(x)
    assert out.shape == (3, 4)
    assert pruned.conv1.out_channels == 6
    assert pruned.conv2.in_channels == 6
    assert pruned.conv2.out_channels == 7
    assert pruned.head.in_features == 7


def test_hierarchical_loss_backpropagates():
    names = ["BenignTraffic", "DoS-UDP_Flood", "DoS-SYN_Flood", "DDoS-UDP_Flood"]
    tax = ciciot2023_taxonomy(names)
    hierarchy = HierarchyTensors(
        tax.class_to_family_index, tax.benign_index, len(tax.families)
    )
    student = torch.randn(8, 4, requires_grad=True)
    teacher = torch.randn(8, 4)
    y = torch.tensor([0,1,2,3,1,2,3,0])
    edges = pd.DataFrame({
        "source_index": [1,2],
        "target_index": [3,3],
        "robust_weight": [0.6,0.4],
    })
    es, et, ew = edges_to_tensors(edges)
    loss, parts = saber_loss(
        student, teacher, y, hierarchy, es, et, ew, SaberLossConfig()
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert student.grad is not None
