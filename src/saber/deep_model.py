"""Depth-probe CNN used by Notebook 20 and post-G5 confirmation notebooks."""
from __future__ import annotations

import torch
from torch import nn


class DeepCNN1D(nn.Module):
    def __init__(self, n_classes: int = 34):
        super().__init__()

        def block(in_channels: int, out_channels: int):
            return [
                nn.Conv1d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
            ]

        self.conv = nn.Sequential(
            *block(1, 64),
            *block(64, 128),
            nn.MaxPool1d(2),
            *block(128, 128),
            *block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        features = self.pool(self.conv(x.float())).squeeze(-1)
        return self.head(features)
