"""
src/models.py — the three edge-scale architectures, size- and macroF1-matched.

All return a torch.nn.Module exposing .features(x) (penultimate representation)
and .forward(x) (logits). The .features() hook is load-bearing: the crux probe,
neural-collapse geometry, and faithfulness work all read the penultimate layer,
and keeping it uniform across architectures AND compression levels is exactly
what the single-stack (PyTorch) decision buys.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class _Base(nn.Module):
    def features(self, x: torch.Tensor) -> torch.Tensor:  # penultimate representation
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # logits
        return self.head(self.features(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MLP(_Base):
    """Dense anchor."""
    def __init__(self, in_dim: int, n_classes: int, hidden=(128, 64)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_classes)

    def features(self, x):
        return self.body(x)


class CNN1D(_Base):
    """Local/convolutional. BUILD FIRST — carries the Stage 1 gate + crux."""
    def __init__(self, in_dim: int, n_classes: int, channels=(32, 64), kernel=3):
        super().__init__()
        convs, c = [], 1
        for ch in channels:
            convs += [nn.Conv1d(c, ch, kernel, padding=kernel // 2), nn.ReLU(),
                      nn.BatchNorm1d(ch)]
            c = ch
        self.conv = nn.Sequential(*convs)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self._flat = channels[-1]
        self.head = nn.Linear(self._flat, n_classes)

    def features(self, x):                 # x: (B, in_dim) -> treat as (B, 1, in_dim)
        z = self.conv(x.unsqueeze(1))
        return self.pool(z).squeeze(-1)


class FTTransformerLite(_Base):
    """Compact attention arm for tabular flow features. If it cannot reach the
    MLP's macro-F1 +/- tolerance on BOTH datasets within the tuning budget,
    fall back to GRUFallback (pre-registered)."""
    def __init__(self, in_dim: int, n_classes: int, d_token=32, n_heads=4, n_layers=2):
        super().__init__()
        self.tokenizer = nn.Linear(1, d_token)          # per-feature scalar -> token
        self.cls = nn.Parameter(torch.randn(1, 1, d_token))
        enc = nn.TransformerEncoderLayer(d_token, n_heads, dim_feedforward=2 * d_token,
                                         batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self._d = d_token
        self.head = nn.Linear(d_token, n_classes)

    def features(self, x):                 # x: (B, in_dim)
        b, f = x.shape
        tok = self.tokenizer(x.unsqueeze(-1))           # (B, F, d)
        cls = self.cls.expand(b, -1, -1)
        z = self.encoder(torch.cat([cls, tok], dim=1))  # (B, F+1, d)
        return z[:, 0]                                   # CLS token


class GRUFallback(_Base):
    """Fallback only. NOTE: a GRU over feature-as-timestep vectors is a
    questionable inductive bias for non-sequential flow data — document if used."""
    def __init__(self, in_dim: int, n_classes: int, hidden=48, layers=1):
        super().__init__()
        self.gru = nn.GRU(1, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def features(self, x):
        out, _ = self.gru(x.unsqueeze(-1))              # (B, F, hidden)
        return out[:, -1]


def build(arch: str, in_dim: int, n_classes: int, **kw) -> _Base:
    return {"mlp": MLP, "cnn1d": CNN1D, "ft_transformer": FTTransformerLite,
            "gru": GRUFallback}[arch](in_dim, n_classes, **kw)
