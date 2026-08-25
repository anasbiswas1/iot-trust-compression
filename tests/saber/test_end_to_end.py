
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.saber.taxonomy import ciciot2023_taxonomy, DEFAULT_COST_PROFILES
from src.saber.risk_graph import (
    build_alert_semantic_vulnerability_graph,
    merge_cost_profile_graphs,
    aggregate_robust_edge_weights,
)
from src.saber.surgery import (
    enumerate_cnn1d_channel_groups,
    prune_cnn1d_channels,
    temporarily_zero_group,
    profile_forward_flops,
)
from src.saber.leverage import (
    magnitude_scores,
    gradient_saliency_scores,
    semantic_boundary_leverage,
    robust_profile_aggregation,
    merge_score_tables,
)
from src.saber.selectors import select_groups_greedy, selection_to_prune_map
from src.saber.metrics import action_weighted_boundary_inversion_rate


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


def test_starter_pipeline_end_to_end():
    torch.manual_seed(4)
    names = [
        "BenignTraffic",
        "DoS-UDP_Flood",
        "DoS-SYN_Flood",
        "DDoS-UDP_Flood",
    ]
    tax = ciciot2023_taxonomy(names)
    x = torch.randn(96, 1, 16)
    y = torch.tensor(np.tile(np.arange(4), 24), dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=24, shuffle=False)
    model = TinyCNN(4).eval()
    with torch.no_grad():
        logits = model(x).numpy()

    profile_graphs = [
        build_alert_semantic_vulnerability_graph(
            logits, y.numpy(), tax, profile, top_k=2, min_class_support=5
        )
        for profile in DEFAULT_COST_PROFILES.values()
    ]
    long = merge_cost_profile_graphs(profile_graphs)
    graph = aggregate_robust_edge_weights(long)
    assert np.isclose(graph.robust_weight.sum(), 1.0)

    groups = enumerate_cnn1d_channel_groups(
        model, x[:4], minimum_remaining_per_layer=4
    )
    mag = magnitude_scores(model, groups)
    grad = gradient_saliency_scores(
        model, loader, groups, device="cpu", max_batches=2
    )
    profile_scores = {}
    for name, profile in DEFAULT_COST_PROFILES.items():
        edges = long[long.cost_profile == name]
        score, _ = semantic_boundary_leverage(
            model, loader, groups, edges, device="cpu",
            max_samples_per_class=8, edge_weight_column="normalized_weight",
        )
        profile_scores[name] = score
    robust_scores = robust_profile_aggregation(profile_scores)
    scores = merge_score_tables(groups, mag, grad, robust_scores)
    assert scores.r_sbl.notna().all()

    selection, summary = select_groups_greedy(
        scores,
        importance_column="r_sbl",
        target_reduction_fraction=0.25,
        minimum_remaining_per_layer=4,
    )
    prune_map = selection_to_prune_map(selection)
    student, audit = prune_cnn1d_channels(
        model, prune_map, x[:4], minimum_remaining_per_layer=4
    )
    with torch.no_grad():
        student_logits = student(x).numpy()
    awbir, edge_audit = action_weighted_boundary_inversion_rate(
        logits, student_logits, y.numpy(), graph
    )
    assert np.isfinite(awbir)
    assert len(edge_audit)
    assert profile_forward_flops(student, x[:4])["flops_per_item"] < profile_forward_flops(model, x[:4])["flops_per_item"]
