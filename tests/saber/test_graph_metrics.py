
import numpy as np

from src.saber.taxonomy import ciciot2023_taxonomy, DEFAULT_COST_PROFILES
from src.saber.risk_graph import (
    build_alert_semantic_vulnerability_graph,
    aggregate_robust_edge_weights,
    merge_cost_profile_graphs,
)
from src.saber.metrics import (
    action_weighted_boundary_inversion_rate,
    semantic_decomposition,
)


def _fixture():
    names = ["BenignTraffic", "DoS-UDP_Flood", "DDoS-UDP_Flood"]
    tax = ciciot2023_taxonomy(names)
    y = np.array([0, 0, 1, 1, 1, 2, 2, 2])
    z = np.array([
        [3, 0, 0], [2, 1, 0],
        [0, 3, 2], [0, 2, 2.1], [0, 1.9, 2.0],
        [0, 1, 3], [0, 2.1, 2.2], [0, 0, 4],
    ], dtype=float)
    return tax, y, z


def test_graph_builds_directed_edges():
    tax, y, z = _fixture()
    frames = []
    for profile in DEFAULT_COST_PROFILES.values():
        frames.append(build_alert_semantic_vulnerability_graph(
            z, y, tax, profile, top_k=1, min_class_support=2
        ))
    merged = merge_cost_profile_graphs(frames)
    robust = aggregate_robust_edge_weights(merged)
    assert len(robust) == tax.n_classes
    assert np.isclose(robust["robust_weight"].sum(), 1.0)


def test_awbir_detects_boundary_reversal():
    tax, y, teacher = _fixture()
    graph = build_alert_semantic_vulnerability_graph(
        teacher, y, tax, DEFAULT_COST_PROFILES["balanced_soc"],
        top_k=1, min_class_support=2,
    )
    graph = graph.rename(columns={"normalized_weight": "robust_weight"})
    student = teacher.copy()
    # Reverse DoS versus DDoS on one source sample.
    idx = np.where(y == 1)[0][0]
    student[idx, 1], student[idx, 2] = student[idx, 2] - 1, student[idx, 1] + 1
    value, audit = action_weighted_boundary_inversion_rate(
        teacher, student, y, graph
    )
    assert value >= 0
    assert len(audit) > 0


def test_semantic_decomposition():
    tax, y, z = _fixture()
    pred = z.argmax(1)
    out = semantic_decomposition(y, pred, tax)
    assert 0 <= out["fine_accuracy"] <= 1
    assert 0 <= out["binary_attack_recall"] <= 1
