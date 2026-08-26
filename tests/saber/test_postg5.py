import numpy as np
import pandas as pd
import torch

from src.saber.postg5 import calibrate_prefix_count, lexicographic_checkpoint_index
from src.saber.constraints_v2 import boundary_aligned_risk_proxies
from src.saber.losses import HierarchyTensors


def test_prefix_calibration_uses_realised_cost():
    order = list(range(10))
    reductions = {k: k / 10 for k in range(11)}
    out = calibrate_prefix_count(
        order,
        target_reduction=0.43,
        build_model=lambda k: k,
        realised_reduction=lambda k: reductions[k],
        tolerance=0.04,
    )
    assert out.prefix_length == 4
    assert np.isclose(out.realised_reduction, 0.4)
    assert out.within_tolerance


def test_lexicographic_selection_prefers_feasible_semantic_epoch():
    history = pd.DataFrame(
        {
            "val_awbir": [0.11, 0.08, 0.09],
            "val_hsr_balanced_soc": [0.14, 0.13, 0.12],
            "val_family_macro_f1": [0.60, 0.59, 0.61],
            "val_macro_f1": [0.58, 0.60, 0.59],
            "val_benign_to_attack": [0.02, 0.10, 0.03],
        }
    )
    idx = lexicographic_checkpoint_index(
        history,
        constraints={"val_benign_to_attack": ("max", 0.05)},
    )
    assert idx == 2


def test_boundary_proxy_tracks_logit_boundary():
    hierarchy = HierarchyTensors([0, 1, 1], benign_index=0, n_families=2)
    y = torch.tensor([0, 1])
    safe = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    unsafe = torch.tensor([[0.0, 3.0, 0.0], [3.0, 0.0, 0.0]])
    p_safe = boundary_aligned_risk_proxies(safe, y, hierarchy)
    p_bad = boundary_aligned_risk_proxies(unsafe, y, hierarchy)
    assert p_bad["benign_false_alert"] > p_safe["benign_false_alert"]
    assert p_bad["attack_miss"] > p_safe["attack_miss"]
