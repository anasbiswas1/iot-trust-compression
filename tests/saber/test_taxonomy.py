
import numpy as np
from src.saber.taxonomy import ciciot2023_taxonomy, DEFAULT_COST_PROFILES


def test_ciciot_taxonomy_and_cost_direction():
    names = [
        "BenignTraffic",
        "DoS-UDP_Flood",
        "DDoS-UDP_Flood",
        "Recon-PortScan",
        "VulnerabilityScan",
    ]
    tax = ciciot2023_taxonomy(names)
    assert tax.benign_index == 0
    assert tax.family_of(1) == "dos"
    assert tax.family_of(2) == "ddos"
    p = DEFAULT_COST_PROFILES["balanced_soc"]
    assert tax.transition_cost(1, 0, p) == p.attack_to_benign
    assert tax.transition_cost(0, 1, p) == p.benign_to_attack
    assert tax.transition_cost(1, 2, p) == p.cross_family_attack
    assert tax.transition_cost(3, 4, p) == p.same_family_attack


def test_family_probability_aggregation():
    names = ["BenignTraffic", "DoS-UDP_Flood", "DoS-SYN_Flood", "DDoS-UDP_Flood"]
    tax = ciciot2023_taxonomy(names)
    probs = np.array([[0.1, 0.2, 0.3, 0.4]])
    fam = tax.aggregate_probabilities(probs)
    assert np.isclose(fam.sum(), 1.0)
    dos = tax.family_to_index["dos"]
    assert np.isclose(fam[0, dos], 0.5)
