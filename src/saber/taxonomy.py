
"""Label hierarchies and asymmetric operational cost profiles for SABER-IDS.

This module is intentionally independent of the historical manuscript code. It can
be imported from Colab notebooks and from unit tests without loading a dataset or
a neural network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import json
import re

import numpy as np


def _key(value: object) -> str:
    """Canonical key used to match label spelling variants."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


@dataclass(frozen=True)
class CostProfile:
    """Ordinal operational costs for directed prediction transitions."""

    name: str
    attack_to_benign: float
    benign_to_attack: float
    cross_family_attack: float
    same_family_attack: float
    exact: float = 0.0

    def validate(self) -> None:
        vals = (
            self.attack_to_benign,
            self.benign_to_attack,
            self.cross_family_attack,
            self.same_family_attack,
            self.exact,
        )
        if any(v < 0 for v in vals):
            raise ValueError(f"Costs must be non-negative, got {vals}")
        if self.exact != 0:
            raise ValueError("The exact-prediction cost must be zero.")

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "attack_to_benign": float(self.attack_to_benign),
            "benign_to_attack": float(self.benign_to_attack),
            "cross_family_attack": float(self.cross_family_attack),
            "same_family_attack": float(self.same_family_attack),
            "exact": float(self.exact),
        }


DEFAULT_COST_PROFILES: dict[str, CostProfile] = {
    # Misses dominate, while false alarms still matter.
    "miss_sensitive": CostProfile(
        name="miss_sensitive",
        attack_to_benign=12.0,
        benign_to_attack=3.0,
        cross_family_attack=6.0,
        same_family_attack=2.0,
    ),
    # Balanced security-operations-centre profile.
    "balanced_soc": CostProfile(
        name="balanced_soc",
        attack_to_benign=10.0,
        benign_to_attack=6.0,
        cross_family_attack=6.0,
        same_family_attack=2.0,
    ),
    # High-volume gateway where alert fatigue is itself safety-relevant.
    "alert_fatigue": CostProfile(
        name="alert_fatigue",
        attack_to_benign=8.0,
        benign_to_attack=10.0,
        cross_family_attack=5.0,
        same_family_attack=2.0,
    ),
}


@dataclass(frozen=True)
class LabelTaxonomy:
    """Binary-family-fine hierarchy for a fixed class order."""

    class_names: tuple[str, ...]
    family_by_class: tuple[str, ...]
    benign_class: str
    dataset: str = "unknown"

    def __post_init__(self) -> None:
        if len(self.class_names) != len(self.family_by_class):
            raise ValueError("class_names and family_by_class must have equal length")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be unique")
        if self.benign_class not in self.class_names:
            raise ValueError(f"benign_class={self.benign_class!r} is absent")
        benign_idx = self.class_names.index(self.benign_class)
        if self.family_by_class[benign_idx] != "benign":
            raise ValueError("The benign class must belong to the 'benign' family")

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def class_to_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.class_names)}

    @property
    def families(self) -> tuple[str, ...]:
        ordered: list[str] = []
        for family in self.family_by_class:
            if family not in ordered:
                ordered.append(family)
        return tuple(ordered)

    @property
    def family_to_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.families)}

    @property
    def class_to_family_index(self) -> np.ndarray:
        fmap = self.family_to_index
        return np.asarray([fmap[f] for f in self.family_by_class], dtype=np.int64)

    @property
    def benign_index(self) -> int:
        return self.class_names.index(self.benign_class)

    def family_of(self, class_index: int) -> str:
        return self.family_by_class[int(class_index)]

    def is_attack(self, class_index: int) -> bool:
        return int(class_index) != self.benign_index

    def transition_type(self, true_index: int, pred_index: int) -> str:
        t = int(true_index)
        p = int(pred_index)
        if t == p:
            return "exact"
        t_attack = self.is_attack(t)
        p_attack = self.is_attack(p)
        if t_attack and not p_attack:
            return "attack_to_benign"
        if not t_attack and p_attack:
            return "benign_to_attack"
        if t_attack and p_attack:
            if self.family_of(t) == self.family_of(p):
                return "same_family_attack"
            return "cross_family_attack"
        # There is only one benign fine label in the current design.
        return "other"

    def transition_cost(
        self,
        true_index: int,
        pred_index: int,
        profile: CostProfile,
    ) -> float:
        profile.validate()
        kind = self.transition_type(true_index, pred_index)
        if kind == "exact":
            return profile.exact
        if kind == "attack_to_benign":
            return profile.attack_to_benign
        if kind == "benign_to_attack":
            return profile.benign_to_attack
        if kind == "cross_family_attack":
            return profile.cross_family_attack
        if kind == "same_family_attack":
            return profile.same_family_attack
        raise RuntimeError(f"Unhandled transition type: {kind}")

    def cost_matrix(self, profile: CostProfile) -> np.ndarray:
        matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.float64)
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                matrix[i, j] = self.transition_cost(i, j, profile)
        return matrix

    def family_targets(self, fine_targets: Sequence[int] | np.ndarray) -> np.ndarray:
        y = np.asarray(fine_targets, dtype=np.int64)
        return self.class_to_family_index[y]

    def binary_targets(self, fine_targets: Sequence[int] | np.ndarray) -> np.ndarray:
        y = np.asarray(fine_targets, dtype=np.int64)
        return (y != self.benign_index).astype(np.int64)

    def aggregate_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.ndim != 2 or probs.shape[1] != self.n_classes:
            raise ValueError(
                f"Expected probabilities [N,{self.n_classes}], got {probs.shape}"
            )
        out = np.zeros((len(probs), len(self.families)), dtype=np.float64)
        mapping = self.class_to_family_index
        for c in range(self.n_classes):
            out[:, mapping[c]] += probs[:, c]
        return out

    def to_records(self) -> list[dict[str, object]]:
        return [
            {
                "dataset": self.dataset,
                "class_index": i,
                "class_name": name,
                "binary_label": "benign" if i == self.benign_index else "attack",
                "family": self.family_by_class[i],
            }
            for i, name in enumerate(self.class_names)
        ]

    def to_json(self) -> str:
        return json.dumps(
            {
                "dataset": self.dataset,
                "class_names": list(self.class_names),
                "family_by_class": list(self.family_by_class),
                "benign_class": self.benign_class,
            },
            indent=2,
        )


_CICIOT_FAMILY_BY_CANONICAL: dict[str, str] = {
    "BenignTraffic": "benign",
    "DDoS-ACK_Fragmentation": "ddos",
    "DDoS-HTTP_Flood": "ddos",
    "DDoS-ICMP_Flood": "ddos",
    "DDoS-ICMP_Fragmentation": "ddos",
    "DDoS-PSHACK_Flood": "ddos",
    "DDoS-RSTFINFlood": "ddos",
    "DDoS-SYN_Flood": "ddos",
    "DDoS-SlowLoris": "ddos",
    "DDoS-SynonymousIP_Flood": "ddos",
    "DDoS-TCP_Flood": "ddos",
    "DDoS-UDP_Flood": "ddos",
    "DDoS-UDP_Fragmentation": "ddos",
    "DoS-HTTP_Flood": "dos",
    "DoS-SYN_Flood": "dos",
    "DoS-TCP_Flood": "dos",
    "DoS-UDP_Flood": "dos",
    "Mirai-greeth_flood": "mirai",
    "Mirai-greip_flood": "mirai",
    "Mirai-udpplain": "mirai",
    "Recon-HostDiscovery": "reconnaissance",
    "Recon-OSScan": "reconnaissance",
    "Recon-PingSweep": "reconnaissance",
    "Recon-PortScan": "reconnaissance",
    "VulnerabilityScan": "reconnaissance",
    "DNS_Spoofing": "spoofing_mitm",
    "MITM-ArpSpoofing": "spoofing_mitm",
    "BrowserHijacking": "malware",
    "Backdoor_Malware": "malware",
    "Uploading_Attack": "malware",
    "XSS": "web_application",
    "SqlInjection": "web_application",
    "CommandInjection": "web_application",
    "DictionaryBruteForce": "brute_force",
}


def _canonical_ciciot_name(label: str) -> str | None:
    target = _key(label)
    # Include common abbreviations used in result tables.
    aliases: dict[str, str] = {
        _key("Benign"): "BenignTraffic",
        _key("DDoS-UDP-Frag"): "DDoS-UDP_Fragmentation",
        _key("DDoS-ICMP-Frag"): "DDoS-ICMP_Fragmentation",
        _key("DDoS-ACK-Frag"): "DDoS-ACK_Fragmentation",
        _key("DDoS-SynonIP"): "DDoS-SynonymousIP_Flood",
        _key("R-HostDiscovery"): "Recon-HostDiscovery",
        _key("R-OSScan"): "Recon-OSScan",
        _key("R-PingSweep"): "Recon-PingSweep",
        _key("R-PortScan"): "Recon-PortScan",
        _key("VulnScan"): "VulnerabilityScan",
        _key("DNS-Spoof"): "DNS_Spoofing",
        _key("MITM-Arp"): "MITM-ArpSpoofing",
        _key("DictBrute"): "DictionaryBruteForce",
        _key("SqlInj"): "SqlInjection",
        _key("CmdInj"): "CommandInjection",
        _key("Upload"): "Uploading_Attack",
        _key("BrowserHijack"): "BrowserHijacking",
        _key("Backdoor Malware"): "Backdoor_Malware",
        _key("M-udpplain"): "Mirai-udpplain",
        _key("M-greip flood"): "Mirai-greip_flood",
        _key("M-greeth flood"): "Mirai-greeth_flood",
    }
    for canonical in _CICIOT_FAMILY_BY_CANONICAL:
        aliases.setdefault(_key(canonical), canonical)
    return aliases.get(target)


def ciciot2023_taxonomy(class_names: Iterable[str] | None = None) -> LabelTaxonomy:
    """Create a CICIoT2023 taxonomy in the exact supplied class order.

    Unknown labels fail loudly because silently assigning an attack to the wrong
    family would invalidate alert-semantic costs.
    """
    if class_names is None:
        class_names = tuple(_CICIOT_FAMILY_BY_CANONICAL)
    names = tuple(str(x) for x in class_names)
    families: list[str] = []
    benign_name: str | None = None
    unknown: list[str] = []
    for label in names:
        canonical = _canonical_ciciot_name(label)
        if canonical is None:
            unknown.append(label)
            families.append("UNKNOWN")
            continue
        family = _CICIOT_FAMILY_BY_CANONICAL[canonical]
        families.append(family)
        if canonical == "BenignTraffic":
            benign_name = label
    if unknown:
        raise ValueError(
            "Unmapped CICIoT2023 labels: "
            + ", ".join(unknown)
            + ". Add explicit aliases before running SABER."
        )
    if benign_name is None:
        raise ValueError("No benign CICIoT2023 label was found.")
    return LabelTaxonomy(
        class_names=names,
        family_by_class=tuple(families),
        benign_class=benign_name,
        dataset="CICIoT2023",
    )


def write_cost_profiles(path: str) -> None:
    payload = {name: profile.as_dict() for name, profile in DEFAULT_COST_PROFILES.items()}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
