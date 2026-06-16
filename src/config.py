"""
src/config.py — load the frozen config and resolve ALL paths centrally.

Usage (top of every notebook, after the Colab bootstrap):
    from src.config import CFG, PATHS, set_all_seeds
    set_all_seeds(CFG["anchor_seed"])
    df_path = PATHS.data("ciciot2023", "clean.parquet")

The golden rule of this repo: a path is NEVER constructed inline with
Path(REPO)/'something' in a notebook or downstream module. It comes from
PATHS. This is the direct fix for the X-IDS multi-seed failure, where
inline path construction made outputs un-redirectable and overwrote the
seed-42 baseline. If you find yourself writing a raw path, add it here.
"""

from __future__ import annotations
import os
import random
from pathlib import Path

import yaml

# --- locate repo root (this file is <repo>/src/config.py) ------------
REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_FILE = REPO_ROOT / "config" / "config.yaml"


def load_config(path: Path = _CONFIG_FILE) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


CFG = load_config()


class _Paths:
    """Resolve every path from config. Methods create parents on demand."""

    def __init__(self, cfg: dict, repo_root: Path):
        self._cfg = cfg
        self.repo = repo_root
        # In Colab the repo lives inside Drive; data/models stay in Drive.
        self.drive_root = Path(cfg["paths"]["drive_root"])

    def _under_repo(self, key: str, *parts) -> Path:
        p = self.repo / self._cfg["paths"][key]
        for part in parts:
            p = p / part
        return p

    # tracked-in-git outputs ------------------------------------------
    def tables(self, *parts) -> Path:
        p = self._under_repo("tables", *parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def figures(self, *parts) -> Path:
        p = self._under_repo("figures", *parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def arrays(self, *parts) -> Path:
        p = self._under_repo("arrays", *parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def logs(self, *parts) -> Path:
        p = self._under_repo("logs", *parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # gitignored data/binaries (Drive) --------------------------------
    def data(self, *parts) -> Path:
        p = self.repo / self._cfg["paths"]["data"]
        for part in parts:
            p = p / part
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def model(self, dataset: str, arch: str, compression: str, seed: int, ext: str = "pt") -> Path:
        """Seed- and compression-suffixed model path. The suffixing here is
        exactly what the old pipeline did inline and got wrong — keep it
        in ONE place so a seed/compression override can never miss a file."""
        name = f"{dataset}__{arch}__{compression}__seed{seed}.{ext}"
        p = self.repo / self._cfg["paths"]["models"] / dataset / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


PATHS = _Paths(CFG, REPO_ROOT)


def set_all_seeds(seed: int) -> None:
    """Seed python, numpy, and torch (CPU+CUDA). Call at the top of every run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # deterministic where feasible; document any op that forces non-determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def require_frozen() -> None:
    """Guard for measure/crux/predict notebooks: refuse to run if the
    freeze-checklist items are still null. Mirrors the prereg discipline —
    no trust metric before the prereg is frozen."""
    missing = []
    if CFG["split"]["grouping_variable"] is None:
        missing.append("split.grouping_variable")
    if CFG["label_granularity"]["chosen"] is None:
        missing.append("label_granularity.chosen")
    if CFG["crux"]["auc_retention_threshold"] is None:
        missing.append("crux.auc_retention_threshold")
    if missing:
        raise RuntimeError(
            "Prereg not frozen — set these in config/config.yaml first: "
            + ", ".join(missing)
        )
