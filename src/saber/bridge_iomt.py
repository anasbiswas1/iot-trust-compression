"""Bridge from the CIC-IoMT-2024 frozen split (IoMT compression project) into the SABER pipeline.

Contract mirrors bridge_ciciot.load_bridge: returns (train_loader, val_loader, class_names, taxonomy,
manifest). The official CIC test parquet is never opened here; the clip bounds and scaler parameters
needed to process it identically are stored in the manifest for a later locked audit.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.saber.taxonomy import LabelTaxonomy

IOMT_SRC = "/content/drive/MyDrive/IOMT_Compression_Research/iomt-compression-research/src"
IOMT_REPORTS = "/content/drive/MyDrive/IOMT_Compression_Research/iomt-compression-research/reports"
FAMILY_LOWER = {"Benign": "benign", "DDoS": "ddos", "DoS": "dos", "MQTT": "mqtt",
                "Recon": "recon", "Spoofing": "spoofing"}
SEED = 42
VAL_FRACTION = 0.20


def _sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def iomt_taxonomy(class_names, family_by_class) -> LabelTaxonomy:
    fams = tuple(FAMILY_LOWER[f] for f in family_by_class)
    benign = [c for c, f in zip(class_names, fams) if f == "benign"]
    assert len(benign) == 1, f"expected exactly one benign class, got {benign}"
    return LabelTaxonomy(class_names=tuple(class_names), family_by_class=fams,
                         benign_class=benign[0], dataset="CIC-IoMT-2024")


def _leakage_drop_list() -> tuple[list[str], str]:
    """Apply the IoMT project's Category-A leakage audit only if its format is unambiguous."""
    path = os.path.join(IOMT_REPORTS, "cat_a_feature_leakage.csv")
    if not os.path.exists(path):
        return [], "leakage report absent; no feature dropped"
    rep = pd.read_csv(path)
    feat_col = next((c for c in rep.columns if c.lower() in ("feature", "column", "name")), None)
    flag_col = next((c for c in rep.columns if c.lower() in ("drop", "flagged", "flag", "leak", "leaky", "exclude")), None)
    if feat_col is None or flag_col is None:
        return [], f"leakage report present but not applied (columns {list(rep.columns)})"
    flags = rep[flag_col]
    if flags.dtype == bool or set(flags.dropna().unique()) <= {0, 1, True, False, "True", "False"}:
        drop = rep.loc[flags.astype(str).isin(["True", "1"]), feat_col].astype(str).tolist()
        return drop, f"dropped {len(drop)} Category-A features per {os.path.basename(path)}[{flag_col}]"
    return [], f"leakage report present but flag column {flag_col!r} not boolean; not applied"


def build_bridge(cache_dir: str | Path, batch_size: int = 1024, force: bool = False):
    """Load, deduplicate, split, preprocess and cache. Returns (train_loader, val_loader, class_names, taxonomy, manifest)."""
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    tensors_path = cache_dir / "tensors.pt"
    if manifest_path.exists() and tensors_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        blob = torch.load(tensors_path, map_location="cpu", weights_only=False)
    else:
        if IOMT_SRC not in sys.path:
            sys.path.insert(0, IOMT_SRC)
        import gc
        import iomtc_data as D  # the IoMT project's loader, used as-is

        try:
            train_df, _test_unused = D.load_official(add_targets=True)
        except Exception as exc:                                     # the loader resolves paths from its own config
            hint = getattr(D, "_official_dir", lambda: "?")()
            raise RuntimeError(f"IoMT loader failed (resolved data dir: {hint}); run from a runtime where "
                               f"{IOMT_SRC} imports and its data directory exists") from exc
        del _test_unused                                             # never touched in the SABER arms
        gc.collect()
        feat_cols = [c for c in D.feature_columns(train_df)]
        drop, leak_note = _leakage_drop_list()
        feat_cols = [c for c in feat_cols if c not in set(drop)]
        non_numeric = [c for c in feat_cols if not np.issubdtype(train_df[c].dtype, np.number)]
        assert not non_numeric, f"non-numeric feature columns would corrupt preprocessing: {non_numeric}"

        n_raw = int(len(train_df))
        train_df = train_df.drop_duplicates(subset=feat_cols + ["attack_type"]).reset_index(drop=True)
        n_dedup = int(len(train_df))

        # frozen stratified validation carve-out from the official train side
        rng = np.random.default_rng(SEED)
        val_mask = np.zeros(n_dedup, dtype=bool)
        for _, idx in train_df.groupby("attack_type").indices.items():
            idx = np.asarray(idx); rng.shuffle(idx)
            val_mask[idx[: int(round(VAL_FRACTION * len(idx)))]] = True

        class_names = sorted(train_df["attack_type"].astype(str).unique().tolist())
        fam_of = train_df.drop_duplicates("attack_type").set_index("attack_type")["family"].astype(str).to_dict()
        family_by_class = [fam_of[c] for c in class_names]
        cls_index = {c: i for i, c in enumerate(class_names)}
        y_all = train_df["attack_type"].astype(str).map(cls_index).to_numpy(np.int64)

        tr_df, va_df = train_df[~val_mask], train_df[val_mask]
        del train_df; gc.collect()
        los, his = D.fit_clip(tr_df, feat_cols)
        X_tr = D.apply_clip(tr_df, feat_cols, los, his)
        del tr_df; gc.collect()
        X_va = D.apply_clip(va_df, feat_cols, los, his)
        del va_df; gc.collect()
        mean, std = X_tr.mean(axis=0), X_tr.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        X_tr = ((X_tr - mean) / std).astype(np.float32)
        X_va = ((X_va - mean) / std).astype(np.float32)
        y_tr, y_va = y_all[~val_mask], y_all[val_mask]
        gc.collect()

        blob = {"X_tr": torch.from_numpy(X_tr), "y_tr": torch.from_numpy(y_tr),
                "X_va": torch.from_numpy(X_va), "y_va": torch.from_numpy(y_va)}
        torch.save(blob, tensors_path)
        manifest = {
            "dataset": "CIC-IoMT-2024 WiFi/MQTT, official CIC train side; official test never opened",
            "n_official_train_rows": n_raw, "n_after_exact_dedup": n_dedup,
            "dedup_rule": "exact duplicates on all retained features plus attack_type",
            "val_fraction": VAL_FRACTION, "split_seed": SEED,
            "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
            "feature_columns": feat_cols, "n_features": len(feat_cols), "leakage_note": leak_note,
            "clip_percentiles": [0.1, 99.9], "clip_low": los.tolist(), "clip_high": his.tolist(),
            "scaler_mean": mean.tolist(), "scaler_std": std.tolist(),
            "class_names": class_names, "family_by_class": family_by_class,
            "train_class_counts": np.bincount(y_tr, minlength=len(class_names)).tolist(),
            "val_class_counts": np.bincount(y_va, minlength=len(class_names)).tolist(),
            "sha256_X_tr": _sha256_array(X_tr), "sha256_y_tr": _sha256_array(y_tr),
            "sha256_X_va": _sha256_array(X_va), "sha256_y_va": _sha256_array(y_va),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

    class_names = manifest["class_names"]
    taxonomy = iomt_taxonomy(class_names, manifest["family_by_class"])
    gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(blob["X_tr"], blob["y_tr"]), batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(TensorDataset(blob["X_va"], blob["y_va"]), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, class_names, taxonomy, manifest
