"""
src/data.py — loading, leakage-critical cleaning, and the PRIMARY grouped split.

Dedup + identity-field removal are MECHANISM-CRITICAL, not hygiene: leaked
identifiers create spurious separability that compression then "loses",
masquerading as capacity loss and invalidating the EXPLAIN stage. Do them first.

Fill the dataset-specific loaders against your downloaded data; the split,
group-count audit, and survival rules are dataset-agnostic and implemented here.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

from .config import CFG


IDENTITY_FIELDS = {  # extend per dataset; these must NOT enter the feature matrix
    "ton_iot": ["src_ip", "dst_ip", "src_port", "dst_port", "ts", "timestamp"],
    "ciciot2023": [],   # CICFlowMeter flow features; confirm no leaked identifiers
}


def load_raw(dataset: str) -> pd.DataFrame:
    """TODO: dataset-specific load from PATHS.data(dataset, ...).
    Return a DataFrame with a 'label' column (fine-grained for ciciot2023)."""
    raise NotImplementedError("Implement per-dataset load against your Drive data.")


def clean(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Drop identity fields, deduplicate near-duplicate flows. MECHANISM-CRITICAL."""
    cfg = CFG["datasets"][dataset]
    if cfg.get("drop_identity_fields"):
        df = df.drop(columns=[c for c in IDENTITY_FIELDS.get(dataset, []) if c in df.columns])
    if cfg.get("deduplicate"):
        df = df.drop_duplicates()
    return df.reset_index(drop=True)


def group_count_audit(df: pd.DataFrame, group_col: str, label_col: str = "label") -> pd.DataFrame:
    """For a candidate grouping variable, how many groups each label appears in.
    This table RESOLVES two freeze-checklist items: the grouping variable and
    whether 34-way is viable. A label needs >= min_groups non-empty groups AND
    >= min_train_instances to survive the grouped split."""
    surv = CFG["split"]["fine_type_survival"]
    g = df.groupby(label_col)[group_col].agg(["nunique", "count"])
    g.columns = ["n_groups", "n_instances"]
    g["survives"] = (g["n_groups"] >= surv["min_groups"]) & (g["n_instances"] >= surv["min_train_instances"])
    return g.sort_values("n_instances")


def grouped_split(df: pd.DataFrame, group_col: str, seed: int,
                  label_col: str = "label") -> dict:
    """PRIMARY split: no group spans the train/test boundary. Returns index arrays.
    TODO: use sklearn GroupShuffleSplit / StratifiedGroupKFold against config fracs.
    Persist the resulting index arrays (locked splits) under PATHS.arrays."""
    raise NotImplementedError("Implement grouped split with locked, persisted indices.")


def random_reference_split(df: pd.DataFrame, seed: int, label_col: str = "label") -> dict:
    """Random stratified split kept ONLY as the leakage-gap reference (a finding),
    never as the headline regime."""
    raise NotImplementedError("Implement stratified random split for the leakage-gap comparison.")
