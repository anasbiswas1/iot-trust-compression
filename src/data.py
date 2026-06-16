
"""
src/data.py — loading, leakage-critical cleaning, and the splits.

Coded against the provenance-preserving multi-file CICIoT2023 release
(nikitamanaenkov/large-scale-attacks-in-iot-environment): one folder per attack
type under data/raw/CICIoT2023_multifile/, each holding one or more
<...>.pcap.csv part-files with 46 CICFlowMeter features + a 'label' column.
Each row is stamped with its capture-folder provenance as 'group'.

SPLIT DECISION (frozen): each attack type was captured separately, so capture
file confounds with the class label. Therefore:
  * PRIMARY  = temporal_within_capture: split each capture's rows by capture
    order (row order proxies capture time). Breaks near-duplicate-burst leakage
    while keeping EVERY rare class evaluable on both sides.
  * ROBUSTNESS = strict_capture_grouped: whole captures to one side; only runs
    on classes with >= 2 capture part-files (essentially Benign) — confirms the
    effect is not an artifact of the temporal-order assumption.
  * REFERENCE = random stratified split, kept ONLY to measure the leakage gap.

Dedup + identity-field removal are MECHANISM-CRITICAL (leaked identifiers create
spurious separability that compression 'loses', masquerading as capacity loss).
"""
from __future__ import annotations
from pathlib import Path
import glob, os, re
import numpy as np
import pandas as pd

from .config import CFG, PATHS

# 46 CICFlowMeter features (everything except the label). Confirmed schema.
LABEL_COL = "label"
GROUP_COL = "group"

# CICIoT2023 multi-file rows carry no in-row identity fields (flow IDs/IPs/timestamps
# were not exported), so identity removal is a no-op there but kept for symmetry.
IDENTITY_FIELDS = {
    "ciciot2023": [],
    "ton_iot": ["src_ip", "dst_ip", "src_port", "dst_port",
                "dns_query", "ssl_subject", "ssl_issuer",
                "http_uri", "http_user_agent",
                "weird_name", "weird_addl", "weird_notice"],
}

# folder spellings vary in case (DDoS-PSHACK_FLOOD vs ..._Flood) -> canonicalise
_CANON = {
    "ddos-pshack_flood": "DDoS-PSHACK_Flood",
    "ddos-rstfinflood": "DDoS-RSTFINFlood",
    "ddos-syn_flood": "DDoS-SYN_Flood",
    "benign_final": "BenignTraffic",
}
def _canon_label(folder: str) -> str:
    return _CANON.get(folder.lower(), folder)


def load_raw(dataset: str, subsample: bool = True, seed: int | None = None) -> pd.DataFrame:
    """Load a dataset into one DataFrame with 'label' and (for ciciot2023) 'group'.

    subsample=True (default, sensible on Colab Pro): cap each MAJORITY class while
    keeping EVERY rare class whole, so the long-tail phenomenon is intact but the
    working set is fast to iterate on. Caps are read from CFG. Run with
    subsample=False once at the end for the full-scale confirmation pass.
    """
    seed = CFG["anchor_seed"] if seed is None else seed
    if dataset == "ciciot2023":
        return _load_ciciot_multifile(subsample=subsample, seed=seed)
    if dataset == "ton_iot":
        return _load_ton(subsample=subsample, seed=seed)
    raise ValueError(dataset)


def _find_capture_root(start: Path) -> Path:
    """Locate the directory that directly contains the per-attack capture folders.
    Some unzips nest everything under an extra wrapper (e.g. .../CICIoT2023_multifile/data/),
    so descend through single-child wrapper folders until we find folders that
    actually contain .csv files. Robust to either layout."""
    cur = Path(start)
    for _ in range(4):  # bounded descent
        subdirs = [d for d in cur.iterdir() if d.is_dir()]
        # does any subdir directly hold csvs? then `cur` is the capture root.
        if any(any(s.glob("*.csv")) for s in subdirs):
            return cur
        # otherwise, if there's exactly one wrapper subdir, descend into it
        if len(subdirs) == 1:
            cur = subdirs[0]; continue
        break
    return cur


def _load_ciciot_multifile(subsample: bool, seed: int) -> pd.DataFrame:
    root = _find_capture_root(PATHS.data("raw", "CICIoT2023_multifile"))
    folders = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    if not folders:
        raise FileNotFoundError(f"No capture folders under {root} — check the download path.")

    cap = CFG.get("subsample", {}).get("majority_cap_per_class", 200_000)
    rng = np.random.default_rng(seed)
    frames = []
    for folder in folders:
        files = sorted(glob.glob(os.path.join(root, folder, "*.csv")))
        parts = []
        for fi, f in enumerate(files):
            df = pd.read_csv(f)
            # capture-order proxy: preserve part-file index then within-file row order
            df["_part"] = fi
            df["_row"] = np.arange(len(df))
            parts.append(df)
        cls = pd.concat(parts, ignore_index=True)
        cls[LABEL_COL] = _canon_label(folder)
        cls[GROUP_COL] = folder                      # provenance = capture folder
        # subsample MAJORITY classes only; keep rare classes whole
        if subsample and len(cls) > cap:
            cls = cls.iloc[rng.choice(len(cls), cap, replace=False)].reset_index(drop=True)
        frames.append(cls)
    df = pd.concat(frames, ignore_index=True)
    return df


def _load_ton(subsample: bool, seed: int) -> pd.DataFrame:
    f = PATHS.data("raw", "TON_IoT", "train_test_network.csv")
    df = pd.read_csv(f)
    # Use the MULTICLASS 'type' as the canonical label. Drop the binary 'label'
    # FIRST, then rename — otherwise renaming 'type'->'label' collides with the
    # existing binary 'label' and creates two identically named columns.
    if "label" in df.columns:
        df = df.drop(columns=["label"])
    df = df.rename(columns={"type": LABEL_COL})
    df[GROUP_COL] = "ton_single_capture"            # TON network ships as one curated file
    # order helpers for uniformity with the multifile loader (TON = single capture, so
    # one part; the 'temporal' split degenerates to a within-class row-order split here.
    # NOTE: the curated TON file may not preserve true capture order, so for TON a
    # random/stratified split is arguably more appropriate — revisit at cross-dataset stage.)
    df["_part"] = 0
    df["_row"] = np.arange(len(df))
    return df


def clean(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Identity-field removal + dedup. MECHANISM-CRITICAL. Done on FEATURES only;
    label/group/order-helper columns are preserved."""
    keep_meta = [c for c in (LABEL_COL, GROUP_COL, "_part", "_row") if c in df.columns]
    drop = [c for c in IDENTITY_FIELDS.get(dataset, []) if c in df.columns]
    df = df.drop(columns=drop)
    # numeric coercion + drop rows with inf/NaN introduced by CICFlowMeter
    feat = [c for c in df.columns if c not in keep_meta]
    df[feat] = df[feat].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feat).reset_index(drop=True)
    # dedup on FEATURES + label (near-duplicate flow rows). Keep order helpers out of the key.
    dedup_key = [c for c in df.columns if c not in ("_part", "_row")]
    df = df.drop_duplicates(subset=dedup_key).reset_index(drop=True)
    return df


def group_count_audit(df: pd.DataFrame, label_col: str = LABEL_COL) -> pd.DataFrame:
    """Per-class instance count and capture-group count, with the survival verdict.
    For THIS dataset 'groups' = capture part-files within the class (≈ how many
    chunks the class was captured in). Drives the 34-vs-survivors reality check and
    flags which classes can take the STRICT capture-grouped robustness split."""
    surv = CFG["split"]["fine_type_survival"]
    g = df.groupby(label_col).agg(
        n_instances=("_row", "size"),
        n_capture_parts=("_part", lambda s: s.nunique()),
    )
    g["survives_min_instances"] = g["n_instances"] >= surv["min_train_instances"]
    g["can_strict_group"] = g["n_capture_parts"] >= surv["min_groups"]   # >=2 parts
    return g.sort_values("n_instances")


def temporal_within_capture_split(df, seed: int, label_col: str = LABEL_COL):
    """PRIMARY split. Within each (class), order by capture (_part,_row) and cut
    chronologically into train/val/test by config fractions. Near-duplicate bursts
    stay on one side; every class appears on all sides. Returns index arrays."""
    tr, va, te = (CFG["split"][k] for k in ("train_frac", "val_frac", "test_frac"))
    train_idx, val_idx, test_idx = [], [], []
    for _, sub in df.groupby(label_col):
        order = sub.sort_values(["_part", "_row"]).index.to_numpy()
        n = len(order); a, b = int(n * tr), int(n * (tr + va))
        train_idx += order[:a].tolist(); val_idx += order[a:b].tolist(); test_idx += order[b:].tolist()
    return {"train": np.array(train_idx), "val": np.array(val_idx), "test": np.array(test_idx)}


def strict_capture_grouped_split(df, seed: int, label_col: str = LABEL_COL):
    """ROBUSTNESS split. Whole capture-parts to one side; only meaningful for classes
    with >= 2 parts (others land entirely in train and are reported as excluded from
    this check). Returns index arrays + the list of classes actually evaluated."""
    surv = CFG["split"]["fine_type_survival"]
    rng = np.random.default_rng(seed)
    train_idx, test_idx, evaluated = [], [], []
    for cls, sub in df.groupby(label_col):
        parts = sorted(sub["_part"].unique())
        if len(parts) >= surv["min_groups"]:
            rng.shuffle(parts)
            k = max(1, int(len(parts) * CFG["split"]["test_frac"]))
            test_parts = set(parts[:k]); evaluated.append(cls)
            test_idx += sub[sub["_part"].isin(test_parts)].index.tolist()
            train_idx += sub[~sub["_part"].isin(test_parts)].index.tolist()
        else:
            train_idx += sub.index.tolist()   # single-capture class: train-only here
    return {"train": np.array(train_idx), "test": np.array(test_idx), "evaluated_classes": evaluated}


def random_reference_split(df, seed: int, label_col: str = LABEL_COL):
    """REFERENCE ONLY — stratified random split to MEASURE the leakage gap vs the
    temporal split. Never the headline regime."""
    from sklearn.model_selection import train_test_split
    tr, va, te = (CFG["split"][k] for k in ("train_frac", "val_frac", "test_frac"))
    idx = np.arange(len(df)); y = df[label_col].to_numpy()
    tr_i, tmp_i = train_test_split(idx, train_size=tr, stratify=y, random_state=seed)
    rel = te / (va + te)
    va_i, te_i = train_test_split(tmp_i, test_size=rel, stratify=y[tmp_i], random_state=seed)
    return {"train": tr_i, "val": va_i, "test": te_i}
