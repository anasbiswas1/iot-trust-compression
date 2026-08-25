"""Frozen CICIoT2023 bridge: rebuilds the exact diagnostic-pipeline objects."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import CFG, set_all_seeds
from src import data as D, models as M


def load_bridge(batch_size: int = 1024):
    seed = int(CFG["anchor_seed"])
    set_all_seeds(seed)

    df = D.clean(D.load_raw("ciciot2023", subsample=True, seed=seed), "ciciot2023")
    splits = D.temporal_within_capture_split(df, seed=seed)

    str_cols = [c for c in df.columns if not np.issubdtype(df[c].dtype, np.number)]
    label_col = next(c for c in str_cols
                     if "BenignTraffic" in set(df[c].astype(str).unique()))
    feat_cols = [c for c in df.columns if c not in str_cols]

    le = LabelEncoder().fit(df[label_col].to_numpy())
    scaler = StandardScaler().fit(
        df.loc[splits["train"], feat_cols].to_numpy(np.float32))
    class_names = list(le.classes_)
    assert len(class_names) == 34, f"expected 34 classes, got {len(class_names)}"

    def _loader(idx, shuffle):
        X = torch.tensor(scaler.transform(df.loc[idx, feat_cols].to_numpy(np.float32)))
        y = torch.tensor(le.transform(df.loc[idx, label_col].to_numpy()),
                         dtype=torch.long)
        return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle,
                          generator=torch.Generator().manual_seed(seed))

    ckpt = Path("models/ciciot2023") / f"ciciot2023__cnn1d__M0__seed{seed}.pt"
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    ch = (int(sd["conv.0.weight"].shape[0]), int(sd["conv.3.weight"].shape[0]))
    model = M.build("cnn1d", in_dim=len(feat_cols),
                    n_classes=len(class_names), channels=ch)
    model.load_state_dict(sd)
    model.eval()
    return (_loader(splits["train"], True), _loader(splits["val"], False),
            _loader(splits["test"], False), model, class_names)
