
"""Repository integration helpers for the existing iot-trust-compression project.

The bridge prefers explicit paths. Auto-discovery is intentionally conservative:
if more than one plausible artifact exists it raises and asks the notebook user
to choose, avoiding silent use of the wrong seed or split.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import importlib
import inspect
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn


KNOWN_COLAB_ROOT = Path(
    "/content/drive/MyDrive/IoT_Trust_Research/iot-trust-compression"
)


@dataclass(frozen=True)
class SaberRepo:
    root: Path
    output_root: Path

    @classmethod
    def discover(cls, explicit: str | Path | None = None) -> "SaberRepo":
        candidates: list[Path] = []
        if explicit is not None:
            candidates.append(Path(explicit))
        env = os.environ.get("SABER_REPO")
        if env:
            candidates.append(Path(env))
        candidates.extend([KNOWN_COLAB_ROOT, Path.cwd()])
        for candidate in candidates:
            candidate = candidate.expanduser().resolve()
            if (candidate / "src").is_dir() and (candidate / "config").is_dir():
                out = candidate / "results" / "saber"
                out.mkdir(parents=True, exist_ok=True)
                if str(candidate) not in sys.path:
                    sys.path.insert(0, str(candidate))
                return cls(candidate, out)
        raise FileNotFoundError(
            "Could not locate the repository. Set SABER_REPO or pass an explicit "
            "path to SaberRepo.discover(...)."
        )

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def output(self, *parts: str) -> Path:
        path = self.output_root.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def config_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def find_files(
    root: str | Path,
    patterns: Sequence[str],
    *,
    exclude_parts: Sequence[str] = ("backup", ".git", "__pycache__"),
) -> list[Path]:
    root = Path(root)
    found: list[Path] = []
    for pattern in patterns:
        for path in root.rglob(pattern):
            lowered = str(path).lower()
            if any(part.lower() in lowered for part in exclude_parts):
                continue
            if path.is_file():
                found.append(path)
    return sorted(set(found))


def choose_single(
    paths: Sequence[Path],
    *,
    description: str,
    preferred_tokens: Sequence[str] = (),
) -> Path:
    if not paths:
        raise FileNotFoundError(f"No candidate found for {description}.")
    scored = []
    for path in paths:
        text = str(path).lower()
        score = sum(token.lower() in text for token in preferred_tokens)
        scored.append((score, path))
    max_score = max(score for score, _ in scored)
    best = [path for score, path in scored if score == max_score]
    if len(best) != 1:
        listing = "\n".join(f"  - {p}" for p in best[:20])
        raise RuntimeError(
            f"Ambiguous {description}; set its path explicitly. Candidates:\n{listing}"
        )
    return best[0]


def discover_anchor_checkpoint(repo: SaberRepo) -> Path:
    candidates = find_files(
        repo.root,
        ["*.pt", "*.pth", "*.ckpt"],
        exclude_parts=("backup", ".git", "saber", "prune50", "prune80", "distill"),
    )
    return choose_single(
        candidates,
        description="uncompressed anchor checkpoint",
        preferred_tokens=("cnn", "anchor", "m0", "seed42", "baseline"),
    )


def load_array(path: str | Path, *, key: str | None = None) -> Any:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        payload = np.load(path, allow_pickle=False)
        if key is not None:
            return payload[key]
        if len(payload.files) == 1:
            return payload[payload.files[0]]
        return {name: payload[name] for name in payload.files}
    if suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu")
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported artifact format: {path}")


def discover_cached_logits(
    repo: SaberRepo,
    split: str,
    *,
    model_token: str = "m0",
) -> tuple[Path, Path]:
    split = split.lower()
    logits = find_files(
        repo.root,
        [
            f"*{split}*logit*.npy",
            f"*{split}*logit*.npz",
            f"*logit*{split}*.npy",
            f"*logit*{split}*.npz",
        ],
    )
    labels = find_files(
        repo.root,
        [
            f"*{split}*label*.npy",
            f"*{split}*target*.npy",
            f"*label*{split}*.npy",
            f"*target*{split}*.npy",
        ],
    )
    logit_path = choose_single(
        logits,
        description=f"{split} teacher logits",
        preferred_tokens=(model_token, "cnn", "anchor"),
    )
    label_path = choose_single(
        labels,
        description=f"{split} labels",
        preferred_tokens=(model_token, "cnn", "anchor"),
    )
    return logit_path, label_path


def unpack_batch(batch):
    if isinstance(batch, Mapping):
        x = batch.get("x", batch.get("features", batch.get("inputs")))
        y = batch.get("y", batch.get("labels", batch.get("target")))
        env = batch.get("environment", batch.get("env", batch.get("provenance")))
        if x is None or y is None:
            raise KeyError("Batch mapping must expose features and labels.")
        return x, y, env
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1], batch[2] if len(batch) > 2 else None
    raise TypeError(f"Unsupported batch: {type(batch)!r}")


def collect_logits(
    model: nn.Module,
    loader,
    *,
    device: str | torch.device,
    max_batches: int | None = None,
    input_transform=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    device = torch.device(device)
    model = model.to(device)
    training = model.training
    model.eval()
    logits_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    env_out: list[np.ndarray] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x, y, env = unpack_batch(batch)
            x = x.to(device)
            if input_transform is not None:
                x = input_transform(x)
            z = model(x).detach().cpu().numpy()
            logits_out.append(z)
            labels_out.append(
                y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)
            )
            if env is not None:
                env_out.append(
                    env.detach().cpu().numpy()
                    if hasattr(env, "detach")
                    else np.asarray(env)
                )
    model.train(training)
    return (
        np.concatenate(logits_out),
        np.concatenate(labels_out).astype(np.int64),
        np.concatenate(env_out) if env_out else None,
    )


def flexible_load_state_dict(model: nn.Module, checkpoint: str | Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "network"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                state = value
                break
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint does not contain a state dictionary.")
    cleaned = {}
    for key, value in state.items():
        new_key = str(key)
        for prefix in ("module.", "model.", "network."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch.\n"
            f"Missing keys: {missing}\nUnexpected keys: {unexpected}"
        )
    return dict(payload) if isinstance(payload, Mapping) else {}


def infer_class_names_from_results(repo: SaberRepo) -> list[str]:
    """Infer the richest available fine-label order from archived result tables.

    The historical repository contains several 13-class/measurable-only tables.
    Returning the first match would silently build a truncated taxonomy, so this
    function ranks candidates by the number of unique labels and prefers tables
    whose names indicate an all-class matrix.
    """
    candidates = find_files(
        repo.root / "results",
        [
            "*per_class*recall*.csv",
            "*recall_matrix*.csv",
            "*class*tier*.csv",
            "*all_class*.csv",
            "*per_class*.csv",
        ],
    )
    ranked: list[tuple[int, int, Path, list[str]]] = []
    for path in candidates:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        for column in ("class", "Class", "class_name", "label"):
            if column not in df.columns:
                continue
            names = [str(v) for v in df[column].dropna().drop_duplicates()]
            if len(names) < 10:
                continue
            token_bonus = sum(
                token in path.name.lower()
                for token in ("all", "34", "matrix", "tier", "complete")
            )
            ranked.append((len(names), token_bonus, path, names))
    if not ranked:
        raise FileNotFoundError(
            "Could not infer class order from result CSVs. Supply CLASS_NAMES "
            "explicitly from the model's label encoder."
        )
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_n, _, best_path, best_names = ranked[0]
    if best_n < 30:
        raise RuntimeError(
            f"The richest inferred label table has only {best_n} classes "
            f"({best_path}). Supply the exact full label-encoder order manually."
        )
    return best_names


def import_existing_module(module_name: str):
    """Import a module from the repository with a clear error message."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise ImportError(
            f"Could not import {module_name!r}. Confirm the repository root and "
            "install the original requirements."
        ) from exc


def save_run_manifest(
    path: str | Path,
    *,
    notebook: str,
    config: Mapping[str, Any],
    artifacts: Sequence[str | Path],
    git_commit: str | None = None,
) -> None:
    output = {
        "notebook": notebook,
        "config": dict(config),
        "config_hash": config_hash(config),
        "git_commit": git_commit,
        "artifacts": [
            {
                "path": str(p),
                "sha256": file_sha256(p) if Path(p).is_file() else None,
            }
            for p in artifacts
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")


def _call_with_available_context(func, context: Mapping[str, Any]):
    """Call a repository function using only parameters available in context."""
    signature = inspect.signature(func)
    kwargs = {}
    for name, param in signature.parameters.items():
        if name in context:
            kwargs[name] = context[name]
        elif param.default is inspect._empty and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(f"Required parameter {name!r} is unavailable")
    return func(**kwargs)


def _parse_loader_bundle(value):
    """Extract train/val/test loaders from common return conventions."""
    if isinstance(value, Mapping):
        aliases = {
            "train": ("train", "train_loader", "loader_train"),
            "val": ("val", "valid", "validation", "val_loader", "valid_loader"),
            "test": ("test", "test_loader", "loader_test"),
        }
        found = {}
        for split, keys in aliases.items():
            for key in keys:
                if key in value:
                    found[split] = value[key]
                    break
        if len(found) == 3:
            return found["train"], found["val"], found["test"], value
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        return value[0], value[1], value[2], value
    raise TypeError("Could not interpret the data-loader bundle")


def auto_discover_loaders(repo: SaberRepo):
    """Best-effort loader discovery for the historical repository.

    This is deliberately best-effort. The notebook prints a manual bridge cell
    when the historical API does not match one of the common conventions.
    """
    config_mod = import_existing_module("src.config")
    data_mod = import_existing_module("src.data")
    context = {
        "CFG": getattr(config_mod, "CFG", None),
        "cfg": getattr(config_mod, "CFG", None),
        "config": getattr(config_mod, "CFG", None),
        "PATHS": getattr(config_mod, "PATHS", None),
        "paths": getattr(config_mod, "PATHS", None),
    }
    candidate_names = (
        "get_dataloaders",
        "make_dataloaders",
        "build_dataloaders",
        "create_dataloaders",
        "load_dataloaders",
        "prepare_dataloaders",
        "load_data",
        "prepare_data",
    )
    errors = {}
    for name in candidate_names:
        func = getattr(data_mod, name, None)
        if not callable(func):
            continue
        try:
            result = _call_with_available_context(func, context)
            return _parse_loader_bundle(result)
        except Exception as exc:
            errors[name] = repr(exc)
    raise RuntimeError(
        "No compatible loader factory was discovered. Define TRAIN_LOADER, "
        "VAL_LOADER, and TEST_LOADER in the notebook bridge cell. Attempts: "
        + json.dumps(errors, indent=2)
    )


class CompatibleCNN1D(nn.Module):
    """Fallback architecture matching the manuscript's two-block CNN1D."""

    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(128, n_classes)

    def features(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        return self.pool(x).squeeze(-1)

    def forward(self, x):
        return self.head(self.features(x))


def auto_build_cnn(
    n_features: int,
    n_classes: int,
    *,
    checkpoint: str | Path | None = None,
):
    """Try repository model factories/classes, then the documented fallback."""
    models_mod = import_existing_module("src.models")
    context = {
        "arch": "cnn1d",
        "architecture": "cnn1d",
        "model_name": "cnn1d",
        "n_features": n_features,
        "num_features": n_features,
        "input_dim": n_features,
        "in_features": n_features,
        "n_classes": n_classes,
        "num_classes": n_classes,
        "output_dim": n_classes,
    }
    candidates = []
    for name in ("build_model", "make_model", "get_model", "create_model"):
        obj = getattr(models_mod, name, None)
        if callable(obj):
            candidates.append((name, obj))
    for name, obj in vars(models_mod).items():
        if inspect.isclass(obj) and issubclass(obj, nn.Module) and "cnn" in name.lower():
            candidates.append((name, obj))
    errors = {}
    model = None
    for name, candidate in candidates:
        try:
            model = _call_with_available_context(candidate, context)
            if isinstance(model, nn.Module):
                break
        except Exception as exc:
            errors[name] = repr(exc)
            model = None
    if model is None:
        model = CompatibleCNN1D(n_features=n_features, n_classes=n_classes)
    if checkpoint is not None:
        flexible_load_state_dict(model, checkpoint)
    return model, errors
