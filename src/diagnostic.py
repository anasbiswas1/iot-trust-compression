"""Compatibility interface for the low-capacity collapse diagnostic.

The canonical manuscript implementation is :mod:`src.predict`. This module
keeps the original public API usable, removes the former ``NotImplementedError``
stubs, and deliberately limits itself to transparent low-capacity rules.

Nothing in this module changes or regenerates the archived manuscript results.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _canonical_predict_module():
    """Import the canonical implementation lazily to keep this module light."""
    from . import predict

    return predict


def build_features(baseline_bundle: Any, **assemble_kwargs: Any) -> pd.DataFrame:
    """Return or construct the per-class M0 feature table.

    Supported inputs
    ----------------
    ``pandas.DataFrame``
        Returned as a defensive copy.
    mapping containing ``feature_df`` or ``features``
        The referenced DataFrame is returned as a copy.
    mapping containing the canonical arguments
        ``model_M0``, ``df``, ``splits``, ``scaler``, ``feat_cols``, and ``le``
        are forwarded to :func:`src.predict.assemble_features`.

    The explicit input contract prevents the old stub from silently guessing
    how an experiment bundle is structured.
    """
    if isinstance(baseline_bundle, pd.DataFrame):
        return baseline_bundle.copy()

    if not isinstance(baseline_bundle, Mapping):
        raise TypeError(
            "baseline_bundle must be a DataFrame or a mapping with either a "
            "feature table or the canonical assemble_features arguments"
        )

    for key in ("feature_df", "features"):
        value = baseline_bundle.get(key)
        if isinstance(value, pd.DataFrame):
            return value.copy()

    required = ("model_M0", "df", "splits", "scaler", "feat_cols", "le")
    missing = [name for name in required if name not in baseline_bundle]
    if missing:
        raise KeyError(
            "cannot construct features; missing canonical bundle keys: "
            + ", ".join(missing)
        )

    predict = _canonical_predict_module()
    kwargs = {name: baseline_bundle[name] for name in required}
    kwargs.update(assemble_kwargs)
    return predict.assemble_features(**kwargs)


def _numeric_frame(
    features: pd.DataFrame | np.ndarray,
    feature_cols: Sequence[str] | None,
) -> pd.DataFrame:
    """Select a small numeric design matrix with stable column names."""
    if isinstance(features, pd.DataFrame):
        if feature_cols is None:
            frame = features.select_dtypes(include=[np.number]).copy()
        else:
            missing = [name for name in feature_cols if name not in features.columns]
            if missing:
                raise KeyError("missing feature columns: " + ", ".join(missing))
            frame = features.loc[:, list(feature_cols)].apply(pd.to_numeric, errors="coerce")
    else:
        array = np.asarray(features)
        if array.ndim != 2:
            raise ValueError("features must be a 2-D array or DataFrame")
        names = [f"x{i}" for i in range(array.shape[1])]
        frame = pd.DataFrame(array, columns=names)

    if frame.shape[1] == 0:
        raise ValueError("no numeric predictor columns are available")
    return frame


def fit_simple_rule(
    features: pd.DataFrame | np.ndarray,
    target: Sequence[float] | np.ndarray | pd.Series,
    *,
    feature_cols: Sequence[str] | None = None,
    alpha: float = 1.0,
) -> Pipeline:
    """Fit a transparent low-capacity ridge rule.

    Missing predictor values are median-imputed, predictors are standardised,
    and a ridge regression is fitted. The returned scikit-learn pipeline can be
    used directly with ``predict``. Feature selection must be explicit when
    more than a small number of candidate columns are present; this helper does
    not perform hidden model search.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    frame = _numeric_frame(features, feature_cols)
    y = np.asarray(target, dtype=float).reshape(-1)
    if frame.shape[0] != y.shape[0]:
        raise ValueError("features and target must have the same number of rows")

    valid_target = np.isfinite(y)
    if valid_target.sum() < 3:
        raise ValueError("at least three finite target values are required")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("rule", Ridge(alpha=alpha)),
        ]
    )
    model.fit(frame.loc[valid_target], y[valid_target])
    return model


def _count_column(features: pd.DataFrame) -> str:
    for name in ("sample_count", "support", "train_count"):
        if name in features.columns:
            return name
    raise KeyError("frequency baseline requires sample_count, support, or train_count")


def baseline_frequency_only(features: pd.DataFrame) -> np.ndarray:
    """Frequency-only score: rarer classes receive larger risk scores."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    counts = pd.to_numeric(features[_count_column(features)], errors="coerce").to_numpy(float)
    if np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("class counts must be finite and non-negative")
    return 1.0 / np.maximum(counts, 1.0)


def _standardised(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("baseline inputs must be finite")
    scale = values.std(ddof=0)
    if scale < 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / scale


def baseline_tran_fioretto(features: pd.DataFrame) -> np.ndarray:
    """Transparent margin/gradient-norm comparison score.

    This is a repository compatibility baseline, not a claim of reproducing the
    full Tran–Fioretto experimental procedure. Lower margin is treated as more
    vulnerable; when a gradient-norm column is available, higher gradient norm
    contributes additional vulnerability. If no gradient norm was archived,
    the function returns the margin-only score and makes that limitation
    explicit through the available inputs rather than inventing values.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if "margin" not in features.columns:
        raise KeyError("margin/gradient baseline requires a margin column")

    margin = pd.to_numeric(features["margin"], errors="coerce").to_numpy(float)
    score = -_standardised(margin)

    grad_name = next(
        (name for name in ("gradient_norm", "grad_norm", "per_class_grad_norm") if name in features.columns),
        None,
    )
    if grad_name is not None:
        grad = pd.to_numeric(features[grad_name], errors="coerce").to_numpy(float)
        score = score + _standardised(grad)
    return score


def _precision_at_k(pred: np.ndarray, actual: np.ndarray, k: int) -> float:
    k = max(1, min(int(k), pred.size))
    predicted_top = set(np.argsort(pred)[-k:])
    actual_top = set(np.argsort(actual)[-k:])
    return len(predicted_top.intersection(actual_top)) / k


def evaluate(
    pred: Sequence[float] | np.ndarray,
    actual_delta_recall: Sequence[float] | np.ndarray,
    *,
    k: int | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate a continuous class-risk score against observed recall change.

    Larger values are interpreted as greater deterioration. The function
    reports rank association, R², MAE, precision@k, and a row-bootstrap
    confidence interval for Spearman correlation. For tiny class-level samples,
    all values should be interpreted as descriptive rather than as a guarantee
    of out-of-family generalisation.
    """
    p = np.asarray(pred, dtype=float).reshape(-1)
    a = np.asarray(actual_delta_recall, dtype=float).reshape(-1)
    if p.shape[0] != a.shape[0]:
        raise ValueError("pred and actual_delta_recall must have equal length")

    valid = np.isfinite(p) & np.isfinite(a)
    p, a = p[valid], a[valid]
    if p.size < 3:
        raise ValueError("at least three finite paired values are required")

    spearman_r, spearman_p = spearmanr(p, a)
    kendall_tau, kendall_p = kendalltau(p, a)
    chosen_k = k if k is not None else max(1, int(np.ceil(0.20 * p.size)))

    rng = np.random.default_rng(seed)
    boot: list[float] = []
    if n_boot < 0:
        raise ValueError("n_boot must be non-negative")
    for _ in range(n_boot):
        idx = rng.integers(0, p.size, p.size)
        if np.unique(p[idx]).size < 2 or np.unique(a[idx]).size < 2:
            continue
        value = spearmanr(p[idx], a[idx]).statistic
        if np.isfinite(value):
            boot.append(float(value))

    if boot:
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    else:
        ci_low = ci_high = np.nan

    return {
        "n": int(p.size),
        "spearman": float(spearman_r),
        "spearman_p": float(spearman_p),
        "spearman_ci_95": (float(ci_low), float(ci_high)),
        "kendall": float(kendall_tau),
        "kendall_p": float(kendall_p),
        "r2": float(r2_score(a, p)),
        "mae": float(mean_absolute_error(a, p)),
        "precision_at_k": float(_precision_at_k(p, a, chosen_k)),
        "k": int(max(1, min(int(chosen_k), p.size))),
    }


# Canonical helpers are exposed lazily so historical imports keep working.
def assemble_features(*args: Any, **kwargs: Any):
    return _canonical_predict_module().assemble_features(*args, **kwargs)


def build_prediction_table(*args: Any, **kwargs: Any):
    return _canonical_predict_module().build_prediction_table(*args, **kwargs)


def screen_features(*args: Any, **kwargs: Any):
    return _canonical_predict_module().screen_features(*args, **kwargs)


def evaluate_predictor(*args: Any, **kwargs: Any):
    return _canonical_predict_module().evaluate_predictor(*args, **kwargs)
