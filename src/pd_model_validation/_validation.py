"""Internal validation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def as_1d(values: Iterable[object], name: str, *, dtype: Any = None) -> np.ndarray:
    """Convert an iterable to a non-empty one-dimensional NumPy array."""
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def clean_binary_inputs(
    y_true: Iterable[object],
    y_prob: Iterable[object],
    *,
    allow_single_class: bool = False,
    check_probability: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate, align, and remove incomplete pairs from binary model inputs."""
    target = as_1d(y_true, "y_true")
    probability = as_1d(y_prob, "y_prob", dtype=float)
    if len(target) != len(probability):
        raise ValueError("y_true and y_prob must have the same length")

    numeric_target = pd.to_numeric(pd.Series(target), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(numeric_target) & np.isfinite(probability)
    target = numeric_target[mask]
    probability = probability[mask]
    if target.size == 0:
        raise ValueError("no complete target/probability pairs remain")
    unique = set(np.unique(target))
    if not unique.issubset({0.0, 1.0}):
        raise ValueError("y_true must contain only 0 and 1")
    if not allow_single_class and len(unique) != 2:
        raise ValueError("y_true must contain both classes")
    if check_probability and np.any((probability < 0) | (probability > 1)):
        raise ValueError("y_prob must be between 0 and 1")
    return target.astype(int), probability


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a clear error when a data frame lacks required columns."""
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")


def quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign equal-frequency bins without splitting tied values.

    The effective number of bins may be lower than ``n_bins`` when the input
    contains few unique values. Bin identifiers are contiguous and ordered.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    series = pd.Series(values)
    if series.isna().any():
        raise ValueError("values must not contain missing entries")
    if series.nunique() == 1:
        return np.zeros(len(series), dtype=int)
    ranked_unique = series.rank(method="dense").astype(int) - 1
    unique_values = np.sort(series.unique())
    unique_counts = series.value_counts(sort=False).reindex(unique_values).to_numpy(dtype=int)
    cumulative_midpoints = np.cumsum(unique_counts) - unique_counts / 2.0
    raw_bins = np.floor(cumulative_midpoints / len(series) * n_bins).astype(int)
    raw_bins = np.clip(raw_bins, 0, n_bins - 1)
    contiguous = pd.factorize(raw_bins, sort=True)[0]
    return np.asarray(contiguous[ranked_unique.to_numpy()], dtype=int)
