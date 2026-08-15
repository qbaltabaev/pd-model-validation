"""Internal validation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def as_1d(values: Iterable[object], name: str, *, dtype: Any = None) -> np.ndarray:
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
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")


def quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    ranks = pd.Series(values).rank(method="first", pct=True)
    result = np.minimum((ranks.to_numpy() * n_bins).astype(int), n_bins - 1)
    return np.asarray(result, dtype=int)
