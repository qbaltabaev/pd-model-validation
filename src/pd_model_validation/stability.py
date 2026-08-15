"""Population, feature, and temporal stability analysis."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from ._validation import as_1d, clean_binary_inputs, require_columns
from .discrimination import gini


def population_stability_index(
    expected: Iterable[object],
    actual: Iterable[object],
    *,
    bins: int = 10,
    strategy: str = "quantile",
    epsilon: float = 1e-6,
) -> float:
    """Calculate PSI for numeric or categorical values.

    Numeric breakpoints are learned exclusively from ``expected``. Missing
    values form their own bucket. Categorical levels are aligned by union.
    """
    if bins < 2:
        raise ValueError("bins must be at least 2")
    expected_array = as_1d(expected, "expected")
    actual_array = as_1d(actual, "actual")
    expected_series = pd.Series(expected_array)
    actual_series = pd.Series(actual_array)
    numeric = pd.api.types.is_numeric_dtype(expected_series) and pd.api.types.is_numeric_dtype(
        actual_series
    )

    if numeric:
        expected_numeric = pd.to_numeric(expected_series, errors="coerce")
        actual_numeric = pd.to_numeric(actual_series, errors="coerce")
        clean_expected = expected_numeric.dropna().to_numpy(dtype=float)
        if clean_expected.size == 0:
            raise ValueError("expected must contain at least one non-null value")
        if strategy == "quantile":
            edges = np.unique(np.quantile(clean_expected, np.linspace(0, 1, bins + 1)))
        elif strategy == "uniform":
            edges = np.linspace(clean_expected.min(), clean_expected.max(), bins + 1)
        else:
            raise ValueError("strategy must be 'quantile' or 'uniform'")
        if len(edges) == 1:
            edges = np.array([-np.inf, np.inf])
        else:
            edges[0], edges[-1] = -np.inf, np.inf
        expected_bucket = pd.cut(expected_numeric, edges.tolist(), include_lowest=True).astype(str)
        actual_bucket = pd.cut(actual_numeric, edges.tolist(), include_lowest=True).astype(str)
    else:
        expected_bucket = expected_series.astype("string").fillna("<MISSING>")
        actual_bucket = actual_series.astype("string").fillna("<MISSING>")

    expected_share = expected_bucket.value_counts(normalize=True, dropna=False)
    actual_share = actual_bucket.value_counts(normalize=True, dropna=False)
    levels = expected_share.index.union(actual_share.index)
    expected_aligned = expected_share.reindex(levels, fill_value=0).clip(lower=epsilon)
    actual_aligned = actual_share.reindex(levels, fill_value=0).clip(lower=epsilon)
    return float(
        ((actual_aligned - expected_aligned) * np.log(actual_aligned / expected_aligned)).sum()
    )


def feature_stability(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Sequence[str],
    *,
    bins: int = 10,
) -> pd.DataFrame:
    """Calculate PSI for a collection of model features."""
    require_columns(reference, features)
    require_columns(current, features)
    return pd.DataFrame(
        {
            "feature": list(features),
            "psi": [
                population_stability_index(reference[feature], current[feature], bins=bins)
                for feature in features
            ],
        }
    ).sort_values("psi", ascending=False, ignore_index=True)


def performance_by_period(
    frame: pd.DataFrame,
    *,
    target: str,
    probability: str,
    date: str,
    frequency: str = "M",
    min_observations: int = 30,
) -> pd.DataFrame:
    """Compute volume, default rate, average PD, and Gini by time period."""
    require_columns(frame, [target, probability, date])
    working = frame[[target, probability, date]].copy()
    working[date] = pd.to_datetime(working[date], errors="coerce")
    working = working.dropna()
    working["period"] = working[date].dt.to_period(frequency).astype(str)
    rows: list[dict[str, object]] = []
    for period, group in working.groupby("period", sort=True):
        y_true, y_prob = clean_binary_inputs(
            group[target], group[probability], allow_single_class=True
        )
        metric = np.nan
        if len(group) >= min_observations and len(np.unique(y_true)) == 2:
            metric = gini(y_true, y_prob)
        rows.append(
            {
                "period": period,
                "n": len(group),
                "defaults": int(y_true.sum()),
                "observed_rate": float(y_true.mean()),
                "mean_pd": float(y_prob.mean()),
                "gini": metric,
            }
        )
    return pd.DataFrame(rows)


def gini_degradation(
    reference_target: Iterable[object],
    reference_probability: Iterable[object],
    current_target: Iterable[object],
    current_probability: Iterable[object],
) -> float:
    """Relative drop in Gini from reference to current sample."""
    reference_gini = gini(reference_target, reference_probability)
    current_gini = gini(current_target, current_probability)
    if np.isclose(reference_gini, 0.0):
        return float(reference_gini - current_gini)
    return float((reference_gini - current_gini) / abs(reference_gini))
