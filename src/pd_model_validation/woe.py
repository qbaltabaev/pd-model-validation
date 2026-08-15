"""Weight-of-evidence transformation and information value."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ._validation import require_columns

MISSING_BIN = "<MISSING>"


def woe_table(
    feature: pd.Series,
    target: pd.Series,
    *,
    smoothing: float = 0.5,
) -> pd.DataFrame:
    """Calculate WoE and IV for an already-binned feature.

    WoE follows the scorecard convention ``log(% non-events / % events)``.
    Additive smoothing keeps empty event/non-event cells finite.
    """
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")
    data = pd.DataFrame(
        {"bin": feature.astype("string").fillna(MISSING_BIN), "target": target}
    ).dropna(subset=["target"])
    if not set(data["target"].unique()).issubset({0, 1}):
        raise ValueError("target must contain only 0 and 1")
    if data["target"].nunique() != 2:
        raise ValueError("target must contain both classes")
    grouped = data.groupby("bin", observed=True)["target"].agg(["count", "sum"])
    grouped = grouped.rename(columns={"sum": "events"})
    grouped["non_events"] = grouped["count"] - grouped["events"]
    n_groups = len(grouped)
    event_total = grouped["events"].sum() + smoothing * n_groups
    non_event_total = grouped["non_events"].sum() + smoothing * n_groups
    grouped["event_share"] = (grouped["events"] + smoothing) / event_total
    grouped["non_event_share"] = (grouped["non_events"] + smoothing) / non_event_total
    grouped["woe"] = np.log(grouped["non_event_share"] / grouped["event_share"])
    grouped["iv_component"] = (grouped["non_event_share"] - grouped["event_share"]) * grouped["woe"]
    return grouped.reset_index()


def information_value(
    feature: pd.Series,
    target: pd.Series,
    *,
    bins: int = 10,
    smoothing: float = 0.5,
) -> float:
    """Calculate IV with quantile binning for numeric features."""
    binned = _fit_and_apply_bins(feature, feature, bins)[1]
    return float(woe_table(binned, target, smoothing=smoothing)["iv_component"].sum())


def _fit_and_apply_bins(
    fit_values: pd.Series, transform_values: pd.Series, bins: int
) -> tuple[np.ndarray | None, pd.Series]:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if pd.api.types.is_numeric_dtype(fit_values):
        clean = pd.to_numeric(fit_values, errors="coerce").dropna().to_numpy(dtype=float)
        if clean.size == 0:
            edges = np.array([-np.inf, np.inf])
        else:
            edges = np.unique(np.quantile(clean, np.linspace(0, 1, bins + 1)))
            if len(edges) == 1:
                edges = np.array([-np.inf, np.inf])
            else:
                edges[0], edges[-1] = -np.inf, np.inf
        binned = pd.cut(
            pd.to_numeric(transform_values, errors="coerce"), edges.tolist(), include_lowest=True
        )
        return edges, binned.astype("string").fillna(MISSING_BIN)
    return None, transform_values.astype("string").fillna(MISSING_BIN)


class WoETransformer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Scikit-learn-compatible WoE transformer.

    Numeric features use reference-sample quantile bins; categorical features
    retain their observed levels. Unknown levels map to neutral WoE (0).
    """

    def __init__(
        self,
        *,
        bins: int = 10,
        smoothing: float = 0.5,
        suffix: str = "_woe",
        handle_unknown: Literal["zero", "error"] = "zero",
    ):
        self.bins = bins
        self.smoothing = smoothing
        self.suffix = suffix
        self.handle_unknown = handle_unknown

    def fit(self, X: pd.DataFrame, y: pd.Series) -> WoETransformer:
        """Learn reference-sample bins, WoE mappings, and feature IVs."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if self.bins < 2:
            raise ValueError("bins must be at least 2")
        if self.smoothing <= 0:
            raise ValueError("smoothing must be positive")
        if self.handle_unknown not in {"zero", "error"}:
            raise ValueError("handle_unknown must be 'zero' or 'error'")
        target = pd.Series(np.asarray(y), index=X.index)
        if len(target) != len(X):
            raise ValueError("X and y must have the same length")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.bin_edges_: dict[str, np.ndarray | None] = {}
        self.mappings_: dict[str, dict[str, float]] = {}
        self.iv_: dict[str, float] = {}
        for feature in X.columns:
            edges, binned = _fit_and_apply_bins(X[feature], X[feature], self.bins)
            table = woe_table(binned, target, smoothing=self.smoothing)
            self.bin_edges_[feature] = edges
            self.mappings_[feature] = dict(zip(table["bin"], table["woe"], strict=True))
            self.iv_[feature] = float(table["iv_component"].sum())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted WoE mappings without relearning them on new data."""
        if not hasattr(self, "mappings_"):
            raise RuntimeError("transformer is not fitted")
        require_columns(X, (str(feature) for feature in self.feature_names_in_))
        transformed: dict[str, pd.Series] = {}
        for raw_feature in self.feature_names_in_:
            feature = str(raw_feature)
            edges = self.bin_edges_[feature]
            if edges is None:
                binned = X[feature].astype("string").fillna(MISSING_BIN)
            else:
                binned = (
                    pd.cut(
                        pd.to_numeric(X[feature], errors="coerce"),
                        edges.tolist(),
                        include_lowest=True,
                    )
                    .astype("string")
                    .fillna(MISSING_BIN)
                )
            mapped = binned.map(self.mappings_[feature])
            if self.handle_unknown == "error" and mapped.isna().any():
                unknown = sorted(binned.loc[mapped.isna()].unique().tolist())
                raise ValueError(f"unknown bins for {feature}: {', '.join(unknown)}")
            transformed[f"{feature}{self.suffix}"] = mapped.fillna(0.0).astype(float)
        return pd.DataFrame(transformed, index=X.index)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return transformed feature names using the configured suffix."""
        if not hasattr(self, "feature_names_in_"):
            raise RuntimeError("transformer is not fitted")
        features = self.feature_names_in_ if input_features is None else np.asarray(input_features)
        if input_features is not None and not np.array_equal(features, self.feature_names_in_):
            raise ValueError("input_features must match the fitted feature names")
        return np.asarray([f"{feature}{self.suffix}" for feature in features], dtype=object)
