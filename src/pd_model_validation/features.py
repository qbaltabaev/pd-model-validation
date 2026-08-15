"""Feature-level diagnostics for PD scorecards and models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from ._validation import clean_binary_inputs, require_columns
from .discrimination import gini
from .woe import WoETransformer, information_value


def feature_information_value(
    frame: pd.DataFrame, features: Sequence[str], *, target: str, bins: int = 10
) -> pd.DataFrame:
    """Rank features by information value."""
    require_columns(frame, [target, *features])
    rows = [
        {
            "feature": feature,
            "information_value": information_value(frame[feature], frame[target], bins=bins),
        }
        for feature in features
    ]
    return pd.DataFrame(rows).sort_values("information_value", ascending=False, ignore_index=True)


def univariate_gini(
    frame: pd.DataFrame, features: Sequence[str], *, target: str, bins: int = 10
) -> pd.DataFrame:
    """Measure each feature's standalone discriminatory power via WoE."""
    require_columns(frame, [target, *features])
    transformer = WoETransformer(bins=bins).fit(frame[list(features)], frame[target])
    transformed = transformer.transform(frame[list(features)])
    rows = []
    for feature in features:
        value = gini(frame[target], transformed[f"{feature}_woe"])
        rows.append({"feature": feature, "gini": abs(value), "direction": int(np.sign(value))})
    return pd.DataFrame(rows).sort_values("gini", ascending=False, ignore_index=True)


def correlation_diagnostics(
    frame: pd.DataFrame, features: Sequence[str], *, method: str = "spearman"
) -> pd.DataFrame:
    """Return each numeric feature's largest absolute pairwise correlation."""
    require_columns(frame, features)
    if len(features) < 2:
        return pd.DataFrame(
            {
                "feature": list(features),
                "max_abs_correlation": [0.0] * len(features),
                "correlated_with": [None] * len(features),
            }
        )
    numeric = frame[list(features)].apply(pd.to_numeric, errors="coerce")
    raw_matrix = numeric.corr(method=method).abs()
    matrix = raw_matrix.mask(np.eye(len(raw_matrix), dtype=bool))
    rows = []
    for feature in features:
        peer = matrix.loc[feature].idxmax()
        rows.append(
            {
                "feature": feature,
                "max_abs_correlation": float(matrix.loc[feature, peer]),
                "correlated_with": peer,
            }
        )
    return pd.DataFrame(rows).sort_values("max_abs_correlation", ascending=False, ignore_index=True)


def variance_inflation_factors(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Calculate VIFs without requiring statsmodels."""
    require_columns(frame, features)
    data = frame[list(features)].apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        raise ValueError("no complete numeric feature rows remain")
    standardized = (data - data.mean()) / data.std(ddof=0).replace(0, np.nan)
    if standardized.isna().any().any():
        constant = standardized.columns[standardized.isna().all()].tolist()
        raise ValueError(f"constant features have undefined VIF: {', '.join(constant)}")
    correlation = standardized.corr().to_numpy()
    inverse = np.linalg.pinv(correlation)
    return pd.DataFrame({"feature": list(features), "vif": np.diag(inverse)})


def logistic_coefficient_test(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    target: str,
    max_iter: int = 100,
) -> pd.DataFrame:
    """Fit logistic regression with IRLS and report Wald coefficient tests."""
    require_columns(frame, [target, *features])
    data = frame[[target, *features]].apply(pd.to_numeric, errors="coerce").dropna()
    y_true, _ = clean_binary_inputs(data[target], np.full(len(data), 0.5))
    design = np.column_stack([np.ones(len(data)), data[list(features)].to_numpy(dtype=float)])
    coefficient = np.zeros(design.shape[1])
    information = np.eye(design.shape[1])
    for _ in range(max_iter):
        fitted = expit(design @ coefficient)
        weights = np.clip(fitted * (1.0 - fitted), 1e-10, None)
        information = design.T @ (weights[:, None] * design)
        score = design.T @ (y_true - fitted)
        step = np.linalg.pinv(information) @ score
        coefficient += step
        if np.max(np.abs(step)) < 1e-9:
            break
    standard_error = np.sqrt(np.clip(np.diag(np.linalg.pinv(information)), 0, None))
    z_score = np.divide(
        coefficient, standard_error, out=np.full_like(coefficient, np.nan), where=standard_error > 0
    )
    p_value = 2.0 * norm.sf(np.abs(z_score))
    names = ["intercept", *features]
    return pd.DataFrame(
        {
            "feature": names,
            "coefficient": coefficient,
            "std_error": standard_error,
            "z_score": z_score,
            "p_value": p_value,
        }
    )
