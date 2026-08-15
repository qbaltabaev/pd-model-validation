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
    """Measure apparent in-sample standalone discrimination via fitted WoE.

    Use a holdout sample or cross-fitting for unbiased performance estimation;
    this helper is intended as a feature-screening diagnostic.
    """
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
        correlations = matrix.loc[feature].dropna()
        if correlations.empty:
            rows.append(
                {
                    "feature": feature,
                    "max_abs_correlation": np.nan,
                    "correlated_with": None,
                }
            )
            continue
        peer = correlations.idxmax()
        rows.append(
            {
                "feature": feature,
                "max_abs_correlation": float(matrix.loc[feature, peer]),
                "correlated_with": peer,
            }
        )
    return pd.DataFrame(rows).sort_values("max_abs_correlation", ascending=False, ignore_index=True)


def variance_inflation_factors(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Calculate VIFs using per-feature auxiliary regressions.

    Exact linear dependence returns infinite VIF instead of being hidden by a
    matrix pseudoinverse. Constant features remain undefined and are rejected.
    """
    require_columns(frame, features)
    data = frame[list(features)].apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        raise ValueError("no complete numeric feature rows remain")
    standardized = (data - data.mean()) / data.std(ddof=0).replace(0, np.nan)
    if standardized.isna().any().any():
        constant = standardized.columns[standardized.isna().all()].tolist()
        raise ValueError(f"constant features have undefined VIF: {', '.join(constant)}")
    values = standardized.to_numpy(dtype=float)
    rows = []
    for index, feature in enumerate(features):
        response = values[:, index]
        predictors = np.delete(values, index, axis=1)
        if predictors.shape[1] == 0:
            value = 1.0
        else:
            design = np.column_stack([np.ones(len(predictors)), predictors])
            fitted = design @ np.linalg.lstsq(design, response, rcond=None)[0]
            residual_sum_squares = float(np.sum((response - fitted) ** 2))
            total_sum_squares = float(np.sum((response - response.mean()) ** 2))
            if residual_sum_squares <= np.finfo(float).eps * max(total_sum_squares, 1.0):
                value = np.inf
            else:
                r_squared = 1.0 - residual_sum_squares / total_sum_squares
                value = 1.0 / max(1.0 - r_squared, np.finfo(float).eps)
        rows.append({"feature": feature, "vif": value})
    return pd.DataFrame(rows)


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
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("feature design matrix is rank deficient")
    converged = False
    iterations = 0
    for iteration in range(1, max_iter + 1):
        iterations = iteration
        fitted = expit(design @ coefficient)
        weights = np.clip(fitted * (1.0 - fitted), 1e-10, None)
        information = design.T @ (weights[:, None] * design)
        score = design.T @ (y_true - fitted)
        step = np.linalg.pinv(information) @ score
        coefficient += step
        if np.max(np.abs(step)) < 1e-9:
            converged = True
            break
    if not converged:
        raise RuntimeError("logistic coefficient fit did not converge")
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
            "converged": converged,
            "iterations": iterations,
        }
    )
