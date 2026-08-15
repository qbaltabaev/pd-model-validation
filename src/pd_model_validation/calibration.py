"""Calibration and backtesting tools for PD estimates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import chi2, norm
from sklearn.metrics import brier_score_loss, log_loss

from ._validation import as_1d, clean_binary_inputs, quantile_bins


@dataclass(frozen=True)
class CalibrationSummary:
    """Core aggregate calibration measures for a PD model."""

    observed_default_rate: float
    mean_pd: float
    observed_expected_ratio: float
    observed_expected_ratio_lower: float
    observed_expected_ratio_upper: float
    brier_score: float
    log_loss: float
    intercept: float
    slope: float


@dataclass(frozen=True)
class CalibrationRegression:
    """Calibration intercept/slope estimates with fit diagnostics."""

    intercept: float
    slope: float
    intercept_std_error: float
    slope_std_error: float
    converged: bool
    iterations: int


def _wilson_interval(events: int, observations: int, confidence: float) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""
    if observations < 1:
        raise ValueError("observations must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    observed = events / observations
    z_value = norm.ppf(0.5 + confidence / 2.0)
    denominator = 1.0 + z_value**2 / observations
    center = (observed + z_value**2 / (2.0 * observations)) / denominator
    margin = (
        z_value
        * np.sqrt(observed * (1.0 - observed) / observations + z_value**2 / (4.0 * observations**2))
        / denominator
    )
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def calibration_regression(
    y_true: Iterable[object],
    y_prob: Iterable[object],
    *,
    epsilon: float = 1e-8,
    max_iter: int = 100,
    tolerance: float = 1e-10,
    raise_on_nonconvergence: bool = True,
) -> CalibrationRegression:
    """Estimate calibration intercept and slope with IRLS diagnostics.

    A constant predicted PD has no identifiable slope; in that case the
    intercept-only estimate is returned and slope fields are ``NaN``.
    Non-convergence raises ``RuntimeError`` by default. The orchestrator opts
    into a diagnostic return and marks the estimate N/A with convergence data.
    """
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be between 0 and 0.5")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    target, probability = clean_binary_inputs(y_true, y_prob)
    score = logit(np.clip(probability, epsilon, 1.0 - epsilon))
    if np.allclose(score, score[0]):
        observed = float(np.clip(target.mean(), epsilon, 1.0 - epsilon))
        intercept = float(logit(observed))
        intercept_se = float(np.sqrt(1.0 / (len(target) * observed * (1.0 - observed))))
        return CalibrationRegression(intercept, np.nan, intercept_se, np.nan, True, 1)

    design = np.column_stack([np.ones(len(score)), score])
    coefficient = np.array([0.0, 1.0])
    information = np.eye(2)
    for iteration in range(1, max_iter + 1):
        fitted = expit(design @ coefficient)
        weights = np.clip(fitted * (1.0 - fitted), epsilon, None)
        information = design.T @ (weights[:, None] * design)
        score_vector = design.T @ (target - fitted)
        step = np.linalg.solve(information, score_vector)
        coefficient += step
        if not np.isfinite(coefficient).all():
            raise RuntimeError("calibration regression produced non-finite coefficients")
        if np.max(np.abs(step)) < tolerance:
            covariance = np.linalg.inv(information)
            standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
            return CalibrationRegression(
                intercept=float(coefficient[0]),
                slope=float(coefficient[1]),
                intercept_std_error=float(standard_errors[0]),
                slope_std_error=float(standard_errors[1]),
                converged=True,
                iterations=iteration,
            )
    if raise_on_nonconvergence:
        raise RuntimeError(f"calibration regression did not converge after {max_iter} iterations")
    covariance = np.linalg.pinv(information)
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    return CalibrationRegression(
        intercept=float(coefficient[0]),
        slope=float(coefficient[1]),
        intercept_std_error=float(standard_errors[0]),
        slope_std_error=float(standard_errors[1]),
        converged=False,
        iterations=max_iter,
    )


def calibration_intercept_slope(
    y_true: Iterable[object], y_prob: Iterable[object], *, epsilon: float = 1e-8
) -> tuple[float, float]:
    """Estimate calibration-in-the-large and slope using logistic IRLS.

    A perfectly calibrated model has intercept 0 and slope 1.
    """
    estimate = calibration_regression(y_true, y_prob, epsilon=epsilon)
    return estimate.intercept, estimate.slope


def calibration_summary(
    y_true: Iterable[object],
    y_prob: Iterable[object],
    *,
    confidence: float = 0.95,
    raise_on_nonconvergence: bool = True,
    allow_single_class: bool = False,
) -> CalibrationSummary:
    """Calculate aggregate calibration, probability-error, and log-loss metrics."""
    target, probability = clean_binary_inputs(y_true, y_prob, allow_single_class=allow_single_class)
    observed = float(target.mean())
    expected = float(probability.mean())
    if len(np.unique(target)) == 2:
        regression = calibration_regression(
            target, probability, raise_on_nonconvergence=raise_on_nonconvergence
        )
        intercept, slope = regression.intercept, regression.slope
    else:
        intercept, slope = np.nan, np.nan
    ratio = observed / expected if expected > 0 else np.inf
    observed_lower, observed_upper = _wilson_interval(int(target.sum()), len(target), confidence)
    ratio_lower = observed_lower / expected if expected > 0 else np.inf
    ratio_upper = observed_upper / expected if expected > 0 else np.inf
    return CalibrationSummary(
        observed_default_rate=observed,
        mean_pd=expected,
        observed_expected_ratio=float(ratio),
        observed_expected_ratio_lower=float(ratio_lower),
        observed_expected_ratio_upper=float(ratio_upper),
        brier_score=float(brier_score_loss(target, probability)),
        log_loss=float(log_loss(target, probability, labels=[0, 1])),
        intercept=float(intercept),
        slope=float(slope),
    )


def calibration_table(
    y_true: Iterable[object],
    y_prob: Iterable[object],
    *,
    n_bins: int = 10,
    confidence: float = 0.95,
    allow_single_class: bool = False,
) -> pd.DataFrame:
    """Create an equal-frequency calibration table with Wilson intervals."""
    target, probability = clean_binary_inputs(y_true, y_prob, allow_single_class=allow_single_class)
    groups = quantile_bins(probability, n_bins)
    frame = pd.DataFrame({"target": target, "pd": probability, "bin": groups})
    result = frame.groupby("bin", observed=True).agg(
        n=("target", "size"),
        defaults=("target", "sum"),
        observed_rate=("target", "mean"),
        mean_pd=("pd", "mean"),
        min_pd=("pd", "min"),
        max_pd=("pd", "max"),
    )
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    z_value = norm.ppf(0.5 + confidence / 2.0)
    denominator = 1.0 + z_value**2 / result["n"]
    center = (result["observed_rate"] + z_value**2 / (2.0 * result["n"])) / denominator
    margin = (
        z_value
        * np.sqrt(
            result["observed_rate"] * (1.0 - result["observed_rate"]) / result["n"]
            + z_value**2 / (4.0 * result["n"] ** 2)
        )
        / denominator
    )
    result["observed_lower"] = center - margin
    result["observed_upper"] = center + margin
    result["expected_defaults"] = result["mean_pd"] * result["n"]
    result["expected_non_defaults"] = (1.0 - result["mean_pd"]) * result["n"]
    result = result.reset_index()
    result.attrs["requested_bins"] = n_bins
    result.attrs["effective_bins"] = len(result)
    result.attrs["confidence"] = confidence
    return result


def calibration_by_group(
    y_true: Iterable[object],
    y_prob: Iterable[object],
    groups: Iterable[object],
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Backtest calibration by rating grade, pool, or business segment.

    The table includes observed and expected defaults, Wilson intervals, O/E,
    and a normal approximation to the Poisson-binomial outcome test. The
    approximation should be interpreted cautiously for sparse groups.
    """
    target = as_1d(y_true, "y_true")
    probability = as_1d(y_prob, "y_prob", dtype=float)
    group = as_1d(groups, "groups")
    if not (len(target) == len(probability) == len(group)):
        raise ValueError("y_true, y_prob, and groups must have the same length")
    numeric_target = pd.to_numeric(pd.Series(target), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(numeric_target) & np.isfinite(probability)
    if not set(np.unique(numeric_target[mask])).issubset({0.0, 1.0}):
        raise ValueError("y_true must contain only 0 and 1")
    if np.any((probability[mask] < 0) | (probability[mask] > 1)):
        raise ValueError("y_prob must be between 0 and 1")
    frame = pd.DataFrame(
        {
            "target": numeric_target[mask].astype(int),
            "pd": probability[mask],
            "group": pd.Series(group[mask]).astype("string").fillna("<MISSING>"),
        }
    )
    if frame.empty:
        raise ValueError("no complete target/probability pairs remain")

    rows = []
    for label, subset in frame.groupby("group", observed=True, sort=True):
        observations = len(subset)
        defaults = int(subset["target"].sum())
        expected_defaults = float(subset["pd"].sum())
        mean_pd = expected_defaults / observations
        lower, upper = _wilson_interval(defaults, observations, confidence)
        variance = float((subset["pd"] * (1.0 - subset["pd"])).sum())
        z_score = (defaults - expected_defaults) / np.sqrt(variance) if variance > 0 else np.nan
        rows.append(
            {
                "group": str(label),
                "n": observations,
                "defaults": defaults,
                "expected_defaults": expected_defaults,
                "observed_rate": defaults / observations,
                "mean_pd": mean_pd,
                "observed_expected_ratio": defaults / expected_defaults
                if expected_defaults > 0
                else np.inf,
                "observed_lower": lower,
                "observed_upper": upper,
                "z_score": z_score,
                "p_value": float(2.0 * norm.sf(abs(z_score))) if np.isfinite(z_score) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def hosmer_lemeshow(
    y_true: Iterable[object], y_prob: Iterable[object], *, n_bins: int = 10
) -> tuple[float, float, int]:
    """Hosmer-Lemeshow statistic, p-value, and effective degrees of freedom."""
    table = calibration_table(y_true, y_prob, n_bins=n_bins)
    if len(table) < 3:
        return np.nan, np.nan, 0
    observed = table["defaults"].to_numpy(dtype=float)
    total = table["n"].to_numpy(dtype=float)
    expected = table["mean_pd"].to_numpy(dtype=float) * total
    denominator = np.clip(expected * (1.0 - expected / total), 1e-12, None)
    statistic = float(np.sum((observed - expected) ** 2 / denominator))
    degrees = max(len(table) - 2, 1)
    return statistic, float(chi2.sf(statistic, degrees)), degrees
