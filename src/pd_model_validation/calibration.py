"""Calibration and backtesting tools for PD estimates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import chi2, norm
from sklearn.metrics import brier_score_loss, log_loss

from ._validation import clean_binary_inputs, quantile_bins


@dataclass(frozen=True)
class CalibrationSummary:
    observed_default_rate: float
    mean_pd: float
    observed_expected_ratio: float
    brier_score: float
    log_loss: float
    intercept: float
    slope: float


def calibration_intercept_slope(
    y_true: Iterable[object], y_prob: Iterable[object], *, epsilon: float = 1e-8
) -> tuple[float, float]:
    """Estimate calibration-in-the-large and slope using logistic IRLS.

    A perfectly calibrated model has intercept 0 and slope 1.
    """
    target, probability = clean_binary_inputs(y_true, y_prob)
    score = logit(np.clip(probability, epsilon, 1.0 - epsilon))
    design = np.column_stack([np.ones(len(score)), score])
    coefficient = np.array([0.0, 1.0])
    for _ in range(100):
        fitted = expit(design @ coefficient)
        weights = np.clip(fitted * (1.0 - fitted), epsilon, None)
        working = design @ coefficient + (target - fitted) / weights
        updated = np.linalg.lstsq(
            design * np.sqrt(weights[:, None]), working * np.sqrt(weights), rcond=None
        )[0]
        if np.max(np.abs(updated - coefficient)) < 1e-10:
            coefficient = updated
            break
        coefficient = updated
    return float(coefficient[0]), float(coefficient[1])


def calibration_summary(y_true: Iterable[object], y_prob: Iterable[object]) -> CalibrationSummary:
    target, probability = clean_binary_inputs(y_true, y_prob)
    observed = float(target.mean())
    expected = float(probability.mean())
    intercept, slope = calibration_intercept_slope(target, probability)
    ratio = observed / expected if expected > 0 else np.inf
    return CalibrationSummary(
        observed_default_rate=observed,
        mean_pd=expected,
        observed_expected_ratio=float(ratio),
        brier_score=float(brier_score_loss(target, probability)),
        log_loss=float(log_loss(target, probability, labels=[0, 1])),
        intercept=intercept,
        slope=slope,
    )


def calibration_table(
    y_true: Iterable[object], y_prob: Iterable[object], *, n_bins: int = 10
) -> pd.DataFrame:
    """Create an equal-frequency calibration table with Wilson intervals."""
    target, probability = clean_binary_inputs(y_true, y_prob)
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
    z_value = norm.ppf(0.975)
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
    result["observed_lower_95"] = center - margin
    result["observed_upper_95"] = center + margin
    return result.reset_index()


def hosmer_lemeshow(
    y_true: Iterable[object], y_prob: Iterable[object], *, n_bins: int = 10
) -> tuple[float, float, int]:
    """Hosmer-Lemeshow statistic, p-value, and effective degrees of freedom."""
    table = calibration_table(y_true, y_prob, n_bins=n_bins)
    observed = table["defaults"].to_numpy(dtype=float)
    total = table["n"].to_numpy(dtype=float)
    expected = table["mean_pd"].to_numpy(dtype=float) * total
    denominator = np.clip(expected * (1.0 - expected / total), 1e-12, None)
    statistic = float(np.sum((observed - expected) ** 2 / denominator))
    degrees = max(len(table) - 2, 1)
    return statistic, float(chi2.sf(statistic, degrees)), degrees
