"""Discrimination metrics for binary PD models."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from ._validation import clean_binary_inputs


def auc(y_true: Iterable[object], y_prob: Iterable[object]) -> float:
    """Area under the ROC curve, with null pairs removed."""
    target, probability = clean_binary_inputs(y_true, y_prob, check_probability=False)
    return float(roc_auc_score(target, probability))


def gini(y_true: Iterable[object], y_prob: Iterable[object]) -> float:
    """Normalized Gini coefficient, ``2 * AUC - 1``."""
    return 2.0 * auc(y_true, y_prob) - 1.0


def ks_statistic(y_true: Iterable[object], y_prob: Iterable[object]) -> float:
    """Maximum separation between cumulative good and bad distributions."""
    target, probability = clean_binary_inputs(y_true, y_prob, check_probability=False)
    false_positive, true_positive, _ = roc_curve(target, probability)
    return float(np.max(true_positive - false_positive))


def bootstrap_gini(
    y_true: Iterable[object],
    y_prob: Iterable[object],
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 1_000,
    random_state: int | None = None,
) -> tuple[float, float, float]:
    """Return Gini and a stratified percentile bootstrap confidence interval."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    target, probability = clean_binary_inputs(y_true, y_prob)
    rng = np.random.default_rng(random_state)
    class_indices = [np.flatnonzero(target == value) for value in (0, 1)]
    estimates = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        sample = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        estimates[index] = 2.0 * roc_auc_score(target[sample], probability[sample]) - 1.0
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return gini(target, probability), float(lower), float(upper)
