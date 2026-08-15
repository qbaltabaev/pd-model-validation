import numpy as np
import pytest

from pd_model_validation import (
    calibration_by_group,
    calibration_regression,
    calibration_summary,
    calibration_table,
    hosmer_lemeshow,
)


def test_calibration_summary_has_expected_fields(reference) -> None:
    summary = calibration_summary(reference["target"], reference["pd"])
    assert 0 <= summary.observed_default_rate <= 1
    assert 0 <= summary.mean_pd <= 1
    assert summary.brier_score > 0
    assert summary.log_loss > 0
    assert np.isfinite(summary.intercept)
    assert np.isfinite(summary.slope)


def test_calibration_table_preserves_observations(reference) -> None:
    table = calibration_table(reference["target"], reference["pd"], n_bins=7)
    assert table["n"].sum() == len(reference)
    assert len(table) == 7
    assert (table["observed_lower"] <= table["observed_upper"]).all()


def test_calibration_table_keeps_tied_probabilities_together() -> None:
    probability = [0.2] * 4 + [0.8] * 4
    table = calibration_table([0, 1, 0, 1, 0, 1, 0, 1], probability, n_bins=4)
    assert len(table) == 2
    assert (table["min_pd"] == table["max_pd"]).all()
    assert table.attrs["effective_bins"] == 2


def test_constant_prediction_has_unidentifiable_slope() -> None:
    estimate = calibration_regression([0, 0, 1, 1], [0.2] * 4)
    assert estimate.converged
    assert np.isnan(estimate.slope)


def test_calibration_regression_reports_non_convergence(reference) -> None:
    with pytest.raises(RuntimeError, match="did not converge"):
        calibration_regression(reference["target"], reference["pd"], max_iter=1)


def test_hosmer_lemeshow_returns_valid_p_value(reference) -> None:
    statistic, p_value, degrees = hosmer_lemeshow(reference["target"], reference["pd"])
    assert statistic >= 0
    assert 0 <= p_value <= 1
    assert degrees == 8


def test_hosmer_lemeshow_is_na_with_too_few_unique_scores() -> None:
    statistic, p_value, degrees = hosmer_lemeshow([0, 0, 1, 1], [0.2] * 4)
    assert np.isnan(statistic)
    assert np.isnan(p_value)
    assert degrees == 0


def test_calibration_by_group_returns_grade_level_evidence(reference) -> None:
    grades = np.where(reference["pd"] < reference["pd"].median(), "low", "high")
    table = calibration_by_group(reference["target"], reference["pd"], grades)
    assert set(table["group"]) == {"high", "low"}
    assert table["n"].sum() == len(reference)
    assert table["p_value"].between(0, 1).all()


def test_grouped_calibration_handles_zero_expected_and_rejects_misalignment() -> None:
    table = calibration_by_group([0, 1], [0.0, 0.0], ["grade", "grade"])
    assert np.isinf(table.loc[0, "observed_expected_ratio"])
    assert np.isnan(table.loc[0, "p_value"])
    with pytest.raises(ValueError, match="same length"):
        calibration_by_group([0, 1], [0.1, 0.9], ["grade"])


def test_calibration_rejects_values_outside_probability_range() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        calibration_summary([0, 1], [-0.1, 0.9])
