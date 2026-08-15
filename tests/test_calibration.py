import numpy as np
import pytest

from pd_model_validation import calibration_summary, calibration_table, hosmer_lemeshow


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
    assert (table["observed_lower_95"] <= table["observed_upper_95"]).all()


def test_hosmer_lemeshow_returns_valid_p_value(reference) -> None:
    statistic, p_value, degrees = hosmer_lemeshow(reference["target"], reference["pd"])
    assert statistic >= 0
    assert 0 <= p_value <= 1
    assert degrees == 8


def test_calibration_rejects_values_outside_probability_range() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        calibration_summary([0, 1], [-0.1, 0.9])
