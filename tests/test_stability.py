import numpy as np
import pandas as pd

from pd_model_validation import (
    feature_stability,
    gini_degradation,
    performance_by_period,
    population_stability_index,
)


def test_psi_is_zero_for_identical_samples() -> None:
    values = np.arange(100)
    assert np.isclose(population_stability_index(values, values), 0.0)


def test_psi_detects_numeric_and_categorical_shift() -> None:
    rng = np.random.default_rng(1)
    expected = rng.normal(0, 1, 2_000)
    actual = rng.normal(1, 1, 2_000)
    assert population_stability_index(expected, actual) > 0.25
    assert population_stability_index(["a"] * 90 + ["b"] * 10, ["a"] * 50 + ["b"] * 50) > 0.25


def test_feature_stability_is_ranked(reference, current) -> None:
    table = feature_stability(reference, current, ["x1", "x2", "category"])
    assert list(table.columns) == ["feature", "psi"]
    assert table["psi"].is_monotonic_decreasing


def test_performance_by_period_handles_single_class_period() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02"]),
            "target": [0, 0, 0, 1],
            "pd": [0.1, 0.2, 0.3, 0.8],
        }
    )
    table = performance_by_period(
        frame, target="target", probability="pd", date="date", min_observations=2
    )
    assert np.isnan(table.loc[0, "gini"])
    assert table.loc[1, "gini"] == 1.0


def test_relative_gini_degradation() -> None:
    target = [0, 0, 1, 1]
    assert gini_degradation(target, [0.1, 0.2, 0.8, 0.9], target, [0.1, 0.8, 0.2, 0.9]) == 0.5
