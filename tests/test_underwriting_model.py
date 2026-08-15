from __future__ import annotations

from pathlib import Path

import pandas as pd

from examples.generate_underwriting_data import generate_dataset
from examples.underwriting_model import (
    FEATURES,
    TARGET,
    fit_and_score,
    load_underwriting_data,
    run_validation,
)


def test_underwriting_data_and_model_are_executable(tmp_path: Path) -> None:
    """The underwriting data must train, score, and produce a complete report."""
    data = load_underwriting_data()
    pd.testing.assert_frame_equal(data, generate_dataset())
    assert len(data) == 160
    assert data["application_id"].is_unique
    assert data["application_date"].is_monotonic_increasing
    assert data["sample"].value_counts().to_dict() == {"reference": 120, "current": 40}
    assert set(data[TARGET]) == {0, 1}
    assert "predicted_pd" not in data
    assert (data["delinquencies_last_3_months"] <= data["delinquencies_last_6_months"]).all()
    assert (data["delinquencies_last_6_months"] <= data["delinquencies_last_12_months"]).all()
    assert set(data["bankruptcy_flag"]) <= {0, 1}

    model, reference, current = fit_and_score(data)
    assert model.classes_.tolist() == [0, 1]
    assert set(reference[TARGET]) == set(current[TARGET]) == {0, 1}
    assert reference["predicted_pd"].between(0, 1).all()
    assert current["predicted_pd"].between(0, 1).all()
    assert set(FEATURES).issubset(reference)

    output = tmp_path / "underwriting-report.html"
    report = run_validation(output_path=output)
    assert output.exists()
    assert "score_psi" in set(report.to_frame()["test"])
