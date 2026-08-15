from __future__ import annotations

from pathlib import Path

from examples.dummy_model import FEATURES, fit_and_score, load_dummy_data, run_validation


def test_dummy_data_and_model_are_executable(tmp_path: Path) -> None:
    """The bundled data must train, score, and produce a complete report."""
    data = load_dummy_data()
    assert len(data) == 48
    assert set(data["sample"]) == {"reference", "current"}
    assert set(data["default_12m"]) == {0, 1}

    model, reference, current = fit_and_score(data)
    assert model.classes_.tolist() == [0, 1]
    assert reference["predicted_pd"].between(0, 1).all()
    assert current["predicted_pd"].between(0, 1).all()
    assert set(FEATURES).issubset(reference)

    output = tmp_path / "dummy-report.html"
    report = run_validation(output_path=output)
    assert output.exists()
    assert "score_psi" in set(report.to_frame()["test"])
