import json

import pandas as pd

from pd_model_validation import PDValidator, Status, Thresholds, ValidationConfig
from pd_model_validation.cli import main


def test_threshold_classification() -> None:
    higher = Thresholds(0.4, 0.2, "higher")
    lower = Thresholds(0.1, 0.25, "lower")
    assert higher.classify(0.5) is Status.GREEN
    assert higher.classify(0.3) is Status.AMBER
    assert lower.classify(0.3) is Status.RED


def test_complete_validator_report(reference, current, tmp_path) -> None:
    config = ValidationConfig(min_period_observations=5, bins=5)
    report = PDValidator(config).validate(
        reference,
        current=current,
        target="target",
        probability="pd",
        features=["x1", "x2", "category"],
        date="date",
    )
    results = report.to_frame()
    assert {"gini", "score_psi", "relative_gini_degradation", "feature_psi"}.issubset(
        set(results["test"])
    )
    assert {"feature_stability", "reference_performance_by_period", "current_calibration"}.issubset(
        report.tables
    )
    assert int(report.summary().sum()) == len(results)

    json_path = tmp_path / "report.json"
    html_path = tmp_path / "report.html"
    csv_path = tmp_path / "report.csv"
    payload = report.to_json(json_path)
    report.to_html(html_path)
    report.to_csv(csv_path)
    assert json.loads(payload)["results"]
    assert "PD validation report" in html_path.read_text()
    assert len(pd.read_csv(csv_path)) == len(results)


def test_cli_writes_report(reference, tmp_path) -> None:
    csv_path = tmp_path / "reference.csv"
    output_path = tmp_path / "output.html"
    reference.to_csv(csv_path, index=False)
    exit_code = main(
        [str(csv_path), "--target", "target", "--probability", "pd", "--output", str(output_path)]
    )
    assert exit_code == 0
    assert output_path.exists()
