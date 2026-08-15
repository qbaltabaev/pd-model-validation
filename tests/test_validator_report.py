import json

import numpy as np
import pandas as pd
import pytest

from pd_model_validation import (
    InputValidationError,
    PDValidator,
    Status,
    Thresholds,
    ValidationConfig,
    ValidationReport,
    ValidationResult,
)
from pd_model_validation.cli import main


def test_threshold_classification() -> None:
    higher = Thresholds(0.4, 0.2, "higher")
    lower = Thresholds(0.1, 0.25, "lower")
    assert higher.classify(0.5) is Status.GREEN
    assert higher.classify(0.3) is Status.AMBER
    assert lower.classify(0.3) is Status.RED
    with pytest.raises(ValueError, match="direction"):
        Thresholds(0.1, 0.2, "sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="green must"):
        Thresholds(0.1, 0.2, "higher")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bins": 1}, "bins"),
        ({"min_period_observations": 0}, "min_period"),
        ({"confidence_level": 1.0}, "confidence"),
        ({"bootstrap_samples": 0}, "bootstrap"),
        ({"threshold_profile": " "}, "threshold_profile"),
    ],
)
def test_validation_config_rejects_invalid_settings(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ValidationConfig(**kwargs)


def test_complete_validator_report(reference, current, tmp_path) -> None:
    config = ValidationConfig(min_period_observations=5, bins=5)
    report = PDValidator(config).validate(
        reference,
        current=current,
        target="target",
        probability="pd",
        features=["x1", "x2", "category"],
        date="date",
        group="category",
    )
    results = report.to_frame()
    assert {"gini", "score_psi", "relative_gini_degradation", "feature_psi"}.issubset(
        set(results["test"])
    )
    assert {"feature_stability", "reference_performance_by_period", "current_calibration"}.issubset(
        report.tables
    )
    assert "score_stability" in report.tables
    assert {"reference_calibration_by_group", "current_calibration_by_group"}.issubset(
        report.tables
    )
    assert "group_observed_expected_ratio_deviation" in set(results["test"])
    assert report.metadata["package_version"]
    current_period_results = results.loc[
        (results["test"] == "period_gini") & (results["segment"] == "current")
    ]
    assert len(current_period_results) == len(report.tables["current_performance_by_period"])
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


def test_validator_rejects_invalid_probabilities_before_metrics(reference) -> None:
    reference.loc[0, "pd"] = 1.1
    with pytest.raises(InputValidationError, match="1 invalid probability"):
        PDValidator().validate(reference, target="target", probability="pd")


def test_zero_expected_defaults_is_a_red_calibration_failure() -> None:
    frame = pd.DataFrame({"target": [0, 0, 1, 1], "pd": [0.0] * 4})
    report = PDValidator(ValidationConfig(bootstrap_samples=10)).validate(
        frame, target="target", probability="pd"
    )
    result = report.to_frame().set_index("test").loc["observed_expected_ratio_deviation"]
    assert result["status"] == "RED"
    assert result["reason"] == "mean predicted PD is zero"


def test_validator_reports_missing_pairs(reference) -> None:
    reference.loc[0, "pd"] = np.nan
    report = PDValidator().validate(reference, target="target", probability="pd")
    result = report.to_frame().set_index("test").loc["excluded_incomplete_pair_count"]
    assert result["value"] == 1
    assert result["details"]["effective_observations"] == len(reference) - 1


def test_unlabeled_current_sample_runs_drift_only(reference, current) -> None:
    report = PDValidator().validate(
        reference,
        current=current.drop(columns="target"),
        target="target",
        probability="pd",
        features=["x1", "category"],
    )
    tests = set(report.to_frame()["test"])
    assert {"score_psi", "feature_psi", "current_outcome_available"}.issubset(tests)
    assert "relative_gini_degradation" not in tests
    assert report.metadata["samples"]["current"]["outcome_available"] is False


def test_all_missing_current_outcome_is_treated_as_immature(reference, current) -> None:
    current["target"] = np.nan
    report = PDValidator().validate(reference, current=current, target="target", probability="pd")
    assert "current_outcome_available" in set(report.to_frame()["test"])
    assert "current_calibration" not in report.tables


def test_single_class_current_reports_na_discrimination(reference, current) -> None:
    current["target"] = 0
    report = PDValidator(ValidationConfig(bootstrap_samples=10)).validate(
        reference, current=current, target="target", probability="pd"
    )
    current_results = report.to_frame().loc[lambda frame: frame["segment"] == "current"]
    gini_result = current_results.set_index("test").loc["gini"]
    assert gini_result["status"] == "N/A"
    assert gini_result["reason"] == "sample contains only one class"
    degradation = report.to_frame().set_index("test").loc["relative_gini_degradation"]
    assert degradation["status"] == "N/A"


def test_monitor_drift_does_not_require_any_target(reference, current) -> None:
    report = PDValidator().monitor_drift(
        reference.drop(columns="target"),
        current.drop(columns="target"),
        probability="pd",
        features=["x1"],
    )
    assert set(report.to_frame()["test"]) == {"score_psi", "feature_psi"}


def test_monitor_drift_rejects_invalid_scores(reference, current) -> None:
    current = current.drop(columns="target")
    current.loc[0, "pd"] = -0.01
    with pytest.raises(InputValidationError, match="current contains 1 invalid probability"):
        PDValidator().monitor_drift(reference, current, probability="pd")


def test_json_export_normalizes_non_finite_values() -> None:
    report = ValidationReport(
        results=[ValidationResult("undefined", np.nan, Status.NOT_APPLICABLE)],
        tables={"values": pd.DataFrame({"value": [np.inf]})},
    )
    payload = report.to_json()
    assert "NaN" not in payload and "Infinity" not in payload
    parsed = json.loads(payload)
    assert parsed["results"][0]["value"] is None
    assert parsed["tables"]["values"][0]["value"] is None


def test_empty_report_and_failure_helpers() -> None:
    assert ValidationReport([]).summary().empty
    report = ValidationReport(
        [
            ValidationResult("review", 1.0, Status.AMBER),
            ValidationResult("pass", 1.0, Status.GREEN),
        ]
    )
    assert not report.has_failures()
    assert report.has_failures(include_amber=True)


def test_cli_writes_report(reference, tmp_path) -> None:
    csv_path = tmp_path / "reference.csv"
    output_path = tmp_path / "output.html"
    reference.to_csv(csv_path, index=False)
    exit_code = main(
        [
            str(csv_path),
            "--target",
            "target",
            "--probability",
            "pd",
            "--output",
            str(output_path),
            "--fail-on",
            "never",
        ]
    )
    assert exit_code == 0
    assert output_path.exists()


def test_cli_writes_strict_json_and_fails_red_gate(reference, tmp_path) -> None:
    csv_path = tmp_path / "reference.csv"
    output_path = tmp_path / "output.json"
    reference = reference.copy()
    reference["pd"] = 1.0 - reference["pd"]
    reference.to_csv(csv_path, index=False)
    exit_code = main(
        [
            str(csv_path),
            "--target",
            "target",
            "--probability",
            "pd",
            "--output",
            str(output_path),
            "--model-id",
            "test-model",
        ]
    )
    assert exit_code == 2
    payload = json.loads(output_path.read_text())
    assert payload["metadata"]["context"]["model_id"] == "test-model"
