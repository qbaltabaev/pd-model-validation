"""Fit and validate a logistic PD model on the bundled synthetic dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pd_model_validation import PDValidator, ValidationConfig, ValidationReport

DATA_PATH = Path(__file__).parent / "data" / "dummy_pd_data.csv"
NUMERIC_FEATURES = ["utilization_ratio", "account_age_months", "delinquency_count"]
CATEGORICAL_FEATURES = ["customer_type"]
FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


def load_dummy_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the compact, fully synthetic development and current samples."""
    return pd.read_csv(path, parse_dates=["observation_date"])


def build_model() -> Pipeline:
    """Build a leakage-safe preprocessing and logistic-regression pipeline."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )
    return Pipeline([("preprocess", preprocessing), ("model", LogisticRegression(max_iter=1_000))])


def fit_and_score(data: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    """Fit on the reference rows and add predicted PDs to both samples."""
    reference = data.loc[data["sample"] == "reference"].copy()
    current = data.loc[data["sample"] == "current"].copy()
    model = build_model().fit(reference[FEATURES], reference["default_12m"])
    reference["predicted_pd"] = model.predict_proba(reference[FEATURES])[:, 1]
    current["predicted_pd"] = model.predict_proba(current[FEATURES])[:, 1]
    return model, reference, current


def run_validation(
    data_path: str | Path = DATA_PATH,
    output_path: str | Path = "dummy-pd-validation.html",
) -> ValidationReport:
    """Fit the dummy model, validate it, and write an HTML evidence report."""
    _, reference, current = fit_and_score(load_dummy_data(data_path))
    validator = PDValidator(ValidationConfig(bins=4, min_period_observations=4))
    report = validator.validate(
        reference,
        current=current,
        target="default_12m",
        probability="predicted_pd",
        features=FEATURES,
        date="observation_date",
    )
    report.to_html(output_path, title="Synthetic PD model validation")
    return report


def main() -> int:
    """Run the dummy model example from the command line."""
    report = run_validation()
    print(report.to_frame().to_string(index=False))
    print("Report written to dummy-pd-validation.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
