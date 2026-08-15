"""Fit and validate a front-book credit-card underwriting PD model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pd_model_validation import PDValidator, ValidationConfig, ValidationReport

DATA_PATH = Path(__file__).parent / "data" / "underwriting_pd_data.csv"
TARGET = "pd90_12m"
NUMERIC_FEATURES = [
    "annual_income",
    "employment_length_years",
    "requested_credit_limit",
    "number_of_credit_cards",
    "credit_card_balance",
    "credit_card_utilization",
    "mortgage_payment",
    "total_loan_balance",
    "auto_loan_payment",
    "deposit_balance",
    "delinquencies_last_3_months",
    "delinquencies_last_6_months",
    "delinquencies_last_12_months",
    "recent_credit_inquiries",
    "months_since_oldest_credit_line",
    "debt_to_income_ratio",
    "bankruptcy_flag",
]
CATEGORICAL_FEATURES = ["employment_status", "housing_status"]
FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


def load_underwriting_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the synthetic reference and current underwriting applications."""
    return pd.read_csv(path, parse_dates=["application_date"])


def build_model() -> Pipeline:
    """Build a leakage-safe preprocessing and logistic-regression pipeline."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocessing), ("model", LogisticRegression(max_iter=1_000))])


def fit_and_score(
    data: pd.DataFrame,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit development data and score distinct validation and OOT samples.

    The checked-in ``reference`` population is split chronologically: the
    earliest 75% trains the pipeline and the latest 25% is a holdout validation
    sample. The later ``current`` population remains out of time.
    """
    reference = data.loc[data["sample"] == "reference"].copy()
    current = data.loc[data["sample"] == "current"].copy()
    split_at = int(len(reference) * 0.75)
    development = reference.iloc[:split_at].copy()
    validation = reference.iloc[split_at:].copy()
    model = build_model().fit(development[FEATURES], development[TARGET])
    development["predicted_pd"] = model.predict_proba(development[FEATURES])[:, 1]
    validation["predicted_pd"] = model.predict_proba(validation[FEATURES])[:, 1]
    current["predicted_pd"] = model.predict_proba(current[FEATURES])[:, 1]
    return model, development, validation, current


def run_validation(
    data_path: str | Path = DATA_PATH,
    output_path: str | Path = "underwriting-pd-validation.html",
) -> ValidationReport:
    """Fit the underwriting model, validate it, and write an HTML report."""
    _, development, validation, current = fit_and_score(load_underwriting_data(data_path))
    validator = PDValidator(ValidationConfig(bins=5, min_period_observations=3))
    report = validator.validate(
        validation,
        current=current,
        target=TARGET,
        probability="predicted_pd",
        features=FEATURES,
        date="application_date",
        metadata={
            "model_id": "synthetic-front-book-pd",
            "model_version": "example-1",
            "development_rows": len(development),
            "validation_role": "chronological_holdout",
            "current_role": "out_of_time",
        },
    )
    report.to_html(output_path, title="Front-book credit-card PD validation")
    return report


def main() -> int:
    """Run the underwriting model example from the command line."""
    data = load_underwriting_data()
    report = run_validation()
    print(
        f"Applications: {len(data)} | defaults: {int(data[TARGET].sum())} "
        f"| default rate: {data[TARGET].mean():.1%}"
    )
    print(report.to_frame().to_string(index=False))
    print("Report written to underwriting-pd-validation.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
